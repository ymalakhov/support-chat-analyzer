#!/usr/bin/env python3
"""
analyze.py — Chat quality analyzer.

Reads generated dialogues from output/chats.json, sends each to an LLM for
quality assessment, and writes structured analysis results to
output/analysis.json.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import (
    AGENT_MISTAKES,
    ANALYSIS_OUTPUT_PATH,
    ANALYZE_PROMPT_PATH,
    CATEGORIES,
    CHATS_OUTPUT_PATH,
    MODEL,
    SATISFACTION_LEVELS,
    SEED,
    TEMPERATURE,
    TOP_P,
)

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv()


def load_prompt_template() -> str:
    """Read the analysis prompt template from disk."""
    if not ANALYZE_PROMPT_PATH.exists():
        print(f"ERROR: Prompt template not found at {ANALYZE_PROMPT_PATH}")
        sys.exit(1)
    return ANALYZE_PROMPT_PATH.read_text(encoding="utf-8")

def format_dialogue_for_prompt(chat: dict) -> str:
    """Convert a chat dict into a readable dialogue string for the prompt."""
    lines = [
        f"Chat ID: {chat['chat_id']}",
        f"Category (metadata): {chat.get('category', 'unknown')}",
        f"Scenario type (metadata): {chat.get('scenario_type', 'unknown')}",
        "",
        "--- Dialogue ---",
    ]
    for msg in chat.get("messages", []):
        role = msg.get("role", "unknown").upper()
        text = msg.get("text", "")
        lines.append(f"[{role}]: {text}")
    lines.append("--- End of Dialogue ---")
    return "\n".join(lines)


def analyze_chat(
    client: OpenAI,
    prompt_template: str,
    chat: dict,
    model: str,
    seed: int,
) -> dict:
    """Call the LLM to analyze a single chat dialogue."""
    dialogue_text = format_dialogue_for_prompt(chat)
    user_prompt = prompt_template.format(dialogue=dialogue_text)

    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=seed,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert customer support quality analyst. "
                    "Always respond with valid JSON only."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content
    analysis = json.loads(raw)

    # Ensure chat_id matches
    analysis["chat_id"] = chat["chat_id"]

    return analysis


def validate_analysis(analysis: dict) -> list[str]:
    """Basic validation of an analysis result. Returns list of issues."""
    issues: list[str] = []
    required_keys = [
        "chat_id",
        "intent",
        "customer_satisfaction",
        "quality_score",
        "agent_mistakes",
        "reasoning",
    ]
    for key in required_keys:
        if key not in analysis:
            issues.append(f"Missing key: {key}")

    if "intent" in analysis:
        valid_intents = CATEGORIES + ["other"]
        if analysis["intent"] not in valid_intents:
            issues.append(f"Invalid intent: {analysis['intent']}")

    if "customer_satisfaction" in analysis:
        if analysis["customer_satisfaction"] not in SATISFACTION_LEVELS:
            issues.append(
                f"Invalid satisfaction: {analysis['customer_satisfaction']}"
            )

    if "quality_score" in analysis:
        score = analysis["quality_score"]
        if not isinstance(score, int) or score < 1 or score > 5:
            issues.append(f"Invalid quality_score: {score} (must be 1-5)")

    if "agent_mistakes" in analysis:
        if not isinstance(analysis["agent_mistakes"], list):
            issues.append("agent_mistakes must be a list")
        else:
            for mistake in analysis["agent_mistakes"]:
                if mistake not in AGENT_MISTAKES:
                    issues.append(f"Unknown agent_mistake: {mistake}")

    return issues


def print_summary(analyses: list[dict]) -> None:
    """Print a summary table of the analysis results."""
    if not analyses:
        print("No analyses to summarize.")
        return

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    # Intent distribution
    intent_counts = Counter(a.get("intent", "?") for a in analyses)
    print("\nIntent Distribution:")
    for intent, count in sorted(intent_counts.items()):
        bar = "█" * count
        print(f"  {intent:<22} {count:>3}  {bar}")

    # Satisfaction distribution
    sat_counts = Counter(a.get("customer_satisfaction", "?") for a in analyses)
    print("\nCustomer Satisfaction:")
    for level in SATISFACTION_LEVELS:
        count = sat_counts.get(level, 0)
        bar = "█" * count
        print(f"  {level:<22} {count:>3}  {bar}")

    # Quality score stats
    scores = [
        a["quality_score"]
        for a in analyses
        if isinstance(a.get("quality_score"), int)
    ]
    if scores:
        avg = sum(scores) / len(scores)
        score_counts = Counter(scores)
        print(f"\nQuality Score (avg: {avg:.2f}):")
        for score in range(1, 6):
            count = score_counts.get(score, 0)
            bar = "█" * count
            print(f"  Score {score}  {count:>3}  {bar}")

    # Agent mistakes
    all_mistakes: list[str] = []
    for a in analyses:
        all_mistakes.extend(a.get("agent_mistakes", []))
    if all_mistakes:
        mistake_counts = Counter(all_mistakes)
        print("\nAgent Mistakes Detected:")
        for mistake, count in sorted(
            mistake_counts.items(), key=lambda x: -x[1]
        ):
            bar = "█" * count
            print(f"  {mistake:<28} {count:>3}  {bar}")
    else:
        print("\nAgent Mistakes Detected: None")

    # Chats with no mistakes
    perfect = sum(1 for a in analyses if not a.get("agent_mistakes"))
    print(f"\nChats with no agent mistakes: {perfect}/{len(analyses)}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze support chat dialogues for quality assessment."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=CHATS_OUTPUT_PATH,
        help=f"Input JSON file with chats (default: {CHATS_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ANALYSIS_OUTPUT_PATH,
        help=f"Output JSON file path (default: {ANALYSIS_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help=f"LLM model to use (default: {MODEL})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for determinism (default: {SEED})",
    )
    args = parser.parse_args()

    # Load input chats
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Run generate.py first to create the chat dataset.")
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    chats = data.get("chats", [])
    if not chats:
        print("ERROR: No chats found in input file.")
        sys.exit(1)

    print(f"Loaded {len(chats)} chats from {args.input}")

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load prompt template
    prompt_template = load_prompt_template()

    # Initialize OpenAI client
    client = OpenAI()

    analyses: list[dict] = []
    errors: list[str] = []

    print(f"Analyzing {len(chats)} dialogues using model '{args.model}'...\n")

    for i, chat in enumerate(chats, 1):
        chat_id = chat.get("chat_id", f"unknown_{i}")
        print(f"  [{i}/{len(chats)}] {chat_id} ... ", end="", flush=True)

        try:
            analysis = analyze_chat(
                client=client,
                prompt_template=prompt_template,
                chat=chat,
                model=args.model,
                seed=args.seed,
            )

            # Validate
            issues = validate_analysis(analysis)
            if issues:
                print(f"WARNINGS: {issues}")
                errors.extend(f"{chat_id}: {issue}" for issue in issues)
            else:
                print("OK")

            analyses.append(analysis)

        except Exception as e:
            error_msg = f"{chat_id}: {e}"
            print(f"FAILED — {e}")
            errors.append(error_msg)

    # Write output
    output_data = {"analyses": analyses}
    args.output.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone! Analyzed {len(analyses)} dialogues → {args.output}")
    if errors:
        print(f"\n{len(errors)} issue(s) encountered:")
        for err in errors:
            print(f"  - {err}")

    # Print summary
    print_summary(analyses)


if __name__ == "__main__":
    main()
