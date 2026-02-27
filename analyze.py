#!/usr/bin/env python3
"""
analyze.py — Chat quality analyzer.

Reads generated dialogues from output/chats.json, sends each to an LLM for
quality assessment, and writes structured analysis results to
output/analysis.json.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from jsonschema import ValidationError, validate
from openai import AsyncOpenAI

from config import (
    AGENT_MISTAKES,
    ANALYSIS_OUTPUT_PATH,
    ANALYSIS_SCHEMA,
    ANALYZE_PROMPT_PATH,
    CATEGORIES,
    MODEL,
    SATISFACTION_LEVELS,
    SEED,
    TEMPERATURE,
    TOP_P,
)
from utils import retry_with_backoff

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv()

DEFAULT_CONCURRENCY = 5


def load_prompt_template() -> str:
    """Read the analysis prompt template from disk."""
    if not ANALYZE_PROMPT_PATH.exists():
        print(f"ERROR: Prompt template not found at {ANALYZE_PROMPT_PATH}")
        sys.exit(1)
    return ANALYZE_PROMPT_PATH.read_text(encoding="utf-8")


def format_dialogue_for_prompt(chat: dict, anonymized_id: str) -> str:
    """Convert a chat dict into a readable dialogue string for the prompt.

    Uses an anonymized ID to prevent the LLM from inferring intent or
    scenario from metadata embedded in the original chat_id.
    """
    lines = [
        f"Chat ID: {anonymized_id}",
        "",
        "--- Dialogue ---",
    ]
    for msg in chat.get("messages", []):
        role = msg.get("role", "unknown").upper()
        text = msg.get("text", "")
        lines.append(f"[{role}]: {text}")
    lines.append("--- End of Dialogue ---")
    return "\n".join(lines)


async def analyze_chat(
    client: AsyncOpenAI,
    prompt_template: str,
    chat: dict,
    model: str,
    seed: int,
    anonymized_id: str = "DIALOGUE",
    max_retries: int = 3,
) -> dict:
    """Call the LLM to analyze a single chat dialogue."""
    dialogue_text = format_dialogue_for_prompt(chat, anonymized_id)
    user_prompt = prompt_template.format(dialogue=dialogue_text)

    async def _call():
        return await client.chat.completions.create(
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

    response = await retry_with_backoff(_call, max_retries=max_retries)

    raw = response.choices[0].message.content
    analysis = json.loads(raw)

    # Restore real chat_id
    analysis["chat_id"] = chat["chat_id"]

    return analysis


def validate_analysis(analysis: dict) -> list[str]:
    """Validate an analysis result against the JSON schema. Returns list of issues."""
    issues: list[str] = []
    try:
        validate(instance=analysis, schema=ANALYSIS_SCHEMA)
    except ValidationError as e:
        path = " -> ".join(str(p) for p in e.absolute_path) or "(root)"
        issues.append(f"Schema validation at {path}: {e.message}")
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

    # Hidden dissatisfaction detection
    hd_detected = sum(1 for a in analyses if a.get("has_hidden_dissatisfaction"))
    print(f"Hidden dissatisfaction detected: {hd_detected}/{len(analyses)}")

    print("=" * 60)


async def analyze_one(
    client: AsyncOpenAI,
    prompt_template: str,
    chat: dict,
    index: int,
    total: int,
    model: str,
    seed: int,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> tuple[dict | None, list[str]]:
    """Analyze a single chat, respecting the concurrency semaphore."""
    async with semaphore:
        chat_id = chat.get("chat_id", f"unknown_{index}")
        anonymized_id = f"DIALOGUE_{index:03d}"

        try:
            analysis = await analyze_chat(
                client=client,
                prompt_template=prompt_template,
                chat=chat,
                model=model,
                seed=seed,
                anonymized_id=anonymized_id,
                max_retries=max_retries,
            )

            issues = validate_analysis(analysis)
            if issues:
                print(f"  [{index}/{total}] {chat_id} ... WARNINGS: {issues}")
                return analysis, [f"{chat_id}: {issue}" for issue in issues]
            else:
                print(f"  [{index}/{total}] {chat_id} ... OK")
                return analysis, []

        except Exception as e:
            print(f"  [{index}/{total}] {chat_id} ... FAILED — {e}")
            return None, [f"{chat_id}: {e}"]


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point for analysis."""
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

    # Initialize async OpenAI client
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)

    print(f"Analyzing {len(chats)} dialogues using model '{args.model}'...\n")

    # Launch all tasks concurrently (bounded by semaphore)
    tasks = [
        analyze_one(
            client=client,
            prompt_template=prompt_template,
            chat=chat,
            index=i,
            total=len(chats),
            model=args.model,
            seed=args.seed,
            semaphore=semaphore,
            max_retries=args.max_retries,
        )
        for i, chat in enumerate(chats, 1)
    ]

    results = await asyncio.gather(*tasks)

    analyses: list[dict] = []
    errors: list[str] = []
    for analysis, errs in results:
        if analysis is not None:
            analyses.append(analysis)
        errors.extend(errs)

    # Write output
    output_data = {"analyses": analyses}
    args.output.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone! Analyzed {len(analyses)} dialogues -> {args.output}")
    if errors:
        print(f"\n{len(errors)} issue(s) encountered:")
        for err in errors:
            print(f"  - {err}")

    # Print summary
    print_summary(analyses)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze support chat dialogues for quality assessment."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input JSON file with chats (e.g. output/chats_27.02.2026_20-34.json)",
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent API calls (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries on transient API errors (default: 3)",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
