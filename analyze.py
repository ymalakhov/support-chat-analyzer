from __future__ import annotations

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

load_dotenv()

DEFAULT_CONCURRENCY = 5


def load_prompt_template() -> str:
    if not ANALYZE_PROMPT_PATH.exists():
        print(f"ERROR: Prompt template not found at {ANALYZE_PROMPT_PATH}")
        sys.exit(1)
    return ANALYZE_PROMPT_PATH.read_text(encoding="utf-8")


def format_dialogue_for_prompt(chat: dict, anonymized_id: str) -> str:
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
    analysis["chat_id"] = chat["chat_id"]

    return analysis


def validate_analysis(analysis: dict) -> list[str]:
    issues: list[str] = []
    try:
        validate(instance=analysis, schema=ANALYSIS_SCHEMA)
    except ValidationError as e:
        path = " -> ".join(str(p) for p in e.absolute_path) or "(root)"
        issues.append(f"Schema validation at {path}: {e.message}")
    return issues


def print_distribution(title: str, counts: Counter, ordered_keys: list[str] | None = None) -> None:
    print(f"\n{title}:")
    items = [(k, counts.get(k, 0)) for k in ordered_keys] if ordered_keys else sorted(counts.items())
    for key, count in items:
        bar = "█" * count
        print(f"  {key:<22} {count:>3}  {bar}")


def print_quality_scores(analyses: list[dict]) -> None:
    scores = [
        a["quality_score"]
        for a in analyses
        if isinstance(a.get("quality_score"), int)
    ]
    if not scores:
        return
    avg = sum(scores) / len(scores)
    score_counts = Counter(scores)
    print(f"\nQuality Score (avg: {avg:.2f}):")
    for score in range(1, 6):
        count = score_counts.get(score, 0)
        bar = "█" * count
        print(f"  Score {score}  {count:>3}  {bar}")


def print_mistake_summary(analyses: list[dict]) -> None:
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

    perfect = sum(1 for a in analyses if not a.get("agent_mistakes"))
    print(f"\nChats with no agent mistakes: {perfect}/{len(analyses)}")


def print_summary(analyses: list[dict]) -> None:
    if not analyses:
        print("No analyses to summarize.")
        return

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    intent_counts = Counter(a.get("intent", "?") for a in analyses)
    print_distribution("Intent Distribution", intent_counts)

    sat_counts = Counter(a.get("customer_satisfaction", "?") for a in analyses)
    print_distribution("Customer Satisfaction", sat_counts, ordered_keys=SATISFACTION_LEVELS)

    print_quality_scores(analyses)
    print_mistake_summary(analyses)

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
) -> tuple:
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


def load_chats(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: Input file not found: {path}")
        print("Run generate.py first to create the chat dataset.")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    chats = data.get("chats", [])
    if not chats:
        print("ERROR: No chats found in input file.")
        sys.exit(1)

    print(f"Loaded {len(chats)} chats from {path}")
    return chats


def write_results(path: Path, analyses: list[dict], label: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {"analyses": analyses}
    if label:
        output_data["label"] = label
    path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def collect_results(results: list[tuple]) -> tuple[list[dict], list[str]]:
    analyses: list[dict] = []
    errors: list[str] = []
    for analysis, errs in results:
        if analysis is not None:
            analyses.append(analysis)
        errors.extend(errs)
    return analyses, errors


def report_errors(errors: list[str]) -> None:
    if not errors:
        return
    print(f"\n{len(errors)} issue(s) encountered:")
    for err in errors:
        print(f"  - {err}")


async def async_main(args: argparse.Namespace) -> None:
    chats = load_chats(args.input)
    prompt_template = load_prompt_template()

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)

    print(f"Analyzing {len(chats)} dialogues using model '{args.model}'...\n")

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
    analyses, errors = collect_results(results)

    write_results(args.output, analyses, label=args.label)

    print(f"\nDone! Analyzed {len(analyses)} dialogues -> {args.output}")
    report_errors(errors)
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
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label describing this run (e.g. 'gpt-4o, tuned prompt v5'). Stored in output JSON.",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
