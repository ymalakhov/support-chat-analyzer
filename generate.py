#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from jsonschema import ValidationError, validate
from openai import AsyncOpenAI

from config import (
    AGENT_MISTAKES,
    CATEGORIES,
    CHAT_SCHEMA,
    CHATS_OUTPUT_PATH,
    CUSTOMER_PERSONAS,
    GENERATE_PROMPT_PATH,
    MODEL,
    PROBLEM_VARIANTS,
    SCENARIO_TYPES,
    SEED,
    TEMPERATURE,
    TOP_P,
)
from utils import retry_with_backoff

load_dotenv()

DEFAULT_VARIANTS_PER_CELL = 2
DEFAULT_CONCURRENCY = 5


def load_prompt_template() -> str:
    if not GENERATE_PROMPT_PATH.exists():
        print(f"ERROR: Prompt template not found at {GENERATE_PROMPT_PATH}")
        sys.exit(1)
    return GENERATE_PROMPT_PATH.read_text(encoding="utf-8")


def build_scenario_matrix(variants_per_cell: int = DEFAULT_VARIANTS_PER_CELL) -> list[dict]:
    scenarios: list[dict] = []
    seq = 1
    mistake_seq = 1

    for category in CATEGORIES:
        problems = PROBLEM_VARIANTS[category]
        for scenario_type in SCENARIO_TYPES:
            for v in range(variants_per_cell):
                persona = CUSTOMER_PERSONAS[(seq - 1) % len(CUSTOMER_PERSONAS)]
                problem = problems[(seq - 1) % len(problems)]
                scenario = {
                    "chat_id": f"{category}_{scenario_type}_{seq:03d}",
                    "category": category,
                    "scenario_type": scenario_type,
                    "hidden_dissatisfaction": False,
                    "persona": persona,
                    "specific_problem": problem,
                    "agent_mistake": "",
                }
                if scenario_type == "agent_error":
                    scenario["agent_mistake"] = AGENT_MISTAKES[(mistake_seq - 1) % len(AGENT_MISTAKES)]
                    mistake_seq += 1
                scenarios.append(scenario)
                seq += 1

    # Hidden dissatisfaction variants: 5 categories * 2 variants = 10 chats
    # Hidden dissatisfaction variants - using "problematic" or "agent_error"
    hidden_scenario_types = [
        "problematic",
        "agent_error",
        "problematic",
        "agent_error",
        "problematic",
    ]
    for category, scenario_type in zip(CATEGORIES, hidden_scenario_types):
        problems = PROBLEM_VARIANTS[category]
        for v in range(variants_per_cell):
            persona = CUSTOMER_PERSONAS[(seq - 1) % len(CUSTOMER_PERSONAS)]
            problem = problems[(seq - 1) % len(problems)]
            scenario = {
                "chat_id": f"{category}_hidden_{seq:03d}",
                "category": category,
                "scenario_type": scenario_type,
                "hidden_dissatisfaction": True,
                "persona": persona,
                "specific_problem": problem,
                "agent_mistake": "",
            }
            if scenario_type == "agent_error":
                scenario["agent_mistake"] = AGENT_MISTAKES[(mistake_seq - 1) % len(AGENT_MISTAKES)]
                mistake_seq += 1
            scenarios.append(scenario)
            seq += 1

    return scenarios


async def generate_chat(
    client: AsyncOpenAI,
    prompt_template: str,
    scenario: dict,
    model: str,
    seed: int,
    max_retries: int = 5,
) -> dict:
    user_prompt = prompt_template.format(
        category=scenario["category"],
        scenario_type=scenario["scenario_type"],
        hidden_dissatisfaction=str(scenario["hidden_dissatisfaction"]).lower(),
        persona=scenario["persona"],
        specific_problem=scenario["specific_problem"],
        agent_mistake=scenario.get("agent_mistake", ""),
    )

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
                        "You are a synthetic data generator. "
                        "Always respond with valid JSON only."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )

    response = await retry_with_backoff(_call, max_retries=max_retries)

    raw = response.choices[0].message.content
    chat_data = json.loads(raw)

    chat_data["chat_id"] = scenario["chat_id"]
    chat_data["agent_mistake"] = scenario.get("agent_mistake", "")
    chat_data["specific_problem"] = scenario.get("specific_problem", "")
    chat_data["persona"] = scenario.get("persona", "")

    return chat_data


def validate_chat(chat: dict) -> list[str]:
    issues: list[str] = []
    try:
        validate(instance=chat, schema=CHAT_SCHEMA)
    except ValidationError as e:
        path = " -> ".join(str(p) for p in e.absolute_path) or "(root)"
        issues.append(f"Schema validation at {path}: {e.message}")

    if "messages" in chat:
        if isinstance(chat["messages"], list) and len(chat["messages"]) < 2:
            issues.append("Messages must have at least 2 entries")

    return issues


async def generate_one(
    client: AsyncOpenAI,
    prompt_template: str,
    scenario: dict,
    index: int,
    total: int,
    model: str,
    seed: int,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> tuple[dict | None, list[str]]:
    async with semaphore:
        label = (
            f"[{index}/{total}] "
            f"{scenario['category']} / {scenario['scenario_type']}"
        )
        if scenario["hidden_dissatisfaction"]:
            label += " (hidden dissatisfaction)"

        try:
            chat = await generate_chat(
                client=client,
                prompt_template=prompt_template,
                scenario=scenario,
                model=model,
                seed=seed,
                max_retries=max_retries,
            )

            issues = validate_chat(chat)
            if issues:
                print(f"  {label} ... WARNINGS: {issues}")
                return chat, [f"{scenario['chat_id']}: {issue}" for issue in issues]
            else:
                print(f"  {label} ... OK")
                return chat, []

        except Exception as e:
            print(f"  {label} ... FAILED — {e}")
            return None, [f"{scenario['chat_id']}: {e}"]


def collect_results(results: list[tuple]) -> tuple[list[dict], list[str]]:
    chats: list[dict] = []
    errors: list[str] = []
    for chat, errs in results:
        if chat is not None:
            chats.append(chat)
        errors.extend(errs)
    return chats, errors


def write_output(path: Path, chats: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {"chats": chats}
    path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def report_errors(errors: list[str]) -> None:
    if not errors:
        print("No issues detected.")
        return
    print(f"\n{len(errors)} issue(s) encountered:")
    for err in errors:
        print(f"  - {err}")


def print_coverage(chats: list[dict]) -> None:
    print("\nCoverage matrix:")
    for cat in CATEGORIES:
        cat_chats = [c for c in chats if c.get("category") == cat]
        types = [c.get("scenario_type", "?") for c in cat_chats]
        hidden = sum(1 for c in cat_chats if c.get("has_hidden_dissatisfaction"))
        print(f"  {cat}: {types} (hidden_dissatisfaction: {hidden})")


async def async_main(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt_template = load_prompt_template()
    scenarios = build_scenario_matrix(variants_per_cell=args.variants)

    print(f"Generating {len(scenarios)} chat dialogues using model '{args.model}'...")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        generate_one(
            client=client,
            prompt_template=prompt_template,
            scenario=scenario,
            index=i,
            total=len(scenarios),
            model=args.model,
            seed=args.seed,
            semaphore=semaphore,
            max_retries=args.max_retries,
        )
        for i, scenario in enumerate(scenarios, 1)
    ]

    results = await asyncio.gather(*tasks)
    chats, errors = collect_results(results)

    write_output(args.output, chats)

    print(f"\nDone! Generated {len(chats)} dialogues -> {args.output}")
    report_errors(errors)
    print_coverage(chats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic support chat dialogues."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CHATS_OUTPUT_PATH,
        help=f"Output JSON file path (default: {CHATS_OUTPUT_PATH})",
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
        "--variants",
        type=int,
        default=DEFAULT_VARIANTS_PER_CELL,
        help=f"Variants per category/scenario cell (default: {DEFAULT_VARIANTS_PER_CELL})",
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
        default=5,
        help="Max retries on transient API errors (default: 5)",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
