#!/usr/bin/env python3
"""
generate.py — Synthetic support chat dataset generator.

Builds a scenario matrix (categories × scenario types), calls an LLM to
generate realistic customer-agent dialogues, and writes the result to
output/chats.json.
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import (
    CATEGORIES,
    CHATS_OUTPUT_PATH,
    GENERATE_PROMPT_PATH,
    MODEL,
    OUTPUT_DIR,
    SCENARIO_TYPES,
    SEED,
    TEMPERATURE,
    TOP_P,
)

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv()


def load_prompt_template() -> str:
    """Read the generation prompt template from disk."""
    if not GENERATE_PROMPT_PATH.exists():
        print(f"ERROR: Prompt template not found at {GENERATE_PROMPT_PATH}")
        sys.exit(1)
    return GENERATE_PROMPT_PATH.read_text(encoding="utf-8")


def build_scenario_matrix() -> list[dict]:
    """
    Create the full list of scenarios to generate.

    Base matrix: 5 categories × 4 scenario types = 20 dialogues.
    Extra: one hidden-dissatisfaction variant per category = 5 more.
    Total: 25 dialogues minimum.
    """
    scenarios: list[dict] = []
    seq = 1

    # Base matrix — all combinations, no hidden dissatisfaction
    for category in CATEGORIES:
        for scenario_type in SCENARIO_TYPES:
            scenarios.append(
                {
                    "chat_id": f"{category}_{scenario_type}_{seq:03d}",
                    "category": category,
                    "scenario_type": scenario_type,
                    "hidden_dissatisfaction": False,
                }
            )
            seq += 1

    # Hidden dissatisfaction variants — one per category, using "problematic"
    # or "agent_error" scenarios where it makes sense
    hidden_scenario_types = [
        "problematic",
        "agent_error",
        "problematic",
        "agent_error",
        "problematic",
    ]
    for category, scenario_type in zip(CATEGORIES, hidden_scenario_types):
        scenarios.append(
            {
                "chat_id": f"{category}_hidden_{seq:03d}",
                "category": category,
                "scenario_type": scenario_type,
                "hidden_dissatisfaction": True,
            }
        )
        seq += 1

    return scenarios


def generate_chat(
    client: OpenAI,
    prompt_template: str,
    scenario: dict,
    model: str,
    seed: int,
) -> dict:
    """Call the LLM to generate a single chat dialogue."""
    user_prompt = prompt_template.format(
        category=scenario["category"],
        scenario_type=scenario["scenario_type"],
        hidden_dissatisfaction=str(scenario["hidden_dissatisfaction"]).lower(),
    )

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
                    "You are a synthetic data generator. "
                    "Always respond with valid JSON only."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content
    chat_data = json.loads(raw)

    # Ensure the chat_id matches our assigned ID
    chat_data["chat_id"] = scenario["chat_id"]

    return chat_data


def validate_chat(chat: dict) -> list[str]:
    """Basic validation of a generated chat. Returns list of issues."""
    issues: list[str] = []
    required_keys = [
        "chat_id",
        "category",
        "scenario_type",
        "has_hidden_dissatisfaction",
        "messages",
    ]
    for key in required_keys:
        if key not in chat:
            issues.append(f"Missing key: {key}")

    if "messages" in chat:
        if not isinstance(chat["messages"], list) or len(chat["messages"]) < 2:
            issues.append("Messages must be a list with at least 2 entries")
        else:
            for i, msg in enumerate(chat["messages"]):
                if "role" not in msg or "text" not in msg:
                    issues.append(f"Message {i} missing 'role' or 'text'")
                elif msg["role"] not in ("customer", "agent"):
                    issues.append(f"Message {i} has invalid role: {msg['role']}")

    if "category" in chat and chat["category"] not in CATEGORIES:
        issues.append(f"Invalid category: {chat['category']}")

    if "scenario_type" in chat and chat["scenario_type"] not in SCENARIO_TYPES:
        issues.append(f"Invalid scenario_type: {chat['scenario_type']}")

    return issues


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
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load prompt template
    prompt_template = load_prompt_template()

    # Build scenario matrix
    scenarios = build_scenario_matrix()
    print(f"Generating {len(scenarios)} chat dialogues using model '{args.model}'...")

    # Initialize OpenAI client
    client = OpenAI()

    chats: list[dict] = []
    errors: list[str] = []

    for i, scenario in enumerate(scenarios, 1):
        label = (
            f"[{i}/{len(scenarios)}] "
            f"{scenario['category']} / {scenario['scenario_type']}"
        )
        if scenario["hidden_dissatisfaction"]:
            label += " (hidden dissatisfaction)"
        print(f"  {label} ... ", end="", flush=True)

        try:
            chat = generate_chat(
                client=client,
                prompt_template=prompt_template,
                scenario=scenario,
                model=args.model,
                seed=args.seed,
            )

            # Validate
            issues = validate_chat(chat)
            if issues:
                print(f"WARNINGS: {issues}")
                errors.extend(
                    f"{scenario['chat_id']}: {issue}" for issue in issues
                )
            else:
                print("OK")

            chats.append(chat)

        except Exception as e:
            error_msg = f"{scenario['chat_id']}: {e}"
            print(f"FAILED — {e}")
            errors.append(error_msg)

    # Write output
    output_data = {"chats": chats}
    args.output.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary
    print(f"\nDone! Generated {len(chats)} dialogues → {args.output}")
    if errors:
        print(f"\n{len(errors)} issue(s) encountered:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("No issues detected.")

    # Coverage summary
    print("\nCoverage matrix:")
    for cat in CATEGORIES:
        cat_chats = [c for c in chats if c.get("category") == cat]
        types = [c.get("scenario_type", "?") for c in cat_chats]
        hidden = sum(1 for c in cat_chats if c.get("has_hidden_dissatisfaction"))
        print(f"  {cat}: {types} (hidden_dissatisfaction: {hidden})")


if __name__ == "__main__":
    main()
