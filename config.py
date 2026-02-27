"""
config.py — Shared constants, enums, and schemas for the support chat analyzer.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "output"

GENERATE_PROMPT_PATH = PROMPTS_DIR / "generate_prompt.txt"
ANALYZE_PROMPT_PATH = PROMPTS_DIR / "analyze_prompt.txt"

CHATS_OUTPUT_PATH = OUTPUT_DIR / "chats.json"
ANALYSIS_OUTPUT_PATH = OUTPUT_DIR / "analysis.json"

# ---------------------------------------------------------------------------
# LLM Settings (deterministic)
# ---------------------------------------------------------------------------
MODEL = os.getenv("LLM_MODEL", "gpt-4o")
TEMPERATURE = 0
SEED = 42
TOP_P = 1

# ---------------------------------------------------------------------------
# Domain Enums
# ---------------------------------------------------------------------------
CATEGORIES = [
    "payment_issues",
    "technical_errors",
    "account_access",
    "rate_questions",
    "refunds",
]

SCENARIO_TYPES = [
    "successful",
    "problematic",
    "conflict",
    "agent_error",
]

SATISFACTION_LEVELS = [
    "satisfied",
    "neutral",
    "unsatisfied",
]

AGENT_MISTAKES = [
    "ignored_question",
    "incorrect_info",
    "rude_tone",
    "no_resolution",
    "unnecessary_escalation",
]

# ---------------------------------------------------------------------------
# JSON Schemas — used for structured output and validation
# ---------------------------------------------------------------------------

# Schema for a single message in a dialogue
MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {
            "type": "string",
            "enum": ["customer", "agent"],
            "description": "Who sent the message.",
        },
        "text": {
            "type": "string",
            "description": "The message content.",
        },
    },
    "required": ["role", "text"],
    "additionalProperties": False,
}

# Schema for a single generated chat (output of generate.py)
CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "chat_id": {
            "type": "string",
            "description": "Unique identifier for the dialogue.",
        },
        "category": {
            "type": "string",
            "enum": CATEGORIES,
            "description": "The support request category.",
        },
        "scenario_type": {
            "type": "string",
            "enum": SCENARIO_TYPES,
            "description": "The outcome type of the dialogue.",
        },
        "has_hidden_dissatisfaction": {
            "type": "boolean",
            "description": "True if the customer appears polite but the issue is unresolved.",
        },
        "messages": {
            "type": "array",
            "items": MESSAGE_SCHEMA,
            "description": "The dialogue messages in chronological order.",
        },
    },
    "required": [
        "chat_id",
        "category",
        "scenario_type",
        "has_hidden_dissatisfaction",
        "messages",
    ],
    "additionalProperties": False,
}

# Schema for the full generate.py output (array wrapper)
CHATS_WRAPPER_SCHEMA = {
    "type": "object",
    "properties": {
        "chats": {
            "type": "array",
            "items": CHAT_SCHEMA,
        }
    },
    "required": ["chats"],
    "additionalProperties": False,
}

# Schema for a single analysis result (output of analyze.py)
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "chat_id": {
            "type": "string",
            "description": "Matches the chat_id from the input dialogue.",
        },
        "intent": {
            "type": "string",
            "enum": CATEGORIES + ["other"],
            "description": "The detected intent / category of the request.",
        },
        "customer_satisfaction": {
            "type": "string",
            "enum": SATISFACTION_LEVELS,
            "description": "Inferred customer satisfaction level.",
        },
        "quality_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Agent quality score from 1 (worst) to 5 (best).",
        },
        "agent_mistakes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": AGENT_MISTAKES,
            },
            "description": "List of detected agent errors (may be empty).",
        },
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning for the assessment.",
        },
    },
    "required": [
        "chat_id",
        "intent",
        "customer_satisfaction",
        "quality_score",
        "agent_mistakes",
        "reasoning",
    ],
    "additionalProperties": False,
}
