"""
config.py — Shared constants, enums, and schemas for the support chat analyzer.
"""

import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "output"

GENERATE_PROMPT_PATH = PROMPTS_DIR / "generate_prompt.txt"
ANALYZE_PROMPT_PATH = PROMPTS_DIR / "analyze_prompt.txt"

_TIMESTAMP = datetime.now().strftime("%d.%m.%Y_%H-%M")

CHATS_OUTPUT_PATH = OUTPUT_DIR / f"chats_{_TIMESTAMP}.json"
ANALYSIS_OUTPUT_PATH = OUTPUT_DIR / f"analysis_{_TIMESTAMP}.json"
EVALUATION_OUTPUT_PATH = OUTPUT_DIR / f"evaluation_{_TIMESTAMP}.json"

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
# Diversity — Personas and problem variants for realistic generation
# ---------------------------------------------------------------------------
CUSTOMER_PERSONAS = [
    "A young professional, direct and impatient, uses abbreviations and short messages",
    "An elderly person, polite but confused by technology, writes in full sentences",
    "A business owner, formal language, expects premium service and fast resolution",
    "A frustrated parent multitasking, short messages with occasional typos",
    "A tech-savvy user who has already tried troubleshooting on their own",
]

PROBLEM_VARIANTS = {
    "payment_issues": [
        "charged twice for a subscription renewal",
        "payment declined despite having sufficient funds",
        "wrong amount charged on an international order",
        "pending charge that should have been cancelled three days ago",
        "credit card charged after switching payment method to PayPal",
    ],
    "technical_errors": [
        "app crashes on launch after the latest update on iPhone",
        "export to PDF feature produces blank documents",
        "search function returns zero results for any query",
        "push notifications stopped arriving on Android device",
        "dashboard loads extremely slowly and times out after 30 seconds",
    ],
    "account_access": [
        "locked out after three failed password attempts",
        "two-factor authentication code not arriving via SMS",
        "account shows as deactivated even though it should be active",
        "cannot log in after company changed SSO provider",
        "password reset link expired before it could be used",
    ],
    "rate_questions": [
        "comparing Basic vs Premium plan features for a small team",
        "asking about student or nonprofit discount availability",
        "wants to downgrade from annual to monthly without losing data",
        "confused about overage charges on current usage-based plan",
        "asking if enterprise plan includes dedicated account manager",
    ],
    "refunds": [
        "ordered wrong item and wants a full refund within return window",
        "service subscription charged after cancellation was confirmed",
        "received a damaged product and needs replacement or refund",
        "auto-renewal charged unexpectedly after free trial ended",
        "partial refund requested for a service outage lasting two days",
    ],
}

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
        "timestamp": {
            "type": "string",
            "description": "ISO 8601 timestamp of the message.",
        },
    },
    "required": ["role", "text", "timestamp"],
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
        "agent_mistake": {
            "type": "string",
            "enum": AGENT_MISTAKES + [""],
            "description": "The specific agent mistake present in agent_error scenarios. Empty string for other scenarios.",
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
        "agent_mistake",
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
        "has_hidden_dissatisfaction": {
            "type": "boolean",
            "description": "True if hidden dissatisfaction was detected (customer polite but issue unresolved).",
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
        "has_hidden_dissatisfaction",
        "reasoning",
    ],
    "additionalProperties": False,
}
