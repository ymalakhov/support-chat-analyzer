# Support Chat Analyzer

LLM-powered tool that **generates** synthetic customer support chat dialogues and **analyzes** them for quality, intent, customer satisfaction, and agent errors.

## Features

- **Synthetic dataset generation** — Realistic customer-agent dialogues across 5 categories and 4 scenario types, including edge cases with hidden dissatisfaction
- **Automated quality analysis** — Intent classification, satisfaction detection, agent quality scoring (1–5), and mistake identification
- **Hidden dissatisfaction detection** — Identifies cases where customers appear polite but their issue remains unresolved
- **Deterministic output** — Reproducible results via fixed temperature, seed, and model parameters
- **Structured JSON output** — Both scripts produce validated, schema-conformant JSON

## Prerequisites

- Python 3.10+
- OpenAI API key (GPT-4o recommended)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd support-chat-analyzer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

## Usage

### 1. Generate chat dataset

```bash
python generate.py
```

This creates `output/chats.json` with ~25 synthetic dialogues.

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `output/chats.json` | Output file path |
| `--model` | `gpt-4o` | LLM model name |
| `--seed` | `42` | Random seed for determinism |

### 2. Analyze dialogues

```bash
python analyze.py
```

This reads `output/chats.json`, analyzes each dialogue, and writes results to `output/analysis.json`. A summary table is printed to stdout.

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `output/chats.json` | Input chats file |
| `--output` | `output/analysis.json` | Output analysis file |
| `--model` | `gpt-4o` | LLM model name |
| `--seed` | `42` | Random seed for determinism |

## Output Format

### chats.json

```json
{
  "chats": [
    {
      "chat_id": "payment_issues_successful_001",
      "category": "payment_issues",
      "scenario_type": "successful",
      "has_hidden_dissatisfaction": false,
      "messages": [
        {"role": "customer", "text": "Hi, my payment failed..."},
        {"role": "agent", "text": "I'd be happy to help..."}
      ]
    }
  ]
}
```

### analysis.json

```json
{
  "analyses": [
    {
      "chat_id": "payment_issues_successful_001",
      "intent": "payment_issues",
      "customer_satisfaction": "satisfied",
      "quality_score": 5,
      "agent_mistakes": [],
      "reasoning": "The agent resolved the payment issue quickly..."
    }
  ]
}
```

## Project Structure

```
support-chat-analyzer/
├── generate.py              # Chat dataset generator
├── analyze.py               # Chat quality analyzer
├── config.py                # Shared constants, enums, and JSON schemas
├── prompts/
│   ├── generate_prompt.txt  # Prompt template for generation
│   └── analyze_prompt.txt   # Prompt template for analysis
├── output/
│   ├── chats.json           # Generated dialogues (created by generate.py)
│   └── analysis.json        # Analysis results (created by analyze.py)
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
└── README.md
```

## Categories & Scenarios

**Intent categories:**
- `payment_issues` — Failed transactions, double charges, billing errors
- `technical_errors` — Bugs, crashes, error messages, connectivity
- `account_access` — Login issues, password resets, locked accounts
- `rate_questions` — Pricing plans, subscription tiers, upgrades
- `refunds` — Refund requests, charge disputes, cancellations

**Scenario types:**
- `successful` — Issue resolved efficiently
- `problematic` — Partial resolution, workarounds, multiple exchanges
- `conflict` — Tense interactions, frustrated customers
- `agent_error` — Agent makes mistakes (incorrect info, rude tone, etc.)

**Agent mistake types:**
- `ignored_question` — Skipped a customer's direct question
- `incorrect_info` — Provided wrong information
- `rude_tone` — Dismissive or condescending language
- `no_resolution` — Chat ended without solving the issue
- `unnecessary_escalation` — Escalated when avoidable

## Architecture Decisions

- **Model**: GPT-4o — best quality for structured output, nuanced analysis, and hidden dissatisfaction detection. Override with `--model gpt-4o-mini` for cheaper iteration.
- **Determinism**: `temperature=0` + `seed=42` + `top_p=1`. OpenAI notes this is "best effort" — minor backend changes may affect output. Generated files serve as cached snapshots.
- **Prompt externalization**: Templates stored in `prompts/` directory, separate from code logic. Easy to iterate on prompts without modifying Python.
- **Chain-of-thought analysis**: The analysis prompt enforces step-by-step reasoning before producing the final JSON, improving accuracy for edge cases.
- **Minimal dependencies**: Only `openai` and `python-dotenv` — no heavy frameworks.

## Docker (Optional)

```bash
# Build
docker build -t support-chat-analyzer .

# Generate chats
docker run --env-file .env support-chat-analyzer python generate.py

# Analyze chats (mount output to persist results)
docker run --env-file .env -v $(pwd)/output:/app/output support-chat-analyzer python analyze.py
```

## License

MIT
