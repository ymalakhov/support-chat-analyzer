# Support Chat Analyzer

LLM-powered tool that **generates** synthetic customer support chat dialogues, **analyzes** them for quality, and **evaluates** analyzer accuracy against ground truth labels.

## Features

- **Synthetic dataset generation** — Realistic customer-agent dialogues across 5 categories and 4 scenario types, with diverse personas and problem variants
- **Automated quality analysis** — Intent classification, satisfaction detection, agent quality scoring (1-5), mistake identification, and hidden dissatisfaction detection
- **Evaluation metrics** — Ground truth comparison with intent accuracy, hidden dissatisfaction detection rate, quality score correlation, and mistake detection metrics
- **Hidden dissatisfaction detection** — Identifies cases where customers appear polite but their issue remains unresolved
- **Deterministic output** — Reproducible results via fixed temperature, seed, and model parameters
- **Async concurrent execution** — Parallel API calls with configurable concurrency and retry logic
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

This creates `output/chats.json` with ~50 synthetic dialogues (5 categories x 4 scenario types x 2 variants + 10 hidden dissatisfaction).

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `output/chats.json` | Output file path |
| `--model` | `gpt-4o` | LLM model name |
| `--seed` | `42` | Random seed for determinism |
| `--variants` | `2` | Variants per category/scenario cell |
| `--concurrency` | `5` | Max concurrent API calls |
| `--max-retries` | `3` | Max retries on transient errors |

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
| `--concurrency` | `5` | Max concurrent API calls |
| `--max-retries` | `3` | Max retries on transient errors |

### 3. Evaluate accuracy

```bash
python evaluate.py
```

Compares analyzer predictions against ground truth labels and prints a detailed metrics report.

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--chats` | `output/chats.json` | Ground truth chats file |
| `--analysis` | `output/analysis.json` | Analysis results file |
| `--output` | `output/evaluation.json` | Evaluation output file |

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
        {"role": "customer", "text": "Hi, my payment failed...", "timestamp": "2024-01-15T10:30:00Z"},
        {"role": "agent", "text": "I'd be happy to help...", "timestamp": "2024-01-15T10:31:15Z"}
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
      "has_hidden_dissatisfaction": false,
      "reasoning": "The agent resolved the payment issue quickly..."
    }
  ]
}
```

### evaluation.json

Contains detailed metrics: intent accuracy with per-category precision/recall/F1, confusion matrix, hidden dissatisfaction detection rates, quality score correlation by scenario type, and agent mistake detection statistics.

## Project Structure

```
support-chat-analyzer/
├── generate.py              # Chat dataset generator (async)
├── analyze.py               # Chat quality analyzer (async)
├── evaluate.py              # Ground truth evaluation
├── config.py                # Shared constants, enums, schemas, personas
├── utils.py                 # Retry logic and shared utilities
├── prompts/
│   ├── generate_prompt.txt  # Prompt template for generation
│   └── analyze_prompt.txt   # Prompt template for analysis
├── output/
│   ├── chats.json           # Generated dialogues (created by generate.py)
│   ├── analysis.json        # Analysis results (created by analyze.py)
│   └── evaluation.json      # Evaluation metrics (created by evaluate.py)
├── tests/                   # Unit tests
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
- **No metadata leakage**: The analyzer receives only the raw dialogue text with anonymized IDs. It does not see the category, scenario type, or any ground truth labels.
- **Prompt externalization**: Templates stored in `prompts/` directory, separate from code logic. Easy to iterate on prompts without modifying Python.
- **Diversity through personas**: Each dialogue uses a unique customer persona and specific problem variant, preventing repetitive openings even at temperature=0.
- **Chain-of-thought analysis**: The analysis prompt enforces step-by-step reasoning before producing the final JSON, improving accuracy for edge cases.
- **Async with retry**: Concurrent API calls with exponential backoff on transient errors (rate limits, timeouts, connection errors).
- **Schema validation**: Uses `jsonschema` library for rigorous validation of all generated and analyzed data.

## Docker (Optional)

```bash
# Build
docker build -t support-chat-analyzer .

# Generate chats
docker run --env-file .env support-chat-analyzer generate.py

# Analyze chats (mount output to persist results)
docker run --env-file .env -v $(pwd)/output:/app/output support-chat-analyzer analyze.py

# Run evaluation
docker run --env-file .env -v $(pwd)/output:/app/output support-chat-analyzer evaluate.py
```

## License

MIT
