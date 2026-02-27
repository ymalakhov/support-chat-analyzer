# Support Chat Analyzer

Generate synthetic support chats with an LLM, then analyze them for quality — intent, satisfaction, agent mistakes, and hidden dissatisfaction.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set your OPENAI_API_KEY
```

## Usage

### 1. Generate chats

```bash
python generate.py
```

Produces a timestamped file like `output/chats_27.02.2026_20-34.json` with ~50 dialogues (5 categories x 4 scenario types x 2 variants + hidden dissatisfaction cases).

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `output/chats_<timestamp>.json` | Output file path |
| `--model` | `gpt-4o` | LLM model |
| `--seed` | `42` | Random seed for reproducibility |
| `--variants` | `2` | Variants per category/scenario cell |
| `--concurrency` | `5` | Max parallel API calls |
| `--max-retries` | `3` | Retries on transient errors |

### 2. Analyze chats

```bash
python analyze.py output/chats_27.02.2026_20-34.json
```

Takes the path to a chats file as a required argument. Writes results to `output/analysis_<timestamp>.json` and prints a summary table.

| Flag | Default | Description |
|------|---------|-------------|
| `input` (positional) | — | **Required.** Path to chats JSON file |
| `--output` | `output/analysis_<timestamp>.json` | Output file path |
| `--model` | `gpt-4o` | LLM model |
| `--seed` | `42` | Random seed for reproducibility |
| `--concurrency` | `5` | Max parallel API calls |
| `--max-retries` | `3` | Retries on transient errors |

### 3. Evaluate

```bash
python evaluate.py
```

Compares analysis predictions against ground truth and outputs accuracy metrics.

## How it works

**No metadata leakage.** The analyzer LLM receives only the raw dialogue text with anonymized IDs. It never sees the category, scenario type, ground truth labels, or any hints about what the "correct" answer should be. This means the analysis is a genuine blind assessment — the LLM must figure out the intent, satisfaction level, and mistakes purely from the conversation.

**Deterministic output.** Both scripts use `temperature=0` + fixed `seed` for reproducible results (OpenAI best-effort).

**Diversity.** Each dialogue gets a unique customer persona and specific problem variant, so chats don't feel repetitive even with deterministic settings.

**Chain-of-thought analysis.** The analysis prompt enforces step-by-step reasoning before producing the final JSON, improving accuracy on edge cases like hidden dissatisfaction.
