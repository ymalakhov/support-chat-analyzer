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

| Flag | Default                         | Description |
|------|---------------------------------|-------------|
| `--output` | `output/chats_<timestamp>.json` | Output file path |
| `--model` | `gpt-4o`                        | LLM model |
| `--seed` | `42`                            | Random seed for reproducibility |
| `--variants` | `2`                             | Variants per category/scenario cell |
| `--concurrency` | `3`                             | Max parallel API calls |
| `--max-retries` | `5`                             | Retries on transient errors |

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
| `--label` | — | Label for this run (stored in output JSON) |

### 3. Evaluate

```bash
python evaluate.py --chats output/chats_27.02.2026_20-34.json output/analysis_v1.json [output/analysis_v2.json ...]
```

Compares one or more analysis files against ground truth labels from the chats file. Outputs per-chat accuracy tables, confusion matrices, and a cross-version comparison.

| Flag | Default | Description |
|------|---------|-------------|
| `--chats` | `output/chats_<timestamp>.json` | Ground truth chats file |
| `analyses` (positional) | — | **Required.** One or more analysis JSON files |

<details>
<summary>Example output</summary>

```
(.venv) support-chat-analyzer % python evaluate.py --chats output/chats_28.02.2026_12-27.json output/analysis_v6.json 
Loaded 50 ground truth chats.
  gpt-4o, tuned prompt v5, new dataset: 50 analyses

=== gpt-4o, tuned prompt v5, new dataset ===
CHAT ID                                    INTENT                 HIDDEN DISSAT.               AGENT MISTAKE
--------------------------------------------------------------------------------------------------------------
payment_issues_successful_001              OK                        OK                              OK
payment_issues_successful_002              OK                        OK                              OK
payment_issues_problematic_003             OK                        OK                              OK
payment_issues_problematic_004             OK                        OK                              OK
payment_issues_conflict_005                OK                        OK                              OK
payment_issues_conflict_006                OK                        OK                              OK
payment_issues_agent_error_007             OK                        FAIL (pred=True)                OK
payment_issues_agent_error_008             OK                        OK                              OK
technical_errors_successful_009            OK                        OK                              OK
technical_errors_successful_010            OK                        OK                              OK
technical_errors_problematic_011           OK                        FAIL (pred=True)                OK
technical_errors_problematic_012           OK                        FAIL (pred=True)                OK
technical_errors_conflict_013              OK                        OK                              OK
technical_errors_conflict_014              OK                        OK                              OK
technical_errors_agent_error_015           OK                        OK                              OK
technical_errors_agent_error_016           OK                        OK                              FAIL (expected=no_resolution, got=[])
account_access_successful_017              OK                        OK                              OK
account_access_successful_018              OK                        OK                              OK
account_access_problematic_019             OK                        OK                              OK
account_access_problematic_020             OK                        OK                              OK
account_access_conflict_021                OK                        OK                              OK
account_access_conflict_022                OK                        OK                              OK
account_access_agent_error_023             OK                        OK                              FAIL (expected=unnecessary_escalation, got=[])
account_access_agent_error_024             OK                        OK                              OK
rate_questions_successful_025              OK                        OK                              OK
rate_questions_successful_026              OK                        OK                              OK
rate_questions_problematic_027             OK                        OK                              OK
rate_questions_problematic_028             OK                        OK                              OK
rate_questions_conflict_029                FAIL (refunds)            OK                              OK
rate_questions_conflict_030                OK                        OK                              OK
rate_questions_agent_error_031             OK                        OK                              OK
rate_questions_agent_error_032             OK                        OK                              OK
refunds_successful_033                     OK                        OK                              OK
refunds_successful_034                     OK                        OK                              OK
refunds_problematic_035                    OK                        OK                              OK
refunds_problematic_036                    OK                        OK                              OK
refunds_conflict_037                       OK                        OK                              OK
refunds_conflict_038                       OK                        OK                              OK
refunds_agent_error_039                    OK                        OK                              OK
refunds_agent_error_040                    OK                        OK                              FAIL (expected=unnecessary_escalation, got=[])
payment_issues_hidden_041                  OK                        FAIL (pred=False)               OK
payment_issues_hidden_042                  OK                        OK                              OK
technical_errors_hidden_043                OK                        FAIL (pred=False)               OK
technical_errors_hidden_044                OK                        FAIL (pred=False)               FAIL (expected=incorrect_info, got=[])
account_access_hidden_045                  OK                        OK                              OK
account_access_hidden_046                  OK                        OK                              OK
rate_questions_hidden_047                  OK                        FAIL (pred=False)               OK
rate_questions_hidden_048                  OK                        OK                              FAIL (expected=no_resolution, got=['ignored_question'])
refunds_hidden_049                         OK                        OK                              OK
refunds_hidden_050                         OK                        OK                              OK
--------------------------------------------------------------------------------------------------------------

--- gpt-4o, tuned prompt v5, new dataset (50 pairs) ---

  METRIC                              SCORE DETAIL
  ----------------------------------------------------------------------
  Intent classification                98%  49/50
  Hidden dissatisfaction               86%  43/50
  Agent mistake detection              90%  45/50

  Hidden dissatisfaction breakdown:
    TP (correctly detected)      6
    TN (correctly rejected)     37
    FP (over-predicted)          3
    FN (missed)                  4

  Agent mistake errors:
    False alarms                 0
    Missed detections            4
    Wrong type                   1

  Predicted mistake distribution:
    ignored_question               5  █████
    no_resolution                  5  █████
    rude_tone                      3  ███
    incorrect_info                 2  ██

  Satisfaction distribution:
    satisfied                     22  ██████████████████████
    neutral                       20  ████████████████████
    unsatisfied                    8  ████████

  Avg quality score:           3.86

================================================================================
COMPARISON ACROSS VERSIONS
================================================================================
  METRIC                       gpt-4o, tuned prompt v5, new dataset
  -----------------------------------------
  Intent                               98%
  Hidden dissatisfaction               86%
  Agent mistakes                       90%

  Hidden FP (over-predict)                3
  Hidden FN (missed)                      4
  Mistake false alarms                    0
  Mistake missed                          4

```

</details>

### Docker

```bash
docker build -t chat-analyzer .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v $(pwd)/output:/app/output chat-analyzer                     # generate
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v $(pwd)/output:/app/output chat-analyzer analyze.py output/chats_27.02.2026_20-34.json
docker run -v $(pwd)/output:/app/output chat-analyzer evaluate.py --chats output/chats_27.02.2026_20-34.json output/analysis_v1.json
```

## How it works

**No metadata leakage.** The analyzer LLM receives only the raw dialogue text with anonymized IDs. It never sees the category, scenario type, ground truth labels, or any hints about what the "correct" answer should be. This means the analysis is a genuine blind assessment — the LLM must figure out the intent, satisfaction level, and mistakes purely from the conversation.

**Deterministic output.** Both scripts use `temperature=0` + fixed `seed` for reproducible results (OpenAI best-effort).

**Diversity.** Each dialogue gets a unique customer persona and specific problem variant, so chats don't feel repetitive even with deterministic settings.

**Chain-of-thought analysis.** The analysis prompt enforces step-by-step reasoning before producing the final JSON, improving accuracy on edge cases like hidden dissatisfaction.

## Approach comparison

We evaluated several approaches to improve accuracy, particularly for hidden dissatisfaction detection (the hardest metric). All results are on a 50-chat dataset with `gpt-4o`.

### Single prompt (chosen)

One LLM call per chat with a 6-step chain-of-thought prompt covering intent, satisfaction, quality, mistakes, hidden dissatisfaction, and reasoning.

| Version | Intent | Hidden dissat. | Mistakes | Hidden FP | Hidden FN |
|---------|--------|----------------|----------|-----------|-----------|
| v5 (baseline) | 98% | 80% | 88% | 0 | 10 |
| v6 (tuned Step 5) | 98% | 86% | 90% | 3 | 4 |

v6 added "pending customer action" rule and shifted focus to the customer's final tone instead of scanning for any frustration anywhere. This balanced FP/FN and improved from 80% → 86%.

### Split prompt (rejected)

Two LLM calls: main prompt (Steps 1–4, 6) + dedicated second call focused only on hidden dissatisfaction with two-phase reasoning (resolution status → tone gap assessment).

| Version | Intent | Hidden dissat. | Mistakes | Hidden FP | Hidden FN |
|---------|--------|----------------|----------|-----------|-----------|
| split v2 (lenient) | 98% | 68% | 90% | 15 | 1 |
| split v3 (strict) | 98% | 80% | 90% | 7 | 3 |
| split v4 (balanced) | 98% | 80% | 90% | 9 | 1 |

The dedicated prompt over-detects hidden dissatisfaction because it's its sole focus - it flags escalations, polite conflict endings, and agent-error scenarios as "hidden" when the frustration is actually visible. Tightening the rules reduces FP but re-introduces FN. After 4 iterations, no split variant beat the single-prompt v6 baseline.

### Why single prompt wins

The single prompt benefits from **cross-task context**: when the model reasons about intent, satisfaction, and mistakes in the same call, those signals naturally calibrate the hidden dissatisfaction decision. The dedicated prompt lacks this context and consistently over-predicts on the full dataset despite performing well on a curated 7-chat subset (100% on AB test).
