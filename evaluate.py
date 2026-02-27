#!/usr/bin/env python3
"""
evaluate.py — Evaluation script for support chat analysis.

Compares analyzer predictions (output/analysis.json) against ground truth
labels (output/chats.json) and produces accuracy metrics for intent
classification, hidden dissatisfaction detection, quality scoring, and
agent mistake detection.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    ANALYSIS_OUTPUT_PATH,
    CATEGORIES,
    CHATS_OUTPUT_PATH,
    EVALUATION_OUTPUT_PATH,
    SCENARIO_TYPES,
)


def load_data(chats_path: Path, analysis_path: Path) -> list[dict]:
    """Load chats and analyses, join on chat_id. Returns list of paired records."""
    if not chats_path.exists():
        print(f"ERROR: Chats file not found: {chats_path}")
        sys.exit(1)
    if not analysis_path.exists():
        print(f"ERROR: Analysis file not found: {analysis_path}")
        print("Run analyze.py first to create the analysis results.")
        sys.exit(1)

    with open(chats_path, encoding="utf-8") as f:
        chats_data = json.load(f)
    with open(analysis_path, encoding="utf-8") as f:
        analysis_data = json.load(f)

    chats = {c["chat_id"]: c for c in chats_data.get("chats", [])}
    analyses = {a["chat_id"]: a for a in analysis_data.get("analyses", [])}

    pairs = []
    for chat_id, chat in chats.items():
        analysis = analyses.get(chat_id)
        if analysis:
            pairs.append({"chat": chat, "analysis": analysis})
        else:
            print(f"  WARNING: No analysis found for {chat_id}")

    return pairs


def compute_intent_accuracy(pairs: list[dict]) -> dict:
    """Compare ground truth category against predicted intent."""
    total = len(pairs)
    correct = 0
    per_category: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    confusion: dict[str, Counter] = defaultdict(Counter)

    for p in pairs:
        gt = p["chat"]["category"]
        pred = p["analysis"]["intent"]
        confusion[gt][pred] += 1

        if gt == pred:
            correct += 1
            per_category[gt]["tp"] += 1
        else:
            per_category[gt]["fn"] += 1
            per_category[pred]["fp"] += 1

    overall_accuracy = correct / total if total else 0

    category_metrics = {}
    for cat in CATEGORIES:
        stats = per_category[cat]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        category_metrics[cat] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": tp + fn,
        }

    return {
        "overall_accuracy": round(overall_accuracy, 3),
        "correct": correct,
        "total": total,
        "per_category": category_metrics,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }


def compute_hidden_dissatisfaction_metrics(pairs: list[dict]) -> dict:
    """Evaluate hidden dissatisfaction detection accuracy."""
    tp = fp = tn = fn = 0

    details = []
    for p in pairs:
        gt_hidden = p["chat"].get("has_hidden_dissatisfaction", False)
        pred_hidden = p["analysis"].get("has_hidden_dissatisfaction", False)

        if gt_hidden and pred_hidden:
            tp += 1
        elif gt_hidden and not pred_hidden:
            fn += 1
            details.append({
                "chat_id": p["chat"]["chat_id"],
                "type": "false_negative",
                "predicted_satisfaction": p["analysis"].get("customer_satisfaction"),
            })
        elif not gt_hidden and pred_hidden:
            fp += 1
            details.append({
                "chat_id": p["chat"]["chat_id"],
                "type": "false_positive",
                "scenario_type": p["chat"].get("scenario_type"),
            })
        else:
            tn += 1

    total = tp + fp + tn + fn
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        "detection_rate": round(detection_rate, 3),
        "false_positive_rate": round(false_positive_rate, 3),
        "precision": round(precision, 3),
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "total_hidden": tp + fn,
        "total": total,
        "misclassified": details,
    }


def compute_quality_correlation(pairs: list[dict]) -> dict:
    """Analyze quality scores vs scenario types for expected correlations."""
    by_scenario: dict[str, list[int]] = defaultdict(list)
    by_satisfaction: dict[str, list[int]] = defaultdict(list)

    for p in pairs:
        scenario = p["chat"].get("scenario_type", "unknown")
        score = p["analysis"].get("quality_score")
        satisfaction = p["analysis"].get("customer_satisfaction", "unknown")
        if isinstance(score, int):
            by_scenario[scenario].append(score)
            by_satisfaction[satisfaction].append(score)

    scenario_stats = {}
    for st in SCENARIO_TYPES:
        scores = by_scenario.get(st, [])
        if scores:
            scenario_stats[st] = {
                "avg_score": round(sum(scores) / len(scores), 2),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
            }

    satisfaction_stats = {}
    for level in ["satisfied", "neutral", "unsatisfied"]:
        scores = by_satisfaction.get(level, [])
        if scores:
            satisfaction_stats[level] = {
                "avg_score": round(sum(scores) / len(scores), 2),
                "count": len(scores),
            }

    # Check expected correlations
    checks = []
    successful_avg = scenario_stats.get("successful", {}).get("avg_score", 0)
    agent_error_avg = scenario_stats.get("agent_error", {}).get("avg_score", 5)
    if successful_avg > 0 and agent_error_avg > 0:
        checks.append({
            "check": "successful_score > agent_error_score",
            "passed": successful_avg > agent_error_avg,
            "successful_avg": successful_avg,
            "agent_error_avg": agent_error_avg,
        })

    return {
        "by_scenario_type": scenario_stats,
        "by_satisfaction": satisfaction_stats,
        "correlation_checks": checks,
    }


def compute_mistake_detection(pairs: list[dict]) -> dict:
    """Evaluate mistake detection for agent_error vs successful scenarios."""
    agent_error_pairs = [p for p in pairs if p["chat"].get("scenario_type") == "agent_error"]
    successful_pairs = [p for p in pairs if p["chat"].get("scenario_type") == "successful"]

    # agent_error scenarios should have at least one detected mistake
    ae_with_mistakes = sum(
        1 for p in agent_error_pairs if p["analysis"].get("agent_mistakes")
    )
    ae_total = len(agent_error_pairs)
    ae_detection_rate = ae_with_mistakes / ae_total if ae_total else 0

    # successful scenarios should generally have empty mistake lists
    s_with_mistakes = sum(
        1 for p in successful_pairs if p["analysis"].get("agent_mistakes")
    )
    s_total = len(successful_pairs)
    s_false_alarm_rate = s_with_mistakes / s_total if s_total else 0

    # Mistake type distribution across all
    all_mistakes: list[str] = []
    for p in pairs:
        all_mistakes.extend(p["analysis"].get("agent_mistakes", []))
    mistake_distribution = dict(Counter(all_mistakes).most_common())

    return {
        "agent_error_detection_rate": round(ae_detection_rate, 3),
        "agent_error_with_mistakes": ae_with_mistakes,
        "agent_error_total": ae_total,
        "successful_false_alarm_rate": round(s_false_alarm_rate, 3),
        "successful_with_mistakes": s_with_mistakes,
        "successful_total": s_total,
        "mistake_distribution": mistake_distribution,
    }


def print_report(metrics: dict) -> None:
    """Print a human-readable evaluation report."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    # Intent accuracy
    intent = metrics["intent_accuracy"]
    print(f"\n--- Intent Classification ---")
    print(f"  Overall accuracy: {intent['overall_accuracy']:.1%} ({intent['correct']}/{intent['total']})")
    print(f"\n  Per-category metrics:")
    print(f"  {'Category':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*48}")
    for cat in CATEGORIES:
        m = intent["per_category"].get(cat, {})
        print(
            f"  {cat:<22} {m.get('precision', 0):>6.3f} {m.get('recall', 0):>6.3f} "
            f"{m.get('f1', 0):>6.3f} {m.get('support', 0):>8}"
        )

    # Confusion matrix
    print(f"\n  Confusion matrix (rows=ground truth, cols=predicted):")
    all_labels = CATEGORIES + ["other"]
    header = "  " + f"{'':>22}" + "".join(f"{l[:8]:>10}" for l in all_labels)
    print(header)
    for gt in CATEGORIES:
        row = intent["confusion_matrix"].get(gt, {})
        vals = "".join(f"{row.get(p, 0):>10}" for p in all_labels)
        print(f"  {gt:>22}{vals}")

    # Hidden dissatisfaction
    hd = metrics["hidden_dissatisfaction"]
    print(f"\n--- Hidden Dissatisfaction Detection ---")
    print(f"  Detection rate (recall):  {hd['detection_rate']:.1%} ({hd['true_positives']}/{hd['total_hidden']})")
    print(f"  Precision:                {hd['precision']:.1%}")
    print(f"  False positive rate:      {hd['false_positive_rate']:.1%} ({hd['false_positives']})")
    if hd["misclassified"]:
        print(f"  Misclassified cases:")
        for case in hd["misclassified"][:10]:
            print(f"    - {case['chat_id']}: {case['type']}")

    # Quality correlation
    qc = metrics["quality_correlation"]
    print(f"\n--- Quality Score Correlation ---")
    print(f"  {'Scenario Type':<22} {'Avg Score':>10} {'Min':>5} {'Max':>5} {'Count':>6}")
    print(f"  {'-'*48}")
    for st in SCENARIO_TYPES:
        s = qc["by_scenario_type"].get(st, {})
        if s:
            print(
                f"  {st:<22} {s['avg_score']:>10.2f} {s['min']:>5} "
                f"{s['max']:>5} {s['count']:>6}"
            )

    for check in qc.get("correlation_checks", []):
        status = "PASS" if check["passed"] else "FAIL"
        print(f"\n  Check: {check['check']} -> {status}")

    # Mistake detection
    md = metrics["mistake_detection"]
    print(f"\n--- Agent Mistake Detection ---")
    print(
        f"  agent_error detection rate:    {md['agent_error_detection_rate']:.1%} "
        f"({md['agent_error_with_mistakes']}/{md['agent_error_total']})"
    )
    print(
        f"  successful false alarm rate:   {md['successful_false_alarm_rate']:.1%} "
        f"({md['successful_with_mistakes']}/{md['successful_total']})"
    )
    if md["mistake_distribution"]:
        print(f"\n  Mistake type distribution:")
        for mistake, count in md["mistake_distribution"].items():
            print(f"    {mistake:<28} {count:>4}")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate analysis results against ground truth labels."
    )
    parser.add_argument(
        "--chats",
        type=Path,
        default=CHATS_OUTPUT_PATH,
        help=f"Ground truth chats file (default: {CHATS_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=ANALYSIS_OUTPUT_PATH,
        help=f"Analysis results file (default: {ANALYSIS_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVALUATION_OUTPUT_PATH,
        help=f"Evaluation output file (default: {EVALUATION_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    # Load and join data
    pairs = load_data(args.chats, args.analysis)
    print(f"Loaded {len(pairs)} chat-analysis pairs for evaluation.")

    # Compute all metrics
    metrics = {
        "intent_accuracy": compute_intent_accuracy(pairs),
        "hidden_dissatisfaction": compute_hidden_dissatisfaction_metrics(pairs),
        "quality_correlation": compute_quality_correlation(pairs),
        "mistake_detection": compute_mistake_detection(pairs),
        "total_pairs_evaluated": len(pairs),
    }

    # Write detailed output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Detailed metrics written to {args.output}")

    # Print report
    print_report(metrics)


if __name__ == "__main__":
    main()
