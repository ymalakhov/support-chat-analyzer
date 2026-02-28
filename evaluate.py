import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from config import CHATS_OUTPUT_PATH

BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"


def load_chats(chats_path):
    if not chats_path.exists():
        print(f"ERROR: Chats file not found: {chats_path}")
        sys.exit(1)
    with open(chats_path, encoding="utf-8") as f:
        return {c["chat_id"]: c for c in json.load(f).get("chats", [])}


def load_analyses(analysis_path):
    if not analysis_path.exists():
        print(f"ERROR: Analysis file not found: {analysis_path}")
        sys.exit(1)
    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)
    label = data.get("label")
    analyses = {a["chat_id"]: a for a in data.get("analyses", [])}
    return analyses, label


def score_pair(chat, analysis):
    intent_match = chat["category"] == analysis["intent"]

    hidden_match = chat["has_hidden_dissatisfaction"] == analysis["has_hidden_dissatisfaction"]

    gt_mistake = chat.get("agent_mistake", "")
    pred_mistakes = analysis.get("agent_mistakes", [])
    if gt_mistake:
        mistake_match = gt_mistake in pred_mistakes
    else:
        mistake_match = len(pred_mistakes) == 0

    return {
        "intent_match": intent_match,
        "intent_pred": analysis["intent"],
        "hidden_match": hidden_match,
        "hidden_pred": analysis["has_hidden_dissatisfaction"],
        "hidden_gt": chat["has_hidden_dissatisfaction"],
        "mistake_match": mistake_match,
        "mistake_gt": gt_mistake,
        "mistake_pred": pred_mistakes,
        "satisfaction": analysis.get("customer_satisfaction", "?"),
        "quality_score": analysis.get("quality_score", 0),
    }


def format_intent_cell(result):
    if result["intent_match"]:
        return f"{GREEN}OK{RESET}"
    return f"{RED}FAIL{RESET} ({result['intent_pred']})"


def format_hidden_cell(result):
    if result["hidden_match"]:
        return f"{GREEN}OK{RESET}"
    return f"{RED}FAIL{RESET} (pred={result['hidden_pred']})"


def format_mistake_cell(result):
    if result["mistake_match"]:
        return f"{GREEN}OK{RESET}"
    if result["mistake_gt"]:
        return f"{RED}FAIL{RESET} (expected={result['mistake_gt']}, got={result['mistake_pred']})"
    return f"{RED}FAIL{RESET} (false alarm: {result['mistake_pred']})"


def print_detail_table(chats, results_by_version, version_names):
    chat_ids = list(chats.keys())

    for ver_name in version_names:
        results = results_by_version[ver_name]
        print(f"\n{BOLD}=== {ver_name} ==={RESET}")
        print(f"{'CHAT ID':<42} {'INTENT':<22} {'HIDDEN DISSAT.':<28} {'AGENT MISTAKE'}")
        print("-" * 110)

        for chat_id in chat_ids:
            if chat_id not in results:
                continue
            r = results[chat_id]
            print(f"{chat_id:<42} {format_intent_cell(r):<34} {format_hidden_cell(r):<40} {format_mistake_cell(r)}")

        print("-" * 110)


def count_matches(results, key):
    return sum(1 for r in results.values() if r[key])


def compute_hidden_confusion_matrix(results):
    return {
        "hidden_tp": sum(1 for r in results.values() if r["hidden_gt"] and r["hidden_pred"]),
        "hidden_fp": sum(1 for r in results.values() if not r["hidden_gt"] and r["hidden_pred"]),
        "hidden_fn": sum(1 for r in results.values() if r["hidden_gt"] and not r["hidden_pred"]),
        "hidden_tn": sum(1 for r in results.values() if not r["hidden_gt"] and not r["hidden_pred"]),
    }


def compute_mistake_breakdown(results):
    return {
        "mistake_false_alarms": sum(
            1 for r in results.values() if not r["mistake_gt"] and r["mistake_pred"]
        ),
        "mistake_missed": sum(
            1 for r in results.values() if r["mistake_gt"] and r["mistake_gt"] not in r["mistake_pred"]
        ),
        "mistake_wrong_type": sum(
            1 for r in results.values()
            if r["mistake_gt"] and r["mistake_pred"] and r["mistake_gt"] not in r["mistake_pred"]
        ),
    }


def compute_stats(results):
    total = len(results)
    if total == 0:
        return {}

    intent_correct = count_matches(results, "intent_match")
    hidden_correct = count_matches(results, "hidden_match")
    mistake_correct = count_matches(results, "mistake_match")

    all_pred_mistakes = []
    for r in results.values():
        all_pred_mistakes.extend(r["mistake_pred"])

    sat_counts = Counter(r["satisfaction"] for r in results.values())

    scores = [r["quality_score"] for r in results.values() if r["quality_score"]]
    avg_quality = sum(scores) / len(scores) if scores else 0

    stats = {
        "total": total,
        "intent_correct": intent_correct,
        "intent_pct": intent_correct / total,
        "hidden_correct": hidden_correct,
        "hidden_pct": hidden_correct / total,
        "mistake_correct": mistake_correct,
        "mistake_pct": mistake_correct / total,
        "mistake_distribution": dict(Counter(all_pred_mistakes).most_common()),
        "satisfaction": dict(sat_counts),
        "avg_quality_score": avg_quality,
    }
    stats.update(compute_hidden_confusion_matrix(results))
    stats.update(compute_mistake_breakdown(results))

    return stats


def print_accuracy_table(stats):
    t = stats["total"]
    print(f"\n  {'METRIC':<28} {'SCORE':>12} {'DETAIL'}")
    print(f"  {'-'*70}")
    print(f"  {'Intent classification':<28} {stats['intent_pct']:>11.0%}  {stats['intent_correct']}/{t}")
    print(f"  {'Hidden dissatisfaction':<28} {stats['hidden_pct']:>11.0%}  {stats['hidden_correct']}/{t}")
    print(f"  {'Agent mistake detection':<28} {stats['mistake_pct']:>11.0%}  {stats['mistake_correct']}/{t}")


def print_hidden_breakdown(stats):
    print(f"\n  Hidden dissatisfaction breakdown:")
    print(f"    TP (correctly detected)    {stats['hidden_tp']:>3}")
    print(f"    TN (correctly rejected)    {stats['hidden_tn']:>3}")
    print(f"    FP (over-predicted)        {stats['hidden_fp']:>3}")
    print(f"    FN (missed)                {stats['hidden_fn']:>3}")


def print_mistake_errors(stats):
    print(f"\n  Agent mistake errors:")
    print(f"    False alarms               {stats['mistake_false_alarms']:>3}")
    print(f"    Missed detections          {stats['mistake_missed']:>3}")
    print(f"    Wrong type                 {stats['mistake_wrong_type']:>3}")


def print_mistake_distribution(stats):
    if not stats["mistake_distribution"]:
        return
    print(f"\n  Predicted mistake distribution:")
    for m, c in stats["mistake_distribution"].items():
        bar = f"{CYAN}{'█' * c}{RESET}"
        print(f"    {m:<28} {c:>3}  {bar}")


def print_satisfaction_distribution(stats):
    print(f"\n  Satisfaction distribution:")
    for level in ["satisfied", "neutral", "unsatisfied"]:
        c = stats["satisfaction"].get(level, 0)
        bar = f"{YELLOW}{'█' * c}{RESET}"
        print(f"    {level:<28} {c:>3}  {bar}")

    print(f"\n  Avg quality score:           {stats['avg_quality_score']:.2f}")


def print_stats(stats, ver_name):
    t = stats["total"]
    print(f"\n{BOLD}--- {ver_name} ({t} pairs) ---{RESET}")

    print_accuracy_table(stats)
    print_hidden_breakdown(stats)
    print_mistake_errors(stats)
    print_mistake_distribution(stats)
    print_satisfaction_distribution(stats)


def format_delta_int(first, last, lower_is_better=False):
    delta = last - first
    if delta == 0:
        return f"  {delta}"
    if lower_is_better:
        color = GREEN if delta < 0 else RED
    else:
        color = GREEN if delta > 0 else RED
    sign = "+" if delta > 0 else ""
    return f"  {color}{sign}{delta}{RESET}"


def format_delta_pct(first, last):
    delta = last - first
    if delta == 0:
        return f"  {delta:.0%}"
    color = GREEN if delta > 0 else RED
    sign = "+" if delta > 0 else ""
    return f"  {color}{sign}{delta:.0%}{RESET}"


def print_accuracy_comparison(all_stats, version_names):
    metrics = [
        ("Intent", "intent_pct", "intent_correct"),
        ("Hidden dissatisfaction", "hidden_pct", "hidden_correct"),
        ("Agent mistakes", "mistake_pct", "mistake_correct"),
    ]

    for label, pct_key, count_key in metrics:
        row = f"  {label:<28}"
        values = []
        for name in version_names:
            s = all_stats[name]
            row += f" {s[pct_key]:>11.0%}"
            values.append(s[pct_key])
        if len(version_names) > 1:
            row += format_delta_pct(values[0], values[-1])
        print(row)


def print_error_comparison(all_stats, version_names):
    print()
    error_metrics = [
        ("Hidden FP (over-predict)", "hidden_fp"),
        ("Hidden FN (missed)", "hidden_fn"),
        ("Mistake false alarms", "mistake_false_alarms"),
        ("Mistake missed", "mistake_missed"),
    ]
    for label, key in error_metrics:
        row = f"  {label:<28}"
        values = []
        for name in version_names:
            s = all_stats[name]
            row += f" {s[key]:>12}"
            values.append(s[key])
        if len(version_names) > 1:
            row += format_delta_int(values[0], values[-1], lower_is_better=True)
        print(row)


def print_comparison_table(all_stats, version_names):
    print(f"\n{BOLD}{'=' * 80}")
    print("COMPARISON ACROSS VERSIONS")
    print(f"{'=' * 80}{RESET}")

    header = f"  {'METRIC':<28}"
    for name in version_names:
        header += f" {name:>12}"
    if len(version_names) > 1:
        header += f" {'DELTA':>10}"
    print(header)
    print(f"  {'-' * (28 + 13 * len(version_names) + (11 if len(version_names) > 1 else 0))}")

    print_accuracy_comparison(all_stats, version_names)
    print_error_comparison(all_stats, version_names)

    print()


def version_display_name(analysis_path, label):
    if label:
        return label
    return analysis_path.stem


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate analysis versions against ground truth.",
        usage="python evaluate.py [--chats CHATS] analysis_v1.json [analysis_v2.json ...]",
    )
    parser.add_argument("--chats", type=Path, default=CHATS_OUTPUT_PATH, help="Ground truth chats file")
    parser.add_argument("analyses", type=Path, nargs="+", help="One or more analysis JSON files to evaluate")
    args = parser.parse_args()

    chats = load_chats(args.chats)
    print(f"Loaded {len(chats)} ground truth chats.")

    results_by_version = {}
    all_stats = {}
    version_names = []

    for analysis_path in args.analyses:
        analyses, label = load_analyses(analysis_path)
        ver_name = version_display_name(analysis_path, label)
        version_names.append(ver_name)

        print(f"  {ver_name}: {len(analyses)} analyses")

        results = {}
        for chat_id, chat in chats.items():
            analysis = analyses.get(chat_id)
            if analysis:
                results[chat_id] = score_pair(chat, analysis)

        results_by_version[ver_name] = results
        all_stats[ver_name] = compute_stats(results)

    print_detail_table(chats, results_by_version, version_names)

    for ver_name in version_names:
        print_stats(all_stats[ver_name], ver_name)

    if len(version_names) >= 1:
        print_comparison_table(all_stats, version_names)


if __name__ == "__main__":
    main()
