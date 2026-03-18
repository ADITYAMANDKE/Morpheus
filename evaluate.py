"""
evaluate.py
───────────
Evaluation metrics for Dialogue State Tracking.

Implements:
    - Joint Goal Accuracy (JGA) on accumulated dialogue states (DST JGA)
    - Turn-Level Belief JGA (TLB JGA) — per-turn accuracy on changes only
    - Slot-value F1 — used internally for contrastive pair construction

References:
    - JGA: Henderson et al. (2014) — "The Second Dialog State Tracking Challenge"
    - TLB JGA: Dey et al. (2022) — "Towards Fair Evaluation of Dialogue State
      Tracking by Flexible Incorporation of Turn-Level Performances"
    - Slot-value similarity: Hu et al. (2022) IC-DST paper

Usage:
    python evaluate.py --pred results/slm_preds.json --gold data/processed/val.jsonl
"""

import argparse
import json
from pathlib import Path

from data_preprocessing import string_to_state, state_to_string


# ─── Core Metrics ─────────────────────────────────────────────────────────────

def states_match(pred: dict[str, str], gold: dict[str, str]) -> bool:
    """
    Checks exact match between two flat state dicts.
    Both must have identical keys AND values (case-insensitive values).

    Args:
        pred: Predicted state dict.
        gold: Ground-truth state dict.

    Returns:
        True if all slots and values match exactly.
    """
    if set(pred.keys()) != set(gold.keys()):
        return False
    return all(pred[k].strip().lower() == gold[k].strip().lower() for k in gold)


def compute_tlb_jga(
    pred_strings: list[str],
    gold_strings: list[str],
) -> float:
    """
    Computes Turn-Level Belief Joint Goal Accuracy (TLB JGA).

    TLB JGA measures accuracy on the CHANGE at each turn (not the accumulated
    state). A turn is correct only if ALL predicted slot-value changes exactly
    match the ground truth changes.

    Args:
        pred_strings: List of predicted TLB strings ("slot = val | ...").
        gold_strings: List of gold TLB strings.

    Returns:
        TLB JGA as a fraction in [0, 1].
    """
    assert len(pred_strings) == len(gold_strings), \
        f"Length mismatch: {len(pred_strings)} preds vs {len(gold_strings)} golds"

    correct = 0
    for pred_str, gold_str in zip(pred_strings, gold_strings):
        pred_state = string_to_state(pred_str)
        gold_state = string_to_state(gold_str)
        if states_match(pred_state, gold_state):
            correct += 1

    return correct / len(pred_strings) if pred_strings else 0.0


def compute_dst_jga(
    pred_dst_strings: list[str],
    gold_dst_strings: list[str],
) -> float:
    """
    Computes Joint Goal Accuracy on ACCUMULATED dialogue states (DST JGA).

    The standard DST metric: a turn is correct only if every slot-value pair
    in the full accumulated dialogue state is exactly right.

    Args:
        pred_dst_strings: List of predicted accumulated DST strings.
        gold_dst_strings: List of gold accumulated DST strings.

    Returns:
        DST JGA as a fraction in [0, 1].
    """
    assert len(pred_dst_strings) == len(gold_dst_strings)

    correct = 0
    for pred_str, gold_str in zip(pred_dst_strings, gold_dst_strings):
        pred_state = string_to_state(pred_str)
        gold_state = string_to_state(gold_str)
        if states_match(pred_state, gold_state):
            correct += 1

    return correct / len(pred_dst_strings) if pred_dst_strings else 0.0


def compute_slot_f1(
    pred: dict[str, str],
    gold: dict[str, str],
) -> float:
    """
    Computes combined slot-value F1 score between two state dicts.

    From Section 3.2 of the paper:
        Sim(TLB_a, TLB_b) = F1_slot_value + F1_slot - 1

    Where:
        F1_slot_value: F1 over (slot, value) pairs
        F1_slot      : F1 over slot names only

    Args:
        pred: Predicted state dict.
        gold: Gold state dict.

    Returns:
        Similarity score in [-1, 1]. Returns 1.0 if both are empty (both correct).
    """
    if not pred and not gold:
        return 1.0  # Both empty → perfect match
    if not pred or not gold:
        return -1.0 if (pred or gold) else 1.0

    # Slot-value F1
    pred_sv = set(f"{k}={v.lower()}" for k, v in pred.items())
    gold_sv = set(f"{k}={v.lower()}" for k, v in gold.items())
    f1_sv = _f1(pred_sv, gold_sv)

    # Slot-only F1
    pred_s = set(pred.keys())
    gold_s = set(gold.keys())
    f1_s = _f1(pred_s, gold_s)

    return f1_sv + f1_s - 1.0


def _f1(pred_set: set, gold_set: set) -> float:
    """Computes F1 between two sets."""
    if not pred_set and not gold_set:
        return 1.0
    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall    = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall)


# ─── Detailed Report ──────────────────────────────────────────────────────────

def evaluate_predictions(
    pred_tlbs: list[dict[str, str]],
    gold_tlbs: list[dict[str, str]],
    pred_dsts: list[dict[str, str]],
    gold_dsts: list[dict[str, str]],
    print_errors: bool = False,
) -> dict:
    """
    Full evaluation report over all turns.

    Args:
        pred_tlbs   : List of predicted TLB dicts (turn changes).
        gold_tlbs   : List of gold TLB dicts.
        pred_dsts   : List of predicted accumulated DST dicts.
        gold_dsts   : List of gold accumulated DST dicts.
        print_errors: If True, prints the first 5 errors per metric.

    Returns:
        Dict with tlb_jga, dst_jga, slot_f1, and per-domain breakdown.
    """
    n = len(pred_tlbs)
    assert n == len(gold_tlbs) == len(pred_dsts) == len(gold_dsts)

    # TLB JGA
    tlb_correct = [states_match(p, g) for p, g in zip(pred_tlbs, gold_tlbs)]
    tlb_jga = sum(tlb_correct) / n if n else 0.0

    # DST JGA
    dst_correct = [states_match(p, g) for p, g in zip(pred_dsts, gold_dsts)]
    dst_jga = sum(dst_correct) / n if n else 0.0

    # Slot F1 (average over turns)
    slot_f1_scores = [compute_slot_f1(p, g) for p, g in zip(pred_tlbs, gold_tlbs)]
    avg_slot_f1 = sum(slot_f1_scores) / n if n else 0.0

    # Per-domain TLB accuracy
    domain_correct: dict[str, list[bool]] = {}
    for pred_tlb, gold_tlb, correct in zip(pred_tlbs, gold_tlbs, tlb_correct):
        domains = set()
        for key in list(pred_tlb.keys()) + list(gold_tlb.keys()):
            domain = key.split("-")[0] if "-" in key else "unknown"
            domains.add(domain)
        for domain in domains:
            domain_correct.setdefault(domain, []).append(correct)

    domain_accuracy = {
        domain: round(sum(vals) / len(vals) * 100, 2)
        for domain, vals in domain_correct.items()
    }

    if print_errors:
        print("\n[First 5 TLB errors]")
        err_count = 0
        for i, (p, g, c) in enumerate(zip(pred_tlbs, gold_tlbs, tlb_correct)):
            if not c and err_count < 5:
                print(f"  Turn {i}: pred={p}  gold={g}")
                err_count += 1

    results = {
        "tlb_jga": round(tlb_jga * 100, 2),
        "dst_jga": round(dst_jga * 100, 2),
        "avg_slot_f1": round(avg_slot_f1 * 100, 2),
        "n_turns": n,
        "per_domain_tlb_accuracy": domain_accuracy,
    }

    print(f"\n{'='*50}")
    print(f"  TLB JGA  : {results['tlb_jga']:.2f}%")
    print(f"  DST JGA  : {results['dst_jga']:.2f}%")
    print(f"  Slot F1  : {results['avg_slot_f1']:.2f}%")
    print(f"  N turns  : {n}")
    print(f"\n  Per-domain TLB accuracy:")
    for domain, acc in sorted(domain_accuracy.items(), key=lambda x: -x[1]):
        print(f"    {domain:<15}: {acc:.2f}%")
    print(f"{'='*50}\n")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate DST predictions")
    parser.add_argument("--pred", required=True,
                        help="JSON file with predictions: list of {tlb: str, dst: str}")
    parser.add_argument("--gold", required=True,
                        help="Gold .jsonl file from data_preprocessing.py")
    parser.add_argument("--print_errors", action="store_true")
    args = parser.parse_args()

    with open(args.pred) as f:
        preds = json.load(f)

    gold_examples = [json.loads(l) for l in open(args.gold) if l.strip()]

    assert len(preds) == len(gold_examples), \
        f"Prediction count ({len(preds)}) doesn't match gold count ({len(gold_examples)})"

    pred_tlbs = [string_to_state(p.get("tlb", "")) for p in preds]
    pred_dsts  = [string_to_state(p.get("dst", "")) for p in preds]
    gold_tlbs  = [string_to_state(ex["target_text"]) for ex in gold_examples]
    gold_dsts  = [string_to_state(ex["dst_text"]) for ex in gold_examples]

    results = evaluate_predictions(pred_tlbs, gold_tlbs, pred_dsts, gold_dsts,
                                   print_errors=args.print_errors)
    output_path = Path(args.pred).with_suffix(".eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {output_path}")


if __name__ == "__main__":
    main()
