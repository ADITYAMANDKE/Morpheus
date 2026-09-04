"""
calibrate_threshold.py
──────────────────────
Offline threshold calibration for the domain-aware overlay (paper Section 2.5).

    "We sweep τ ∈ {0.30, 0.40, 0.50, 0.60, 0.70, 0.80} on the holdout set,
     simulating the override rate without making any new API calls by reusing
     cached predictions. We choose the smallest τ whose override rate lies in
     the 5–15% band."

What this script does:
    1. Computes per-domain reliability r_d (Equation 1) from the expert pools.
    2. Replays every holdout dialogue turn by turn. The accumulated DST_{t-1}
       used for domain-switch detection is rebuilt from the *cached* holdout
       predictions of the default expert (the SLM), i.e. exactly what the
       online detector would have seen — no API calls are made. Falls back to
       gold TLBs if cached predictions are not available.
    3. For each τ, reports the override rate, the domains that triggered it,
       and a simulated TLB JGA (override → cached LLM prediction, otherwise →
       cached SLM prediction; this isolates the effect of the override from
       the KNN vote).
    4. Selects the smallest τ whose override rate lies in the configured band
       (default 5–15%) and optionally writes it back to config.yaml.

Usage:
    python calibrate_threshold.py --config config.yaml
    python calibrate_threshold.py --config config.yaml --write   # update reliability_threshold
"""

import argparse
import json
from pathlib import Path

import yaml

from data_preprocessing import load_jsonl, string_to_state
from domain_router import (
    compute_domain_reliability,
    detect_domain_switch,
    should_override_to_llm,
)
from evaluate import states_match


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_json_if_exists(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def sweep(
    dialogues: dict[str, list[dict]],
    reliability: dict[str, float],
    domain_keywords: dict | None,
    thresholds: list[float],
    slm_preds: dict[str, str] | None,
    llm_preds: dict[str, str] | None,
) -> list[dict]:
    """Simulates the override for every τ. Returns one summary dict per τ."""
    total_turns = sum(len(t) for t in dialogues.values())
    rows = []

    for tau in thresholds:
        n_override = 0
        n_correct = 0
        triggered: dict[str, int] = {}

        for did, turns in dialogues.items():
            acc_dst: dict[str, str] = {}
            for turn in turns:
                key = f"{turn['dialogue_id']}_{turn['turn_idx']}"
                gold_tlb = string_to_state(turn["target_text"])

                new_domains = detect_domain_switch(acc_dst, turn["user_utt"], domain_keywords)
                override = should_override_to_llm(new_domains, reliability, tau)
                if override:
                    n_override += 1
                    for d in new_domains:
                        if reliability.get(d, 0.5) < tau:
                            triggered[d] = triggered.get(d, 0) + 1

                # Which cached prediction would have been used for this turn?
                if slm_preds is not None and llm_preds is not None:
                    pred_str = llm_preds.get(key, "") if override else slm_preds.get(key, "")
                    pred_tlb = string_to_state(pred_str)
                    if states_match(pred_tlb, gold_tlb):
                        n_correct += 1
                    # The online detector sees the *predicted* accumulated state.
                    # The default expert (SLM) drives accumulation, as in a
                    # steady-state deployment; overrides use the LLM output.
                    acc_dst.update(pred_tlb)
                else:
                    acc_dst.update(gold_tlb)

        rows.append({
            "threshold": tau,
            "overrides": n_override,
            "override_rate": n_override / total_turns if total_turns else 0.0,
            "triggered_domains": dict(sorted(triggered.items(), key=lambda x: -x[1])),
            "simulated_tlb_jga": (n_correct / total_turns) if (slm_preds and total_turns) else None,
        })
    return rows


def select_threshold(rows: list[dict], band: tuple[float, float]) -> float | None:
    """Smallest τ whose override rate lies inside [lo, hi] (paper Section 2.5)."""
    lo, hi = band
    for r in sorted(rows, key=lambda r: r["threshold"]):
        if lo <= r["override_rate"] <= hi:
            return r["threshold"]
    return None


def main():
    parser = argparse.ArgumentParser(description="Calibrate the domain-aware override threshold τ")
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--thresholds", nargs="+", type=float, default=None,
                        help="Override the sweep grid from config")
    parser.add_argument("--write", action="store_true",
                        help="Write the selected τ back to router.reliability_threshold in config")
    parser.add_argument("--out", default=None, help="Save sweep table as JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    rtr_cfg = config.get("router", {})
    cal_cfg = rtr_cfg.get("calibration", {})
    pool_dir = Path(paths.get("expert_pool_dir", "./data/expert_pools"))
    proc_dir = Path(paths.get("processed_dir", "./data/processed"))

    thresholds = args.thresholds or cal_cfg.get("thresholds", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    band = tuple(cal_cfg.get("override_rate_band", [0.05, 0.15]))

    # ── Step 1: reliability from the expert pools (Equation 1) ──
    with open(pool_dir / "slm_pool.json") as f:
        slm_pool = json.load(f)
    with open(pool_dir / "llm_pool.json") as f:
        llm_pool = json.load(f)
    reliability = compute_domain_reliability(slm_pool, llm_pool)

    print("=" * 64)
    print("Per-domain SLM reliability r_d  (Equation 1, from expert pools)")
    print("=" * 64)
    for domain, score in sorted(reliability.items(), key=lambda x: x[1]):
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        print(f"  {domain:15s}  {bar}  {score:.3f}")
    print()

    # ── Step 2: holdout dialogues + cached predictions ──
    holdout = load_jsonl(str(proc_dir / "holdout.jsonl"))
    dialogues: dict[str, list] = {}
    for ex in holdout:
        dialogues.setdefault(ex["dialogue_id"], []).append(ex)
    for did in dialogues:
        dialogues[did].sort(key=lambda x: x["turn_idx"])
    total_turns = sum(len(t) for t in dialogues.values())

    slm_preds = _load_json_if_exists(pool_dir / "slm_holdout_preds.json")
    llm_preds = _load_json_if_exists(pool_dir / "llm_holdout_preds.json")
    if slm_preds and llm_preds:
        print(f"Using cached holdout predictions ({len(slm_preds)} SLM, {len(llm_preds)} LLM) — no API calls.")
    else:
        print("Cached holdout predictions not found — accumulating state from gold TLBs.")
    print()

    # ── Step 3: sweep ──
    rows = sweep(dialogues, reliability, rtr_cfg.get("domain_keywords"), thresholds, slm_preds, llm_preds)

    print("=" * 64)
    print(f"Threshold sweep over {total_turns} holdout turns")
    print("=" * 64)
    hdr = f"  {'τ':>5s}  {'overrides':>9s}  {'rate':>7s}  {'sim. TLB JGA':>12s}  triggered domains"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in rows:
        jga = f"{r['simulated_tlb_jga'] * 100:11.2f}%" if r["simulated_tlb_jga"] is not None else f"{'n/a':>12s}"
        doms = ", ".join(f"{d}({c})" for d, c in r["triggered_domains"].items()) or "none"
        print(f"  {r['threshold']:5.2f}  {r['overrides']:9d}  {r['override_rate'] * 100:6.1f}%  {jga}  {doms}")
    print()

    # ── Step 4: selection rule ──
    chosen = select_threshold(rows, band)
    print(f"Selection rule: smallest τ with override rate in [{band[0] * 100:.0f}%, {band[1] * 100:.0f}%]")
    if chosen is None:
        print("  → No τ in the band. Widen the grid or the band, or inspect the table above.")
    else:
        print(f"  → Selected τ = {chosen:.2f} (config currently: {rtr_cfg.get('reliability_threshold')})")
        if args.write:
            config["router"]["reliability_threshold"] = float(chosen)
            with open(args.config, "w") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
            print(f"  → Wrote reliability_threshold: {chosen:.2f} to {args.config}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"reliability": reliability, "band": band, "selected": chosen, "sweep": rows}, f, indent=2)
        print(f"Sweep saved → {args.out}")


if __name__ == "__main__":
    main()
