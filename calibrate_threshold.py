"""
calibrate_threshold.py
──────────────────────
Offline threshold calibration for domain-aware routing.

Sweeps reliability_threshold values and simulates the routing impact
using existing holdout predictions. No API calls are made.

Usage:
    python calibrate_threshold.py --config config.yaml
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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Calibrate domain-aware routing threshold")
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    rtr_cfg = config.get("router", {})
    pool_dir = Path(paths.get("expert_pool_dir", "./data/expert_pools"))
    proc_dir = Path(paths.get("processed_dir", "./data/processed"))

    # Load expert pools
    with open(pool_dir / "slm_pool.json") as f:
        slm_pool = json.load(f)
    with open(pool_dir / "llm_pool.json") as f:
        llm_pool = json.load(f)

    # Compute domain reliability
    reliability = compute_domain_reliability(slm_pool, llm_pool)
    print("=" * 60)
    print("Domain Reliability Scores (from expert pools)")
    print("=" * 60)
    for domain, score in sorted(reliability.items(), key=lambda x: x[1]):
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        print(f"  {domain:15s}  {bar}  {score:.3f}")
    print()

    # Load holdout examples
    holdout = load_jsonl(str(proc_dir / "holdout.jsonl"))

    # Load domain keywords from config
    domain_keywords = rtr_cfg.get("domain_keywords", None)

    # Reconstruct dialogues to simulate per-turn routing
    dialogues: dict[str, list] = {}
    for ex in holdout:
        dialogues.setdefault(ex["dialogue_id"], []).append(ex)
    for did in dialogues:
        dialogues[did].sort(key=lambda x: x["turn_idx"])

    total_turns = sum(len(turns) for turns in dialogues.values())

    # Sweep thresholds
    print("=" * 60)
    print(f"Threshold Sweep ({total_turns} holdout turns)")
    print("=" * 60)
    print(f"  {'Threshold':>10s}  {'Overrides':>10s}  {'Override %':>10s}  {'Triggered Domains'}")
    print(f"  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 30}")

    for threshold in args.thresholds:
        override_count = 0
        triggered_domains: dict[str, int] = {}

        for did, turns in dialogues.items():
            accumulated_dst: dict[str, str] = {}
            for turn in turns:
                new_domains = detect_domain_switch(
                    accumulated_dst,
                    turn["user_utt"],
                    domain_keywords,
                )
                if should_override_to_llm(new_domains, reliability, threshold):
                    override_count += 1
                    for d in new_domains:
                        triggered_domains[d] = triggered_domains.get(d, 0) + 1

                # Simulate DST accumulation with ground truth
                gold_tlb = string_to_state(turn["target_text"])
                accumulated_dst.update(gold_tlb)

        override_pct = override_count / total_turns * 100 if total_turns else 0
        domain_str = ", ".join(
            f"{d}({c})" for d, c in sorted(triggered_domains.items(), key=lambda x: -x[1])
        ) or "none"
        print(f"  {threshold:>10.2f}  {override_count:>10d}  {override_pct:>9.1f}%  {domain_str}")

    print()
    print("Choose a threshold that overrides ~5-15% of turns for best results.")


if __name__ == "__main__":
    main()
