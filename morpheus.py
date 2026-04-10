"""
morpheus.py
───────────
Morpheus: Full pipeline orchestrator combining Prompt-DST (SLM),
IC-DST (LLM), and the retrieval-based router.

At inference time (Figure 1 of the paper):
    1. Encode the input triplet (DST_{t-1}, A_{t-1}, U_t) via SenBERT.
    2. Retrieve top-K nearest neighbours from SLM and LLM expert pools.
    3. Majority vote → assign to SLM (Prompt-DST) or LLM (IC-DST).
    4. Run the selected expert and return TLB_t.
    5. Aggregate TLB_t into the running DST.

Usage:
    # Full evaluation
    python morpheus.py eval --config config.yaml

    # Single-turn interactive inference
    python morpheus.py infer --config config.yaml
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import yaml

from data_preprocessing import (
    load_jsonl,
    string_to_state,
    state_to_string,
    format_input,
)
from evaluate import evaluate_predictions, compute_tlb_jga, compute_dst_jga
from prompt_dst import PromptDST
from ic_dst import ICDST
from router import Retriever, encode_triplet
from domain_router import (
    compute_domain_reliability,
    detect_domain_switch,
    should_override_to_llm,
)


# ─── Morpheus ─────────────────────────────────────────────────────────────────

class Morpheus:
    """
    Full Morpheus pipeline: router + SLM expert + LLM expert.

    The router dynamically dispatches each turn to the most appropriate expert
    based on semantic similarity to the expert pools.
    """

    def __init__(self, config: dict):
        self.config = config
        self.rtr_cfg = config.get("router", {})
        self.paths   = config.get("paths", {})

        # Components (loaded lazily)
        self.retriever: Optional[Retriever] = None
        self.slm: Optional[PromptDST] = None
        self.llm: Optional[ICDST] = None

        # Routing stats
        self.routing_log: list[dict] = []

        # Domain-aware routing
        self.domain_aware = self.rtr_cfg.get("domain_aware", False)
        self.domain_reliability: dict[str, float] = {}
        self.domain_keywords = self.rtr_cfg.get("domain_keywords", None)
        self.reliability_threshold = self.rtr_cfg.get("reliability_threshold", 0.5)
        self.domain_override_count = 0

        # Force all turns to one expert (for baseline comparison)
        self.force_expert: Optional[str] = None

    def load(
        self,
        retriever_path: Optional[str] = None,
        slm_checkpoint: Optional[str] = None,
        exemplar_examples: Optional[list[dict]] = None,
    ):
        """
        Loads all three components: router, SLM expert, LLM expert.

        Args:
            retriever_path    : Path to saved Retriever. If None, uses off-the-shelf SenBERT.
            slm_checkpoint    : Path to fine-tuned FLAN-T5 checkpoint.
            exemplar_examples : Training examples for IC-DST exemplar pool.
        """
        pool_dir  = Path(self.paths.get("expert_pool_dir", "./data/expert_pools"))
        model_dir = Path(self.paths.get("model_dir", "./models/prompt_dst"))

        # ── Router ──
        r_path = retriever_path or str(pool_dir / "retriever")
        if Path(r_path).exists():
            print("[Morpheus] Loading fine-tuned retriever...")
            self.retriever = Retriever.load(r_path)
        else:
            print("[Morpheus] Using off-the-shelf SenBERT retriever...")
            backbone = self.rtr_cfg.get("retriever_backbone",
                                        "sentence-transformers/all-mpnet-base-v2")
            self.retriever = Retriever(backbone=backbone)

            # Index expert pools
            slm_pool_path = pool_dir / "slm_pool.json"
            llm_pool_path = pool_dir / "llm_pool.json"
            if slm_pool_path.exists() and llm_pool_path.exists():
                with open(slm_pool_path) as f:
                    slm_pool = json.load(f)
                with open(llm_pool_path) as f:
                    llm_pool = json.load(f)
                self.retriever.index_pools(slm_pool, llm_pool)
            else:
                print("[Morpheus] WARNING: Expert pools not found. Run router.py build_pools first.")

        # ── Domain Reliability (offline, once) ──
        if self.domain_aware:
            slm_pool_path = pool_dir / "slm_pool.json"
            llm_pool_path = pool_dir / "llm_pool.json"
            if slm_pool_path.exists() and llm_pool_path.exists():
                with open(slm_pool_path) as f:
                    slm_pool_data = json.load(f)
                with open(llm_pool_path) as f:
                    llm_pool_data = json.load(f)
                self.domain_reliability = compute_domain_reliability(
                    slm_pool_data, llm_pool_data
                )
                print(f"[Morpheus] Domain reliability: {self.domain_reliability}")
            else:
                print("[Morpheus] WARNING: Cannot compute domain reliability — pools not found.")

        # ── SLM Expert (Prompt-DST) ──
        ckpt = slm_checkpoint or str(model_dir / "best")
        print(f"[Morpheus] Loading SLM expert from: {ckpt}")
        self.slm = PromptDST(self.config)
        self.slm.load(ckpt if Path(ckpt).exists() else None)

        # ── LLM Expert (IC-DST) ──
        print("[Morpheus] Initialising LLM expert (IC-DST)...")
        self.llm = ICDST(self.config)
        if exemplar_examples:
            self.llm.load_exemplar_pool(exemplar_examples)

        print("[Morpheus] All components loaded.")
        return self

    def predict_turn(
        self,
        prev_dst: dict[str, str],
        agent_utt: str,
        user_utt: str,
        dialogue_id: Optional[str] = None,
    ) -> tuple[str, dict[str, str], str]:
        """
        Routes and predicts the TLB for a single dialogue turn.

        Args:
            prev_dst    : Previous accumulated dialogue state.
            agent_utt   : Previous agent utterance.
            user_utt    : Current user utterance.
            dialogue_id : Source dialogue ID (for IC-DST exemplar exclusion).

        Returns:
            Tuple of (assigned_expert, predicted_tlb_dict, predicted_tlb_string).
        """
        assert self.retriever and self.slm and self.llm, "Call .load() first"

        domain_overridden = False

        # Forced expert mode (bypass router entirely)
        if self.force_expert:
            expert = self.force_expert
            routing_details = {"assigned": expert, "reason": "forced"}

        # Step 0: Domain-switch override (before KNN)
        elif self.domain_aware and self.domain_reliability:
            new_domains = detect_domain_switch(
                prev_dst, user_utt, self.domain_keywords
            )
            if should_override_to_llm(
                new_domains, self.domain_reliability, self.reliability_threshold
            ):
                expert = "llm"
                domain_overridden = True
                routing_details = {
                    "assigned": "llm",
                    "reason": "domain_switch_override",
                    "new_domains": list(new_domains),
                    "reliability_scores": {
                        d: self.domain_reliability.get(d, 0.5)
                        for d in new_domains
                    },
                }
                self.domain_override_count += 1
            else:
                expert, routing_details = self.retriever.route(
                    prev_dst, agent_utt, user_utt,
                    top_k=self.rtr_cfg.get("top_k", 10),
                    tie_break=self.rtr_cfg.get("tie_break", "slm"),
                )

        # Step 1 & 2: Route via retriever
        else:
            expert, routing_details = self.retriever.route(
                prev_dst, agent_utt, user_utt,
                top_k=self.rtr_cfg.get("top_k", 10),
                tie_break=self.rtr_cfg.get("tie_break", "slm"),
            )

        self.routing_log.append({
            "expert": expert,
            "domain_overridden": domain_overridden,
            **routing_details,
        })

        # Step 3: Execute assigned expert
        if expert == "slm":
            input_text = format_input(prev_dst, agent_utt, user_utt)
            pred_str = self.slm.predict_turn(input_text)
        else:
            _, pred_str = self.llm.predict_turn(
                prev_dst=prev_dst,
                agent_utt=agent_utt,
                user_utt=user_utt,
                dialogue_id=dialogue_id,
            )

        pred_tlb = string_to_state(pred_str)
        return expert, pred_tlb, pred_str

    def predict_dialogue(
        self,
        turns: list[dict],
    ) -> tuple[list[dict], dict, list[str]]:
        """
        Runs Morpheus over all turns of a dialogue.

        Args:
            turns: Sorted list of turn dicts for one dialogue.

        Returns:
            - predicted_tlbs  : List of predicted TLB dicts per turn.
            - final_dst       : Final accumulated dialogue state.
            - expert_sequence : List of assigned experts per turn ("slm"|"llm").
        """
        predicted_tlbs: list[dict]  = []
        expert_sequence: list[str]  = []
        accumulated_dst: dict[str, str] = {}
        did = turns[0]["dialogue_id"] if turns else None

        for turn in turns:
            expert, pred_tlb, _ = self.predict_turn(
                prev_dst=accumulated_dst,
                agent_utt=turn["agent_utt"],
                user_utt=turn["user_utt"],
                dialogue_id=did,
            )
            predicted_tlbs.append(pred_tlb)
            expert_sequence.append(expert)
            accumulated_dst.update(pred_tlb)  # Replace updated slots

        return predicted_tlbs, accumulated_dst, expert_sequence

    def evaluate(
        self,
        examples: list[dict],
        max_dialogues: Optional[int] = None,
    ) -> dict:
        """
        Full evaluation of Morpheus on a set of turn examples.

        Reports TLB JGA, DST JGA, SLM assignment ratio, and FLOPs estimate.

        Args:
            examples      : Turn-level examples from load_jsonl().
            max_dialogues : Limit evaluation for speed/cost.

        Returns:
            Full results dict.
        """
        assert self.retriever and self.slm and self.llm, "Call .load() first"

        # Group by dialogue
        dialogues: dict[str, list] = {}
        for ex in examples:
            dialogues.setdefault(ex["dialogue_id"], []).append(ex)
        for did in dialogues:
            dialogues[did].sort(key=lambda x: x["turn_idx"])

        if max_dialogues:
            dial_ids = list(dialogues.keys())[:max_dialogues]
            dialogues = {k: dialogues[k] for k in dial_ids}

        self.routing_log = []
        all_pred_tlbs, all_gold_tlbs = [], []
        all_pred_dsts, all_gold_dsts = [], []

        start = time.time()
        for i, (did, turns) in enumerate(dialogues.items()):
            if i % 5 == 0:
                print(f"[Morpheus] Dialogue {i+1}/{len(dialogues)}: {did}")

            accumulated_dst: dict[str, str] = {}
            for turn in turns:
                expert, pred_tlb, pred_str = self.predict_turn(
                    prev_dst=accumulated_dst,
                    agent_utt=turn["agent_utt"],
                    user_utt=turn["user_utt"],
                    dialogue_id=did,
                )
                accumulated_dst.update(pred_tlb)

                all_pred_tlbs.append(pred_tlb)
                all_gold_tlbs.append(string_to_state(turn["target_text"]))
                all_pred_dsts.append(dict(accumulated_dst))
                all_gold_dsts.append(string_to_state(turn["dst_text"]))

        elapsed = time.time() - start

        # Compute metrics
        results = evaluate_predictions(
            all_pred_tlbs, all_gold_tlbs,
            all_pred_dsts, all_gold_dsts,
        )

        # Routing stats
        n_turns = len(self.routing_log)
        n_slm   = sum(1 for r in self.routing_log if r["expert"] == "slm")
        n_llm   = n_turns - n_slm
        n_domain_overrides = sum(
            1 for r in self.routing_log if r.get("domain_overridden", False)
        )

        results.update({
            "slm_assignment_ratio": round(n_slm / n_turns * 100, 1) if n_turns else 0,
            "llm_assignment_ratio": round(n_llm / n_turns * 100, 1) if n_turns else 0,
            "n_slm_calls": n_slm,
            "n_llm_calls": n_llm,
            "n_domain_overrides": n_domain_overrides,
            "domain_override_ratio": round(n_domain_overrides / n_turns * 100, 1) if n_turns else 0,
            "domain_reliability": self.domain_reliability,
            "eval_time_seconds": round(elapsed, 1),
        })

        print(f"\n[Routing] SLM: {results['slm_assignment_ratio']}% | "
              f"LLM: {results['llm_assignment_ratio']}%")
        if n_domain_overrides:
            print(f"[Domain Override] {n_domain_overrides} turns ({results['domain_override_ratio']}%)")
        print(f"[Timing] {elapsed:.1f}s for {n_turns} turns")

        return results

    def routing_summary(self) -> dict:
        """Returns a summary of routing decisions made so far."""
        if not self.routing_log:
            return {}
        n = len(self.routing_log)
        n_slm = sum(1 for r in self.routing_log if r["expert"] == "slm")
        return {
            "total_turns": n,
            "slm_turns": n_slm,
            "llm_turns": n - n_slm,
            "slm_pct": round(n_slm / n * 100, 1),
        }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Morpheus Full Pipeline")
    parser.add_argument("mode", choices=["eval", "infer"])
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--max_dialogues", type=int, default=None)
    parser.add_argument("--force_expert", choices=["slm", "llm"], default=None,
                        help="Force all turns to one expert (for baseline comparison)")
    parser.add_argument("--results_out", default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    paths  = config.get("paths", {})
    proc_dir = Path(paths.get("processed_dir", "./data/processed"))

    # Load training examples for IC-DST exemplar pool
    train_ex = load_jsonl(str(proc_dir / "train.jsonl"))

    pipeline = Morpheus(config)
    if args.force_expert:
        pipeline.force_expert = args.force_expert
        print(f"[Morpheus] FORCED MODE: all turns → {args.force_expert.upper()}")
    pipeline.load(exemplar_examples=train_ex)

    if args.mode == "eval":
        val_ex  = load_jsonl(str(proc_dir / "val.jsonl"))
        results = pipeline.evaluate(val_ex, max_dialogues=args.max_dialogues)

        out_path = args.results_out or str(
            Path(paths.get("results_dir", "./results")) / "morpheus_results.json"
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out_path}")

    elif args.mode == "infer":
        print("\n[Morpheus Interactive Mode]")
        print("Type 'quit' to exit. Press Enter to use empty utterances.\n")
        accumulated_dst: dict[str, str] = {}
        prev_agent = ""
        turn = 0

        while True:
            print(f"\n─── Turn {turn + 1} ───")
            print(f"Current DST: {state_to_string(accumulated_dst)}")
            agent_utt = input(f"Agent utterance (previous): ").strip()
            if agent_utt.lower() == "quit":
                break
            user_utt  = input(f"User utterance: ").strip()
            if user_utt.lower() == "quit":
                break

            expert, pred_tlb, pred_str = pipeline.predict_turn(
                prev_dst=accumulated_dst,
                agent_utt=agent_utt or prev_agent,
                user_utt=user_utt,
            )
            accumulated_dst.update(pred_tlb)
            prev_agent = agent_utt
            turn += 1

            print(f"\n  → Routed to  : {expert.upper()}")
            print(f"  → TLB update : {pred_str or '[none]'}")
            print(f"  → Full DST   : {state_to_string(accumulated_dst)}")


if __name__ == "__main__":
    main()
