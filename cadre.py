"""
cadre.py
───────────
CADRE: Context- and Domain-Aware Routing Engine for Dialogue State Tracking.
Full pipeline orchestrator combining Prompt-DST (SLM), IC-DST (LLM), the
retrieval router and the domain-aware reliability overlay.

At inference time (Figure 1 of the paper):
    1. Encode the input triplet (DST_{t-1}, A_{t-1}, U_t) via SenBERT.
    2. Base routing decision:
         knn — top-K nearest neighbours over the expert pools, majority vote
               (ties → SLM)                                   [Section 2.3]
         mrl — learned classifier head, P(LLM) >= 0.5        [Section 4.1]
    3. Domain-switch override: if the user activates a domain d with
       reliability r_d < τ, force the LLM regardless of the vote [Section 2.4]
    4. Run the selected expert and return TLB_t.
    5. Aggregate TLB_t into the running DST.

Configurations reported in the paper (Tables 1 and 3):
    SLM only            python cadre.py eval --force_expert slm
    LLM only            python cadre.py eval --force_expert llm
    Base KNN router     python cadre.py eval --no_domain_aware
    CADRE (full)        python cadre.py eval
    + Semantic exemplars python cadre.py eval --no_domain_aware --exemplar_selection semantic
    MRL classifier head python cadre.py eval --no_domain_aware --router mrl

    Add --max_turns 204 (headline slice) or --max_turns 736 (wide slice).

Usage:
    python cadre.py eval  --config config.yaml [flags above]
    python cadre.py infer --config config.yaml
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
from mrl_router import MRLRouter
from domain_router import (
    compute_domain_reliability,
    detect_domain_switch,
    should_override_to_llm,
)


# ─── CADRE ─────────────────────────────────────────────────────────────────

class CADRE:
    """
    Full CADRE pipeline: router + SLM expert + LLM expert.

    The router dynamically dispatches each turn to the most appropriate expert
    based on semantic similarity to the expert pools.
    """

    def __init__(self, config: dict):
        self.config = config
        self.rtr_cfg = config.get("router", {})
        self.paths   = config.get("paths", {})

        # Components (loaded lazily)
        self.router_type = self.rtr_cfg.get("router_type", "knn")   # "knn" | "mrl"
        self.retriever: Optional[Retriever] = None
        self.mrl: Optional[MRLRouter] = None
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

        # ── Base router ──
        if self.router_type == "mrl":
            # Section 4.1 — learned classifier head (negative result)
            mrl_path = str(pool_dir / "mrl_router")
            print(f"[CADRE] Router: MRL classifier head from {mrl_path}")
            self.mrl = MRLRouter.load(mrl_path)
            self.mrl.threshold = self.config.get("mrl_router", {}).get("decision_threshold", 0.5)
        else:
            # Section 2.3 — KNN majority vote over expert pools
            r_path = retriever_path or str(pool_dir / "retriever")
            use_ft = self.rtr_cfg.get("use_fine_tuned_retriever", False)
            if use_ft and Path(r_path).exists():
                # Optional, not part of the paper (see config.yaml)
                print("[CADRE] Router: KNN with contrastively fine-tuned retriever (non-paper setting)")
                self.retriever = Retriever.load(r_path)
            else:
                backbone = self.rtr_cfg.get("retriever_backbone",
                                            "sentence-transformers/all-mpnet-base-v2")
                print(f"[CADRE] Router: KNN with off-the-shelf {backbone}")
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
                    print("[CADRE] WARNING: Expert pools not found. Run router.py build_pools first.")

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
                print(f"[CADRE] Domain reliability: {self.domain_reliability}")
            else:
                print("[CADRE] WARNING: Cannot compute domain reliability — pools not found.")

        # ── SLM Expert (Prompt-DST) ──
        ckpt = slm_checkpoint or str(model_dir / "best")
        print(f"[CADRE] Loading SLM expert from: {ckpt}")
        self.slm = PromptDST(self.config)
        self.slm.load(ckpt if Path(ckpt).exists() else None)

        # ── LLM Expert (IC-DST) ──
        print("[CADRE] Initialising LLM expert (IC-DST)...")
        self.llm = ICDST(self.config)
        if exemplar_examples:
            self.llm.load_exemplar_pool(exemplar_examples)

        print("[CADRE] All components loaded.")
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
        assert (self.retriever or self.mrl) and self.slm and self.llm, "Call .load() first"

        domain_overridden = False

        # Forced expert mode (bypass router entirely) — SLM-only / LLM-only baselines
        if self.force_expert:
            expert = self.force_expert
            routing_details = {"assigned": expert, "reason": "forced"}

        # Domain-switch override (Section 2.4): fires only on newly activated
        # domains whose offline reliability r_d < τ; otherwise fall through to the
        # base router.
        elif self.domain_aware and self.domain_reliability and should_override_to_llm(
            new_domains := detect_domain_switch(prev_dst, user_utt, self.domain_keywords),
            self.domain_reliability,
            self.reliability_threshold,
        ):
            expert = "llm"
            domain_overridden = True
            routing_details = {
                "assigned": "llm",
                "reason": "domain_switch_override",
                "new_domains": sorted(new_domains),
                "reliability_scores": {
                    d: round(self.domain_reliability.get(d, 0.5), 3) for d in new_domains
                },
            }
            self.domain_override_count += 1

        # Base routing decision
        else:
            expert, routing_details = self._base_route(prev_dst, agent_utt, user_utt)

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

    def _base_route(self, prev_dst: dict[str, str], agent_utt: str, user_utt: str) -> tuple[str, dict]:
        """KNN majority vote (Section 2.3) or MRL classifier head (Section 4.1)."""
        if self.router_type == "mrl":
            return self.mrl.route(prev_dst, agent_utt, user_utt)
        return self.retriever.route(
            prev_dst, agent_utt, user_utt,
            top_k=self.rtr_cfg.get("top_k", 10),
            tie_break=self.rtr_cfg.get("tie_break", "slm"),
        )

    def predict_dialogue(
        self,
        turns: list[dict],
    ) -> tuple[list[dict], dict, list[str]]:
        """
        Runs CADRE over all turns of a dialogue.

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
        max_turns: Optional[int] = None,
    ) -> dict:
        """
        Full evaluation of CADRE on a set of turn examples.

        Reports TLB JGA, DST JGA, average slot F1, per-domain TLB JGA, routing
        share (% SLM / % LLM), domain-override rate and wall-clock time
        (paper Section 3.1, "Metrics").

        Args:
            examples      : Turn-level examples from load_jsonl().
            max_dialogues : Limit evaluation to the first N dialogues.
            max_turns     : Limit evaluation to the first N turns, taking whole
                            dialogues in file order (paper slices: 204 and 736).

        Returns:
            Full results dict.
        """
        assert (self.retriever or self.mrl) and self.slm and self.llm, "Call .load() first"

        # Group by dialogue
        dialogues: dict[str, list] = {}
        for ex in examples:
            dialogues.setdefault(ex["dialogue_id"], []).append(ex)
        for did in dialogues:
            dialogues[did].sort(key=lambda x: x["turn_idx"])

        if max_dialogues:
            dial_ids = list(dialogues.keys())[:max_dialogues]
            dialogues = {k: dialogues[k] for k in dial_ids}

        if max_turns:
            # Take whole dialogues in order until the slice holds max_turns turns.
            picked, n = {}, 0
            for did, turns in dialogues.items():
                if n >= max_turns:
                    break
                picked[did] = turns
                n += len(turns)
            dialogues = picked
            print(f"[CADRE] Evaluation slice: {n} turns / {len(dialogues)} dialogues "
                  f"(requested {max_turns})")

        self.routing_log = []
        all_pred_tlbs, all_gold_tlbs = [], []
        all_pred_dsts, all_gold_dsts = [], []

        start = time.time()
        for i, (did, turns) in enumerate(dialogues.items()):
            if i % 5 == 0:
                print(f"[CADRE] Dialogue {i+1}/{len(dialogues)}: {did}")

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
            "configuration": self.configuration_name(),
            "router_type": self.router_type,
            "domain_aware": bool(self.domain_aware and not self.force_expert),
            "reliability_threshold": self.reliability_threshold,
            "exemplar_selection": self.llm.exemplar_selection,
            "force_expert": self.force_expert,
            "slm_assignment_ratio": round(n_slm / n_turns * 100, 1) if n_turns else 0,
            "llm_assignment_ratio": round(n_llm / n_turns * 100, 1) if n_turns else 0,
            "n_slm_calls": n_slm,
            "n_llm_calls": n_llm,
            "n_domain_overrides": n_domain_overrides,
            "domain_override_ratio": round(n_domain_overrides / n_turns * 100, 1) if n_turns else 0,
            "domain_reliability": {k: round(v, 3) for k, v in self.domain_reliability.items()},
            "eval_time_seconds": round(elapsed, 1),
        })

        print(f"\n[Routing] SLM: {results['slm_assignment_ratio']}% | "
              f"LLM: {results['llm_assignment_ratio']}%")
        if n_domain_overrides:
            print(f"[Domain Override] {n_domain_overrides} turns ({results['domain_override_ratio']}%)")
        print(f"[Timing] {elapsed:.1f}s for {n_turns} turns")

        return results

    def configuration_name(self) -> str:
        """Short name for the configuration, matching the rows of Tables 1 and 3."""
        if self.force_expert == "slm":
            return "slm_only"
        if self.force_expert == "llm":
            return "llm_only"
        if self.router_type == "mrl":
            return "mrl_classifier"
        parts = ["knn_router"]
        if self.domain_aware:
            parts.append("domain_aware")
        if self.llm and self.llm.exemplar_selection == "semantic":
            parts.append("semantic_exemplars")
        return "+".join(parts) if len(parts) > 1 else "base_knn_router"

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
    parser = argparse.ArgumentParser(description="CADRE Full Pipeline")
    parser.add_argument("mode", choices=["eval", "infer"])
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--max_dialogues", type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=None,
                        help="Evaluate on the first N turns (whole dialogues). Paper: 204 / 736")
    parser.add_argument("--force_expert", choices=["slm", "llm"], default=None,
                        help="Force all turns to one expert (SLM-only / LLM-only baselines)")
    parser.add_argument("--router", choices=["knn", "mrl"], default=None,
                        help="Base router: knn (Section 2.3) or mrl classifier head (Section 4.1)")
    parser.add_argument("--no_domain_aware", action="store_true",
                        help="Disable the domain-switch override (base router)")
    parser.add_argument("--reliability_threshold", type=float, default=None,
                        help="Override τ from config")
    parser.add_argument("--exemplar_selection", choices=["random", "semantic"], default=None,
                        help="IC-DST exemplar selection (Section 2.2 random / Section 4.2 semantic)")
    parser.add_argument("--results_out", default=None,
                        help="Path to save results JSON (default: results/<configuration>_<n>turns.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    rtr = config.setdefault("router", {})
    if args.router:
        rtr["router_type"] = args.router
    if args.no_domain_aware:
        rtr["domain_aware"] = False
    if args.reliability_threshold is not None:
        rtr["reliability_threshold"] = args.reliability_threshold
    if args.exemplar_selection:
        config.setdefault("ic_dst", {})["exemplar_selection"] = args.exemplar_selection

    paths  = config.get("paths", {})
    proc_dir = Path(paths.get("processed_dir", "./data/processed"))

    # Load training examples for IC-DST exemplar pool
    train_ex = load_jsonl(str(proc_dir / "train.jsonl"))

    pipeline = CADRE(config)
    if args.force_expert:
        pipeline.force_expert = args.force_expert
        print(f"[CADRE] FORCED MODE: all turns → {args.force_expert.upper()}")
    pipeline.load(exemplar_examples=train_ex)
    print(f"[CADRE] Configuration: {pipeline.configuration_name()}")

    if args.mode == "eval":
        val_ex  = load_jsonl(str(proc_dir / "val.jsonl"))
        results = pipeline.evaluate(val_ex, max_dialogues=args.max_dialogues,
                                    max_turns=args.max_turns)

        out_path = args.results_out or str(
            Path(paths.get("results_dir", "./results"))
            / f"{pipeline.configuration_name()}_{results['n_turns']}turns.json"
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out_path}")

    elif args.mode == "infer":
        print("\n[CADRE Interactive Mode]")
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
