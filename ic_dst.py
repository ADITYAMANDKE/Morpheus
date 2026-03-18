"""
ic_dst.py
─────────
IC-DST: Few-shot Dialogue State Tracking with LLMs via in-context learning.

Implements equation (1) from the OrchestraLLM paper:
    TLB_t = LLM(T, E_{1:K}, DST_{t-1}, A_{t-1}, U_t)

Where:
    T         = Schema table (all domains and slots)
    E_{1:K}   = K in-context exemplars (turn → TLB pairs)
    DST_{t-1} = Previous accumulated dialogue state
    A_{t-1}   = Previous agent utterance
    U_t       = Current user utterance

Key differences from Prompt-DST:
    - No fine-tuning: the LLM is used zero/few-shot
    - K exemplars are included in the prompt for in-context learning
    - Exemplars are retrieved from a small labelled set (similar to IC-DST paper)

Usage:
    python ic_dst.py eval --config config.yaml
    python ic_dst.py infer --config config.yaml \\
        --agent_utt "Do you have a price range?" \\
        --user_utt "Cheap please." \\
        --prev_dst "hotel-semi-type = hotel"
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import anthropic
import yaml

from data_preprocessing import (
    load_jsonl,
    string_to_state,
    state_to_string,
    format_input,
    SCHEMA_PROMPT,
)
from evaluate import compute_tlb_jga, compute_dst_jga


# ─── Exemplar Pool ────────────────────────────────────────────────────────────

class ExemplarPool:
    """
    Manages the pool of in-context exemplars for IC-DST.

    In the paper, exemplars are turn-level (agent_utt, user_utt) → TLB pairs
    sampled from the training set. For simplicity we use random sampling here;
    the router module handles semantic retrieval for OrchestraLLM.
    """

    def __init__(self, examples: list[dict], seed: int = 42):
        # Only keep turns with non-empty TLBs as exemplars (more informative)
        self.pool = [ex for ex in examples if ex.get("tlb")]
        self.rng = random.Random(seed)
        print(f"[ExemplarPool] {len(self.pool)} exemplars available (non-empty TLB turns)")

    def sample(self, k: int, exclude_id: Optional[str] = None) -> list[dict]:
        """
        Randomly samples K exemplars, optionally excluding a specific dialogue.

        Args:
            k          : Number of exemplars to sample.
            exclude_id : Dialogue ID to exclude (avoid leaking test dialogue).

        Returns:
            List of K exemplar dicts.
        """
        pool = self.pool
        if exclude_id:
            pool = [ex for ex in pool if ex["dialogue_id"] != exclude_id]
        k = min(k, len(pool))
        return self.rng.sample(pool, k)

    def format_exemplars(self, exemplars: list[dict]) -> str:
        """
        Formats K exemplars into the few-shot block included in the LLM prompt.

        Each exemplar shows:
            Previous state | Agent | User → TLB update

        Args:
            exemplars: List of exemplar dicts.

        Returns:
            Formatted string block for inclusion in the LLM prompt.
        """
        lines = ["--- Examples ---\n"]
        for i, ex in enumerate(exemplars, 1):
            prev_dst_str = state_to_string(ex.get("prev_dst", {}))
            lines.append(f"Example {i}:")
            lines.append(f"  Previous state: {prev_dst_str}")
            lines.append(f"  Agent: {ex.get('agent_utt', '').strip()}")
            lines.append(f"  User:  {ex.get('user_utt', '').strip()}")
            lines.append(f"  State update: {ex.get('target_text', '[none]')}")
            lines.append("")
        lines.append("--- End Examples ---\n")
        return "\n".join(lines)


# ─── Prompt Builder ───────────────────────────────────────────────────────────

def build_ic_prompt(
    prev_dst: dict[str, str],
    agent_utt: str,
    user_utt: str,
    exemplars_block: str,
    schema_prompt: str = SCHEMA_PROMPT,
) -> str:
    """
    Builds the full IC-DST prompt including schema, exemplars, and test instance.

    Structure (eq. 1): T + E_{1:K} + DST_{t-1} + A_{t-1} + U_t

    Args:
        prev_dst       : Previous accumulated dialogue state.
        agent_utt      : Previous agent utterance.
        user_utt       : Current user utterance.
        exemplars_block: Formatted few-shot exemplars string.
        schema_prompt  : Schema table T.

    Returns:
        Complete prompt string.
    """
    prev_dst_str = state_to_string(prev_dst)
    return (
        f"{schema_prompt}"
        f"{exemplars_block}"
        f"Now predict the state update for this turn:\n"
        f"  Previous state: {prev_dst_str}\n"
        f"  Agent: {agent_utt.strip()}\n"
        f"  User:  {user_utt.strip()}\n"
        f"  State update (new or changed slots only, format: slot = value | slot = value, "
        f"or [none] if no change):"
    )


# ─── LLM Client ───────────────────────────────────────────────────────────────

class ICDST:
    """
    IC-DST using Anthropic Claude as the LLM backbone.

    Supports:
        - Single-turn inference
        - Full dialogue inference with DST aggregation
        - Batch evaluation with rate-limit handling
    """

    def __init__(self, config: dict):
        self.config = config
        self.llm_cfg = config.get("ic_dst", {})
        self.model = self.llm_cfg.get("model", "claude-haiku-4-5-20251001")
        self.num_exemplars = self.llm_cfg.get("num_exemplars", 10)
        self.max_tokens = self.llm_cfg.get("max_tokens", 512)
        self.temperature = self.llm_cfg.get("temperature", 0.0)

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.exemplar_pool: Optional[ExemplarPool] = None

        print(f"[IC-DST] Model: {self.model} | K={self.num_exemplars} exemplars")

    def load_exemplar_pool(self, examples: list[dict]):
        """Initialises the exemplar pool from training/holdout examples."""
        self.exemplar_pool = ExemplarPool(examples)

    def _call_llm(self, prompt: str, retries: int = 3) -> str:
        """
        Calls the Anthropic API with retry logic for rate limits.

        Args:
            prompt : Full prompt string.
            retries: Number of retry attempts on rate limit errors.

        Returns:
            Raw LLM response text.
        """
        system_msg = (
            "You are a precise dialogue state tracking assistant. "
            "Extract only new or changed slot values from the conversation turn. "
            "Output ONLY the state update in the format 'slot = value | slot = value' "
            "or '[none]' if nothing changed. Do not explain or add extra text."
        )

        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip()
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                print(f"[IC-DST] Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            except anthropic.APIError as e:
                print(f"[IC-DST] API error: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(1)
        return "[none]"

    def predict_turn(
        self,
        prev_dst: dict[str, str],
        agent_utt: str,
        user_utt: str,
        dialogue_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Predicts the TLB for a single dialogue turn.

        Args:
            prev_dst    : Previous accumulated dialogue state.
            agent_utt   : Previous agent utterance.
            user_utt    : Current user utterance.
            dialogue_id : Source dialogue (excluded from exemplar sampling).

        Returns:
            Tuple of (raw_llm_output, parsed_tlb_string).
        """
        assert self.exemplar_pool, "Call load_exemplar_pool() first"
        exemplars = self.exemplar_pool.sample(self.num_exemplars, exclude_id=dialogue_id)
        exemplars_block = self.exemplar_pool.format_exemplars(exemplars)
        prompt = build_ic_prompt(prev_dst, agent_utt, user_utt, exemplars_block)
        raw = self._call_llm(prompt)
        return raw, raw  # raw IS the parsed string in this structured output setup

    def predict_dialogue(
        self,
        turns: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        Runs IC-DST over all turns of a single dialogue.

        Args:
            turns: Sorted list of turn dicts for one dialogue.

        Returns:
            - predicted_tlbs : List of predicted TLB dicts per turn.
            - final_dst      : Final accumulated dialogue state.
        """
        predicted_tlbs: list[dict] = []
        accumulated_dst: dict[str, str] = {}
        did = turns[0]["dialogue_id"] if turns else None

        for turn in turns:
            _, pred_str = self.predict_turn(
                prev_dst=accumulated_dst,
                agent_utt=turn["agent_utt"],
                user_utt=turn["user_utt"],
                dialogue_id=did,
            )
            pred_tlb = string_to_state(pred_str)
            predicted_tlbs.append(pred_tlb)
            accumulated_dst.update(pred_tlb)

        return predicted_tlbs, accumulated_dst

    def evaluate(
        self,
        examples: list[dict],
        max_dialogues: Optional[int] = None,
    ) -> dict:
        """
        Evaluates IC-DST on a set of turn examples.

        Args:
            examples      : List of turn examples.
            max_dialogues : Limit evaluation to N dialogues (for cost control).

        Returns:
            Dict with tlb_jga, dst_jga scores.
        """
        assert self.exemplar_pool, "Call load_exemplar_pool() first"

        # Group by dialogue
        dialogues: dict[str, list] = {}
        for ex in examples:
            dialogues.setdefault(ex["dialogue_id"], []).append(ex)
        for did in dialogues:
            dialogues[did].sort(key=lambda x: x["turn_idx"])

        if max_dialogues:
            dial_ids = list(dialogues.keys())[:max_dialogues]
            dialogues = {k: dialogues[k] for k in dial_ids}
            print(f"[IC-DST] Evaluating {max_dialogues} dialogues (cost limit)")

        all_preds, all_targets = [], []
        all_dst_preds, all_dst_targets = [], []

        for i, (did, turns) in enumerate(dialogues.items()):
            if i % 10 == 0:
                print(f"[IC-DST] Dialogue {i+1}/{len(dialogues)}: {did}")
            accumulated_dst: dict[str, str] = {}
            for turn in turns:
                _, pred_str = self.predict_turn(
                    prev_dst=accumulated_dst,
                    agent_utt=turn["agent_utt"],
                    user_utt=turn["user_utt"],
                    dialogue_id=did,
                )
                pred_tlb = string_to_state(pred_str)
                accumulated_dst.update(pred_tlb)

                all_preds.append(pred_str)
                all_targets.append(turn["target_text"])
                all_dst_preds.append(state_to_string(accumulated_dst))
                all_dst_targets.append(turn["dst_text"])

        tlb_jga = compute_tlb_jga(all_preds, all_targets)
        dst_jga  = compute_dst_jga(all_dst_preds, all_dst_targets)

        results = {
            "tlb_jga": round(tlb_jga * 100, 2),
            "dst_jga": round(dst_jga * 100, 2),
            "n_turns": len(all_preds),
            "n_dialogues": len(dialogues),
        }
        print(f"[IC-DST] TLB JGA: {results['tlb_jga']:.2f}%  |  DST JGA: {results['dst_jga']:.2f}%")
        return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="IC-DST with Claude LLM")
    parser.add_argument("mode", choices=["eval", "infer"])
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--max_dialogues", type=int, default=None,
                        help="Limit eval to N dialogues (cost control)")
    parser.add_argument("--agent_utt", default="")
    parser.add_argument("--user_utt",  default="")
    parser.add_argument("--prev_dst",  default="")
    args = parser.parse_args()

    config = load_config(args.config)
    paths  = config.get("paths", {})
    proc_dir = Path(paths.get("processed_dir", "./data/processed"))

    model = ICDST(config)

    # Load exemplars from training set
    train_ex = load_jsonl(str(proc_dir / "train.jsonl"))
    model.load_exemplar_pool(train_ex)

    if args.mode == "eval":
        val_ex  = load_jsonl(str(proc_dir / "val.jsonl"))
        results = model.evaluate(val_ex, max_dialogues=args.max_dialogues)
        print(json.dumps(results, indent=2))

    elif args.mode == "infer":
        prev_dst   = string_to_state(args.prev_dst) if args.prev_dst else {}
        raw, pred  = model.predict_turn(prev_dst, args.agent_utt, args.user_utt)
        print("\n[Raw LLM Output]\n", raw)
        print("\n[Parsed TLB]\n", string_to_state(pred))


if __name__ == "__main__":
    main()
