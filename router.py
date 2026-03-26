"""
router.py
─────────
OrchestraLLM Router: Retrieval-based dynamic routing between SLM and LLM.

Implements Section 3 of the paper:

    Step 1 — Expert Pool Construction (Section 3.1):
        Run both SLM and LLM on a small holdout set. Assign each turn to the
        expert that predicted it correctly (prefer SLM on ties).

    Step 2 — Triplet Representation Learning (Section 3.2):
        Fine-tune SenBERT with contrastive loss using task-aware and/or
        expert-aware supervision to better separate SLM vs LLM examples.

    Step 3 — Routing (Figure 1):
        For a new turn:
            1. Encode the (DST_{t-1}, A_{t-1}, U_t) triplet with SenBERT.
            2. Retrieve top-K nearest neighbours from SLM+LLM expert pools.
            3. Majority vote → assign to SLM or LLM expert.

Usage:
    # Build expert pools (requires both models to have predicted on holdout set)
    python router.py build_pools --config config.yaml

    # Fine-tune retriever
    python router.py train_retriever --config config.yaml

    # Route a single example
    python router.py route --config config.yaml \\
        --agent_utt "Any specific area?" \\
        --user_utt "No, just cheap please." \\
        --prev_dst "hotel-semi-type = hotel"
"""

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

from data_preprocessing import (
    load_jsonl,
    string_to_state,
    state_to_string,
)
from evaluate import compute_slot_f1


# ─── Triplet Encoder ──────────────────────────────────────────────────────────

def encode_triplet(
    prev_dst: dict[str, str],
    agent_utt: str,
    user_utt: str,
) -> str:
    """
    Encodes the routing triplet (DST_{t-1}, A_{t-1}, U_t) as a single string
    for embedding. This is the unit used to query expert pools.

    Args:
        prev_dst  : Previous accumulated dialogue state.
        agent_utt : Previous agent utterance.
        user_utt  : Current user utterance.

    Returns:
        Concatenated string representation.
    """
    dst_str = state_to_string(prev_dst)
    return f"State: {dst_str} | Agent: {agent_utt.strip()} | User: {user_utt.strip()}"


# ─── Expert Pool ──────────────────────────────────────────────────────────────

class ExpertPool:
    """
    Stores turn-level examples tagged with their correct expert (SLM or LLM).

    Construction logic (Section 3.1):
        - Both correct → assign to SLM pool (prefer cheaper model)
        - Only SLM correct → assign to SLM pool
        - Only LLM correct → assign to LLM pool
        - Neither correct → discard
    """

    def __init__(self):
        self.slm_pool: list[dict] = []
        self.llm_pool: list[dict] = []

    def add(
        self,
        example: dict,
        slm_correct: bool,
        llm_correct: bool,
    ):
        """
        Adds an example to the appropriate expert pool.

        Args:
            example     : Turn dict with triplet and ground truth.
            slm_correct : Whether Prompt-DST predicted TLB correctly.
            llm_correct : Whether IC-DST predicted TLB correctly.
        """
        entry = {
            "triplet_text": encode_triplet(
                example.get("prev_dst", {}),
                example.get("agent_utt", ""),
                example.get("user_utt", ""),
            ),
            "expert": "slm" if (slm_correct or not llm_correct) else "llm",
            "prev_dst": example.get("prev_dst", {}),
            "agent_utt": example.get("agent_utt", ""),
            "user_utt": example.get("user_utt", ""),
            "target_text": example.get("target_text", ""),
        }

        if slm_correct:           # Both correct or only SLM → SLM pool
            self.slm_pool.append(entry)
        elif llm_correct:         # Only LLM correct → LLM pool
            self.llm_pool.append(entry)
        # Neither correct: discard (not used in either pool)

    def sample_pools(self, pool_size: int, seed: int = 42) -> tuple[list, list]:
        """
        Samples up to pool_size entries from each pool.

        Args:
            pool_size: Max entries per pool (paper: 100 turns for MWOZ).
            seed     : Random seed.

        Returns:
            Tuple of (sampled_slm_pool, sampled_llm_pool).
        """
        rng = random.Random(seed)
        slm = rng.sample(self.slm_pool, min(pool_size, len(self.slm_pool)))
        llm = rng.sample(self.llm_pool, min(pool_size, len(self.llm_pool)))
        print(f"[ExpertPool] SLM pool: {len(slm)} | LLM pool: {len(llm)}")
        return slm, llm

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"slm": self.slm_pool, "llm": self.llm_pool}, f)
        print(f"[ExpertPool] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ExpertPool":
        with open(path, "rb") as f:
            data = pickle.load(f)
        pool = cls()
        pool.slm_pool = data["slm"]
        pool.llm_pool = data["llm"]
        print(f"[ExpertPool] Loaded — SLM: {len(pool.slm_pool)} | LLM: {len(pool.llm_pool)}")
        return pool


# ─── Contrastive Dataset ──────────────────────────────────────────────────────

def compute_instance_similarity(ex_a: dict, ex_b: dict) -> float:
    """
    Computes task-aware similarity between two instances.

    From Section 3.2:
        Sim = 0.5 * SimDST + 0.5 * SimTLB
        SimTLB = F1_slot_value + F1_slot - 1

    Args:
        ex_a, ex_b: Turn example dicts with 'prev_dst' and 'tlb' fields.

    Returns:
        Scalar similarity in [-1, 1].
    """
    tlb_a = ex_a.get("tlb", {})
    tlb_b = ex_b.get("tlb", {})
    dst_a = ex_a.get("prev_dst", {})
    dst_b = ex_b.get("prev_dst", {})

    sim_tlb = compute_slot_f1(tlb_a, tlb_b)
    sim_dst = compute_slot_f1(dst_a, dst_b)
    return 0.5 * sim_dst + 0.5 * sim_tlb


class ContrastiveDataset(Dataset):
    """
    Contrastive pairs dataset for fine-tuning the SenBERT retriever.

    Supports three supervision modes (Section 3.2):
        - task    : Pairs based on slot-value similarity of ground-truth states.
        - expert  : Pairs based on which expert was correct.
        - task+expert: Union of both.

    Each item yields (anchor, positive, negative) triplet strings.
    """

    def __init__(
        self,
        examples: list[dict],
        expert_labels: Optional[dict[str, str]],  # dialogue_id+turn → "slm"|"llm"
        supervision: str = "task+expert",
        num_pairs: int = 25,
        seed: int = 42,
    ):
        self.pairs: list[tuple[str, str, int]] = []  # (text_a, text_b, label)
        rng = random.Random(seed)

        # Build triplet texts
        for ex in examples:
            ex["_triplet"] = encode_triplet(
                ex.get("prev_dst", {}),
                ex.get("agent_utt", ""),
                ex.get("user_utt", ""),
            )

        if supervision in ("task", "task+expert"):
            self._add_task_pairs(examples, num_pairs, rng)

        if supervision in ("expert", "task+expert") and expert_labels:
            self._add_expert_pairs(examples, expert_labels, num_pairs, rng)

        print(f"[ContrastiveDataset] {len(self.pairs)} pairs (supervision={supervision})")

    def _add_task_pairs(self, examples, num_pairs, rng):
        """Adds task-aware positive/negative pairs by slot-value similarity."""
        scored: list[tuple[float, int, int]] = []
        n = len(examples)
        # Sample random pairs to avoid O(n²) computation
        sampled_indices = [(rng.randint(0, n-1), rng.randint(0, n-1))
                           for _ in range(min(num_pairs * 20, n * n))]
        for i, j in sampled_indices:
            if i != j:
                sim = compute_instance_similarity(examples[i], examples[j])
                scored.append((sim, i, j))

        scored.sort(key=lambda x: -x[0])
        # Top-l as positives (label=1), bottom-l as negatives (label=0)
        for _, i, j in scored[:num_pairs]:
            self.pairs.append((examples[i]["_triplet"], examples[j]["_triplet"], 1))
        for _, i, j in scored[-num_pairs:]:
            self.pairs.append((examples[i]["_triplet"], examples[j]["_triplet"], 0))

    def _add_expert_pairs(self, examples, expert_labels, num_pairs, rng):
        """
        Adds expert-aware positive/negative pairs (Section 3.2).

        Paper: "Expert-Aware Supervision first groups instances by expert label.
        We compute pairwise triplet similarities using an off-the-shelf embedder
        (e.g., SenBERT). The l highest scoring pairs with the same expert label
        are positive examples, and the l lowest scoring pairs with different
        expert labels are negative examples."
        """
        # Group examples by expert label
        by_expert: dict[str, list] = {"slm": [], "llm": []}
        for ex in examples:
            key = f"{ex['dialogue_id']}_{ex['turn_idx']}"
            label = expert_labels.get(key, "slm")
            by_expert[label].append(ex)

        slm_exs = by_expert["slm"]
        llm_exs = by_expert["llm"]

        # Compute SenBERT embeddings for all examples
        all_exs = slm_exs + llm_exs
        if len(all_exs) < 2:
            return

        embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        triplet_texts = [ex["_triplet"] for ex in all_exs]
        embeddings = embedder.encode(triplet_texts, convert_to_tensor=True,
                                     normalize_embeddings=True)

        n_slm = len(slm_exs)

        # ── Positive pairs: highest-similarity same-expert pairs ──
        # Sample candidate pairs within each expert group, score by embedding sim
        same_expert_scored: list[tuple[float, int, int]] = []
        for pool_start, pool_end in [(0, n_slm), (n_slm, len(all_exs))]:
            pool_size = pool_end - pool_start
            if pool_size < 2:
                continue
            # Sample candidate pairs to avoid O(n²)
            n_candidates = min(num_pairs * 20, pool_size * (pool_size - 1))
            for _ in range(n_candidates):
                i = rng.randint(pool_start, pool_end - 1)
                j = rng.randint(pool_start, pool_end - 1)
                if i != j:
                    sim = (embeddings[i] @ embeddings[j]).item()
                    same_expert_scored.append((sim, i, j))

        same_expert_scored.sort(key=lambda x: -x[0])
        for _, i, j in same_expert_scored[:num_pairs]:
            self.pairs.append((all_exs[i]["_triplet"], all_exs[j]["_triplet"], 1))

        # ── Negative pairs: lowest-similarity cross-expert pairs ──
        cross_expert_scored: list[tuple[float, int, int]] = []
        if slm_exs and llm_exs:
            n_candidates = min(num_pairs * 20, n_slm * len(llm_exs))
            for _ in range(n_candidates):
                i = rng.randint(0, n_slm - 1)             # SLM index
                j = rng.randint(n_slm, len(all_exs) - 1)  # LLM index
                sim = (embeddings[i] @ embeddings[j]).item()
                cross_expert_scored.append((sim, i, j))

            cross_expert_scored.sort(key=lambda x: x[0])  # ascending: lowest first
            for _, i, j in cross_expert_scored[:num_pairs]:
                self.pairs.append((all_exs[i]["_triplet"], all_exs[j]["_triplet"], 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text_a, text_b, label = self.pairs[idx]
        return {"text_a": text_a, "text_b": text_b, "label": label}


# ─── Retriever (SenBERT Bi-Encoder) ──────────────────────────────────────────

class Retriever:
    """
    SenBERT bi-encoder retriever for OrchestraLLM routing.

    Encodes dialogue triplets into embeddings, then uses cosine similarity
    to find nearest neighbours in the expert pools.
    """

    def __init__(self, backbone: str = "sentence-transformers/all-mpnet-base-v2"):
        self.backbone = backbone
        self.model = SentenceTransformer(backbone)
        print(f"[Retriever] Loaded: {backbone}")

        # Expert pool embeddings (populated by index_pools)
        self.slm_embeddings: Optional[torch.Tensor] = None
        self.llm_embeddings: Optional[torch.Tensor] = None
        self.slm_pool: list[dict] = []
        self.llm_pool: list[dict] = []

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encodes a list of strings to unit-normalised embeddings."""
        embs = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return embs  # shape: (N, D)

    def index_pools(self, slm_pool: list[dict], llm_pool: list[dict]):
        """
        Pre-computes and caches embeddings for all entries in both expert pools.

        Args:
            slm_pool: List of SLM expert pool entries.
            llm_pool: List of LLM expert pool entries.
        """
        self.slm_pool = slm_pool
        self.llm_pool = llm_pool

        slm_texts = [e["triplet_text"] for e in slm_pool]
        llm_texts = [e["triplet_text"] for e in llm_pool]

        print(f"[Retriever] Indexing {len(slm_texts)} SLM + {len(llm_texts)} LLM pool entries...")
        self.slm_embeddings = self.encode(slm_texts) if slm_texts else torch.zeros(0, 768)
        self.llm_embeddings = self.encode(llm_texts) if llm_texts else torch.zeros(0, 768)
        print("[Retriever] Indexing complete.")

    def route(
        self,
        prev_dst: dict[str, str],
        agent_utt: str,
        user_utt: str,
        top_k: int = 10,
        tie_break: str = "slm",
    ) -> tuple[str, dict]:
        """
        Routes a new input instance to SLM or LLM via majority vote.

        Step-by-step (Figure 1):
            1. Encode the query triplet.
            2. Compute cosine similarity against all pool entries.
            3. Take top-K nearest across both pools.
            4. Count votes: how many of the K came from SLM vs LLM pool.
            5. Return majority; break ties by preferring `tie_break`.

        Args:
            prev_dst  : Previous dialogue state.
            agent_utt : Previous agent utterance.
            user_utt  : Current user utterance.
            top_k     : Number of nearest neighbours to retrieve.
            tie_break : "slm" or "llm" — which expert to prefer on tie.

        Returns:
            Tuple of (assigned_expert, routing_details_dict).
        """
        assert self.slm_embeddings is not None, "Call index_pools() first"

        query_text = encode_triplet(prev_dst, agent_utt, user_utt)
        query_emb  = self.encode([query_text])  # (1, D)

        # Cosine similarities (embeddings already normalised → dot product = cosine)
        slm_scores = (query_emb @ self.slm_embeddings.T).squeeze(0)  # (N_slm,)
        llm_scores = (query_emb @ self.llm_embeddings.T).squeeze(0)  # (N_llm,)

        # Combine with expert label
        all_scores  = torch.cat([slm_scores, llm_scores])
        all_experts = ["slm"] * len(self.slm_pool) + ["llm"] * len(self.llm_pool)

        # Top-K
        k = min(top_k, len(all_scores))
        top_indices = torch.topk(all_scores, k).indices.tolist()
        top_experts = [all_experts[i] for i in top_indices]
        top_scores  = [all_scores[i].item() for i in top_indices]

        # Majority vote
        slm_votes = top_experts.count("slm")
        llm_votes = top_experts.count("llm")

        if slm_votes > llm_votes:
            assigned = "slm"
        elif llm_votes > slm_votes:
            assigned = "llm"
        else:
            assigned = tie_break  # Paper: prefer SLM on tie

        details = {
            "assigned": assigned,
            "slm_votes": slm_votes,
            "llm_votes": llm_votes,
            "top_k": k,
            "top_scores": top_scores[:5],
            "top_experts": top_experts[:5],
        }
        return assigned, details

    def fine_tune(
        self,
        dataset: ContrastiveDataset,
        config: dict,
        save_path: Optional[str] = None,
    ):
        """
        Fine-tunes SenBERT using contrastive loss (InfoNCE-style).

        For each (anchor, positive) pair, the positive embedding should be
        close and the in-batch negatives far in embedding space.

        Args:
            dataset  : ContrastiveDataset with (text_a, text_b, label) pairs.
            config   : Router config dict from config.yaml.
            save_path: Where to save the fine-tuned model.
        """
        from sentence_transformers import losses, InputExample
        from sentence_transformers import SentenceTransformer
        from torch.utils.data import DataLoader as STDataLoader

        # Convert to SentenceTransformers InputExample format
        train_examples = []
        for item in dataset:
            train_examples.append(
                InputExample(texts=[item["text_a"], item["text_b"]], label=float(item["label"]))
            )

        rtr_cfg = config.get("router", {})
        loader = STDataLoader(train_examples, shuffle=True,
                              batch_size=rtr_cfg.get("contrastive_batch_size", 16))

        # CosineSimilarityLoss: pulls positives together, pushes negatives apart
        loss_fn = losses.CosineSimilarityLoss(self.model)

        print(f"[Retriever] Fine-tuning for {rtr_cfg.get('contrastive_epochs', 5)} epochs...")
        self.model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=rtr_cfg.get("contrastive_epochs", 5),
            warmup_steps=int(len(loader) * 0.1),
            optimizer_params={"lr": rtr_cfg.get("contrastive_lr", 2e-5)},
            show_progress_bar=True,
        )

        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            self.model.save(save_path)
            print(f"[Retriever] Saved fine-tuned model → {save_path}")

    def save(self, path: str):
        """Saves retriever state (model + pool embeddings)."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save(str(Path(path)))
        torch.save({
            "slm_embeddings": self.slm_embeddings,
            "llm_embeddings": self.llm_embeddings,
            "slm_pool": self.slm_pool,
            "llm_pool": self.llm_pool,
        }, str(Path(path) / "pool_index.pt"))
        print(f"[Retriever] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "Retriever":
        """Loads a saved retriever."""
        r = cls(backbone=str(Path(path)))
        data = torch.load(str(Path(path) / "pool_index.pt"), map_location="cpu", weights_only=False)
        device = r.model.device
        r.slm_embeddings = data["slm_embeddings"].to(device)
        r.llm_embeddings = data["llm_embeddings"].to(device)
        r.slm_pool = data["slm_pool"]
        r.llm_pool = data["llm_pool"]
        print(f"[Retriever] Loaded from {path}")
        return r


# ─── Pool Builder ─────────────────────────────────────────────────────────────

def build_expert_pools(
    holdout_examples: list[dict],
    slm_predictions: dict[str, str],   # "dial_id_turn_idx" → predicted TLB string
    llm_predictions: dict[str, str],
    pool_size: int = 100,
    seed: int = 42,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """
    Constructs SLM and LLM expert pools from holdout set predictions.

    Args:
        holdout_examples : All holdout turn examples.
        slm_predictions  : Dict mapping "dial_id_turn_idx" → SLM predicted TLB string.
        llm_predictions  : Dict mapping "dial_id_turn_idx" → LLM predicted TLB string.
        pool_size        : Max entries per expert pool.
        seed             : Random seed.

    Returns:
        Tuple of (slm_pool, llm_pool, expert_labels_dict).
    """
    pool = ExpertPool()
    expert_labels: dict[str, str] = {}

    for ex in holdout_examples:
        key = f"{ex['dialogue_id']}_{ex['turn_idx']}"
        target = string_to_state(ex["target_text"])

        slm_pred = string_to_state(slm_predictions.get(key, ""))
        llm_pred = string_to_state(llm_predictions.get(key, ""))

        slm_correct = (slm_pred == target)
        llm_correct = (llm_pred == target)

        pool.add(ex, slm_correct, llm_correct)

        # Expert label for retriever fine-tuning
        if slm_correct:
            expert_labels[key] = "slm"
        elif llm_correct:
            expert_labels[key] = "llm"

    slm_pool, llm_pool = pool.sample_pools(pool_size, seed=seed)
    return slm_pool, llm_pool, expert_labels


# ─── CLI ──────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="OrchestraLLM Router")
    parser.add_argument("mode", choices=["build_pools", "train_retriever", "route"])
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--agent_utt", default="")
    parser.add_argument("--user_utt",  default="")
    parser.add_argument("--prev_dst",  default="")
    args = parser.parse_args()

    config = load_config(args.config)
    paths  = config.get("paths", {})
    rtr_cfg = config.get("router", {})
    proc_dir = Path(paths.get("processed_dir", "./data/processed"))
    pool_dir = Path(paths.get("expert_pool_dir", "./data/expert_pools"))

    if args.mode == "build_pools":
        print("Loading holdout predictions from disk...")
        slm_preds_path = pool_dir / "slm_holdout_preds.json"
        llm_preds_path = pool_dir / "llm_holdout_preds.json"
        if not slm_preds_path.exists() or not llm_preds_path.exists():
            print(
                "ERROR: Run prompt_dst.py and ic_dst.py on the holdout set first,\n"
                f"then save predictions as JSON dicts to:\n"
                f"  {slm_preds_path}\n  {llm_preds_path}"
            )
            return
        with open(slm_preds_path) as f:
            slm_preds = json.load(f)
        with open(llm_preds_path) as f:
            llm_preds = json.load(f)

        holdout_ex = load_jsonl(str(proc_dir / "holdout.jsonl"))
        slm_pool, llm_pool, expert_labels = build_expert_pools(
            holdout_ex, slm_preds, llm_preds,
            pool_size=rtr_cfg.get("slm_pool_size", 100),
        )
        pool_dir.mkdir(parents=True, exist_ok=True)
        with open(pool_dir / "slm_pool.json", "w") as f:
            json.dump(slm_pool, f, indent=2)
        with open(pool_dir / "llm_pool.json", "w") as f:
            json.dump(llm_pool, f, indent=2)
        with open(pool_dir / "expert_labels.json", "w") as f:
            json.dump(expert_labels, f, indent=2)
        print("Expert pools saved.")

    elif args.mode == "train_retriever":
        holdout_ex = load_jsonl(str(proc_dir / "holdout.jsonl"))
        with open(pool_dir / "expert_labels.json") as f:
            expert_labels = json.load(f)

        dataset = ContrastiveDataset(
            holdout_ex,
            expert_labels,
            supervision=rtr_cfg.get("supervision", "task+expert"),
            num_pairs=rtr_cfg.get("contrastive_pairs", 25),
        )

        retriever = Retriever(backbone=rtr_cfg.get("retriever_backbone",
                                                    "sentence-transformers/all-mpnet-base-v2"))
        retriever.fine_tune(dataset, config, save_path=str(pool_dir / "retriever"))

    elif args.mode == "route":
        retriever_path = str(pool_dir / "retriever")
        if Path(retriever_path).exists():
            retriever = Retriever.load(retriever_path)
        else:
            retriever = Retriever(backbone=rtr_cfg.get("retriever_backbone",
                                                        "sentence-transformers/all-mpnet-base-v2"))

        with open(pool_dir / "slm_pool.json") as f:
            slm_pool = json.load(f)
        with open(pool_dir / "llm_pool.json") as f:
            llm_pool = json.load(f)
        retriever.index_pools(slm_pool, llm_pool)

        prev_dst = string_to_state(args.prev_dst) if args.prev_dst else {}
        expert, details = retriever.route(
            prev_dst, args.agent_utt, args.user_utt,
            top_k=rtr_cfg.get("top_k", 10),
            tie_break=rtr_cfg.get("tie_break", "slm"),
        )
        print(f"\n[Router] Assigned to: {expert.upper()}")
        print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
