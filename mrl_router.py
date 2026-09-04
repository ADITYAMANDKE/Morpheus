"""
mrl_router.py
─────────────
Margin-ranked learned router (CADRE paper, Section 4.1 — negative result).

Instead of the non-parametric KNN majority vote, a small MLP classifier head
is trained on top of the SenBERT bi-encoder to predict P(LLM) directly.

Training data (paper Section 4.1, "Setup"):
    The *discriminative* subset of the holdout set: every turn for which exactly
    one of the two experts was correct.
        label = 1.0  if only the LLM was correct
        label = 0.0  if only the SLM was correct
    Ties (both correct / neither correct) are dropped.
    A pos_weight term in BCE-with-logits loss handles the class imbalance.

Training schedule (paper Appendix A, Table 4):
    lr 2e-5, batch 16, 5 epochs = 2 head-only warm-up (encoder frozen)
    + 3 end-to-end (encoder + head), classifier hidden dim 128.

Inference:
    route to LLM iff sigmoid(logit) >= 0.5.

Usage:
    python mrl_router.py train --config config.yaml
    python mrl_router.py route --config config.yaml \\
        --agent_utt "Any specific area?" \\
        --user_utt "No, just cheap please." \\
        --prev_dst "hotel-semi-type = hotel"
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

from data_preprocessing import load_jsonl, string_to_state
from router import encode_triplet


# ─── Data ─────────────────────────────────────────────────────────────────────

def build_discriminative_set(
    holdout_examples: list[dict],
    slm_predictions: dict[str, str],
    llm_predictions: dict[str, str],
) -> list[tuple[str, float]]:
    """
    Builds the discriminative training set: turns where exactly one expert
    was correct. Returns (triplet_text, label) with label 1.0 = LLM-only correct.
    """
    data: list[tuple[str, float]] = []
    for ex in holdout_examples:
        key = f"{ex['dialogue_id']}_{ex['turn_idx']}"
        target = string_to_state(ex["target_text"])
        slm_ok = string_to_state(slm_predictions.get(key, "")) == target
        llm_ok = string_to_state(llm_predictions.get(key, "")) == target
        if slm_ok == llm_ok:
            continue  # tie → not discriminative
        text = encode_triplet(ex.get("prev_dst", {}), ex.get("agent_utt", ""), ex.get("user_utt", ""))
        data.append((text, 1.0 if llm_ok else 0.0))

    n_pos = sum(1 for _, y in data if y == 1.0)
    print(f"[MRL] Discriminative set: {len(data)} turns "
          f"(LLM-only correct: {n_pos}, SLM-only correct: {len(data) - n_pos})")
    return data


class _TextLabelDataset(Dataset):
    def __init__(self, items: list[tuple[str, float]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _collate(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.float32)


# ─── Model ────────────────────────────────────────────────────────────────────

class MRLRouter(nn.Module):
    """
    SenBERT bi-encoder + MLP classifier head predicting P(LLM).
    """

    def __init__(
        self,
        backbone: str = "sentence-transformers/all-mpnet-base-v2",
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.encoder = SentenceTransformer(backbone)
        emb_dim = self.encoder.get_sentence_embedding_dimension()
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.threshold = 0.5

    @property
    def device(self):
        return next(self.head.parameters()).device

    def _embed(self, texts: list[str], grad: bool) -> torch.Tensor:
        """Encodes texts with the SenBERT encoder, optionally with gradients."""
        if not grad:
            with torch.no_grad():
                return self.encoder.encode(
                    texts, convert_to_tensor=True, normalize_embeddings=True,
                    device=str(self.device), show_progress_bar=False,
                )
        features = self.encoder.tokenize(texts)
        features = {k: v.to(self.device) for k, v in features.items()}
        out = self.encoder(features)["sentence_embedding"]
        return nn.functional.normalize(out, p=2, dim=-1)

    def forward(self, texts: list[str], train_encoder: bool = False) -> torch.Tensor:
        emb = self._embed(texts, grad=train_encoder)
        return self.head(emb).squeeze(-1)  # logits, shape (B,)

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        data: list[tuple[str, float]],
        lr: float = 2e-5,
        batch_size: int = 16,
        warmup_epochs: int = 2,
        e2e_epochs: int = 3,
        seed: int = 42,
    ):
        """
        Two-phase training (paper Section 4.1):
            Phase 1: head-only warm-up, encoder frozen.
            Phase 2: end-to-end fine-tuning of encoder + head.
        BCE-with-logits with pos_weight = n_neg / n_pos.
        """
        torch.manual_seed(seed)
        random.seed(seed)

        n_pos = sum(1 for _, y in data if y == 1.0)
        n_neg = len(data) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[MRL] pos_weight = {pos_weight.item():.3f}")

        loader = DataLoader(_TextLabelDataset(data), batch_size=batch_size,
                            shuffle=True, collate_fn=_collate)

        def run_phase(name: str, epochs: int, params, train_encoder: bool):
            if epochs <= 0:
                return
            opt = torch.optim.AdamW(params, lr=lr)
            self.train()
            for ep in range(epochs):
                total, n = 0.0, 0
                for texts, labels in loader:
                    labels = labels.to(self.device)
                    logits = self(texts, train_encoder=train_encoder)
                    loss = loss_fn(logits, labels)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += loss.item() * len(texts)
                    n += len(texts)
                print(f"[MRL] {name} epoch {ep + 1}/{epochs} — loss {total / max(n, 1):.4f}")

        # Phase 1: head only
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        run_phase("warm-up (head only)", warmup_epochs, self.head.parameters(), train_encoder=False)

        # Phase 2: end-to-end
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        run_phase("end-to-end", e2e_epochs, self.parameters(), train_encoder=True)
        self.eval()

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def route(self, prev_dst: dict[str, str], agent_utt: str, user_utt: str) -> tuple[str, dict]:
        """Returns ("slm"|"llm", details) by thresholding P(LLM) at 0.5."""
        self.eval()
        text = encode_triplet(prev_dst, agent_utt, user_utt)
        p_llm = torch.sigmoid(self(text if isinstance(text, list) else [text])).item()
        assigned = "llm" if p_llm >= self.threshold else "slm"
        return assigned, {"assigned": assigned, "p_llm": round(p_llm, 4),
                          "reason": "mrl_classifier"}

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.encoder.save(str(p / "encoder"))
        torch.save(self.head.state_dict(), p / "head.pt")
        with open(p / "meta.json", "w") as f:
            json.dump({"hidden_dim": self.head[0].out_features,
                       "threshold": self.threshold}, f)
        print(f"[MRL] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "MRLRouter":
        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)
        model = cls(backbone=str(p / "encoder"), hidden_dim=meta["hidden_dim"])
        model.head.load_state_dict(torch.load(p / "head.pt", map_location="cpu"))
        model.threshold = meta.get("threshold", 0.5)
        model.to(model.encoder.device)
        model.eval()
        print(f"[MRL] Loaded from {path}")
        return model


# ─── CLI ──────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="CADRE margin-ranked learned router (Section 4.1)")
    parser.add_argument("mode", choices=["train", "route"])
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--agent_utt", default="")
    parser.add_argument("--user_utt", default="")
    parser.add_argument("--prev_dst", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    rtr_cfg = config.get("router", {})
    mrl_cfg = config.get("mrl_router", {})
    proc_dir = Path(paths.get("processed_dir", "./data/processed"))
    pool_dir = Path(paths.get("expert_pool_dir", "./data/expert_pools"))
    save_path = str(pool_dir / "mrl_router")

    if args.mode == "train":
        holdout = load_jsonl(str(proc_dir / "holdout.jsonl"))
        with open(pool_dir / "slm_holdout_preds.json") as f:
            slm_preds = json.load(f)
        with open(pool_dir / "llm_holdout_preds.json") as f:
            llm_preds = json.load(f)

        data = build_discriminative_set(holdout, slm_preds, llm_preds)
        model = MRLRouter(
            backbone=rtr_cfg.get("retriever_backbone", "sentence-transformers/all-mpnet-base-v2"),
            hidden_dim=mrl_cfg.get("hidden_dim", 128),
        )
        model.to(model.encoder.device)
        model.fit(
            data,
            lr=mrl_cfg.get("learning_rate", 2e-5),
            batch_size=mrl_cfg.get("batch_size", 16),
            warmup_epochs=mrl_cfg.get("warmup_epochs", 2),
            e2e_epochs=mrl_cfg.get("e2e_epochs", 3),
        )
        model.save(save_path)

    elif args.mode == "route":
        model = MRLRouter.load(save_path)
        prev_dst = string_to_state(args.prev_dst) if args.prev_dst else {}
        expert, details = model.route(prev_dst, args.agent_utt, args.user_utt)
        print(f"\n[MRL] Assigned to: {expert.upper()}")
        print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
