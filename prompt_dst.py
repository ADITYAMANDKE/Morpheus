"""
prompt_dst.py
─────────────
Prompt-DST: Fine-tuning and inference with FLAN-T5-large for Dialogue State
Tracking, implementing equation (2) and (3) from the Morpheus paper.

    TLB_t = SLM(T, DST_{t-1}, A_{t-1}, U_t)
    max log P(TLB_t | T, DST_{t-1}, A_{t-1}, U_t)

Key design choices vs. paper:
  - FLAN-T5-large (770M) instead of T5-base (220M): better instruction following
    and schema grounding, fits in ≤16GB GPU with bf16.
  - Greedy decoding at inference (paper default).
  - TLB aggregation with slot-value replacement per paper footnote 1.

Usage:
    # Fine-tune
    python prompt_dst.py train --config config.yaml

    # Evaluate on val set
    python prompt_dst.py eval --config config.yaml --checkpoint ./models/prompt_dst/best

    # Single-turn inference
    python prompt_dst.py infer --config config.yaml \\
        --agent_utt "Do you have a price range in mind?" \\
        --user_utt "Yes, something cheap please." \\
        --prev_dst "hotel-semi-type = hotel"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# Internal helpers
from data_preprocessing import (
    load_jsonl,
    string_to_state,
    state_to_string,
    format_input,
    SCHEMA_PROMPT,
)
from evaluate import compute_tlb_jga, compute_dst_jga


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DSTDataset(Dataset):
    """
    PyTorch Dataset wrapping preprocessed .jsonl turn-level examples.

    Each example yields:
        input_ids      : tokenised model input  (T, DST_{t-1}, A_{t-1}, U_t)
        attention_mask : padding mask for input
        labels         : tokenised TLB target   (slot = value | ...)
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer: AutoTokenizer,
        max_input_length: int = 1024,
        max_output_length: int = 256,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        # Tokenise input
        enc = self.tokenizer(
            ex["input_text"],
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
        )
        # Tokenise target (TLB)
        # In older transformers, as_target_tokenizer() switched the tokenizer
        # into "decoder mode" for T5. In v5+, this is no longer needed — T5
        # tokenizes source and target identically, so we call it directly.
        dec = self.tokenizer(
            ex["target_text"],
            max_length=self.max_output_length,
            truncation=True,
            padding=False,
        )
        labels = dec["input_ids"]
        # Replace padding token id with -100 so it's ignored in loss
        labels = [l if l != self.tokenizer.pad_token_id else -100 for l in labels]

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            # Keep raw strings for evaluation
            "target_text": ex["target_text"],
            "prev_dst": json.dumps(ex["prev_dst"]),
            "dialogue_id": ex["dialogue_id"],
            "turn_idx": ex["turn_idx"],
        }


# ─── TLB Aggregator ───────────────────────────────────────────────────────────

def aggregate_tlbs(tlbs: list[dict[str, str]]) -> dict[str, str]:
    """
    Aggregates a sequence of Turn-Level Beliefs into a full dialogue state.

    Per paper footnote 1: later values overwrite earlier ones for the same slot.
    This matches how DST_{t} is built from TLB_{1..t}.

    Args:
        tlbs: Ordered list of TLB dicts from turn 1 to t.

    Returns:
        Accumulated dialogue state DST_t.
    """
    dst: dict[str, str] = {}
    for tlb in tlbs:
        dst.update(tlb)
    return dst


# ─── Metrics Callback ─────────────────────────────────────────────────────────

def decode_predictions(
    predictions,
    tokenizer: AutoTokenizer,
) -> list[str]:
    """Decodes model output token ids to strings, skipping special tokens."""
    if hasattr(predictions, "predictions"):
        preds = predictions.predictions
    else:
        preds = predictions
    # Replace -100 (ignored tokens) with pad_token_id before decoding
    preds = [
        [(p if p != -100 else tokenizer.pad_token_id) for p in pred]
        for pred in preds
    ]
    return tokenizer.batch_decode(preds, skip_special_tokens=True)


# ─── Model Wrapper ────────────────────────────────────────────────────────────

class PromptDST:
    """
    Wrapper around FLAN-T5-large for Prompt-DST.

    Handles:
        - Training with HuggingFace Seq2SeqTrainer
        - Inference (single turn or batched)
        - TLB → DST aggregation across a full dialogue
    """

    def __init__(self, config: dict, device: Optional[str] = None):
        self.config = config
        self.slm_cfg = config.get("prompt_dst", {})
        self.backbone = self.slm_cfg.get("backbone", "google/flan-t5-large")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PromptDST] Using device: {self.device}")
        print(f"[PromptDST] Backbone: {self.backbone}")

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSeq2SeqLM] = None

    def load(self, checkpoint_path: Optional[str] = None):
        """Loads tokenizer and model from HuggingFace hub or a local checkpoint."""
        source = checkpoint_path if checkpoint_path else self.backbone
        print(f"[PromptDST] Loading from: {source}")
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            source,
            dtype=torch.bfloat16 if self.slm_cfg.get("bf16", True) else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        # Enable gradient checkpointing to reduce GPU memory usage during
        # training. Instead of storing all intermediate activations in memory,
        # it recomputes them during the backward pass. This trades ~20-30%
        # slower training for ~40% less memory — essential on the T4 (15GB).
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        return self

    def train(
        self,
        train_examples: list[dict],
        val_examples: list[dict],
        output_dir: str,
    ):
        """
        Fine-tunes FLAN-T5 on the training set.

        Training objective (eq. 3):
            max log P(TLB_t | T, DST_{t-1}, A_{t-1}, U_t)

        Implemented via cross-entropy loss on decoder output tokens
        (standard seq2seq teacher-forcing).

        Args:
            train_examples : Preprocessed examples from data_preprocessing.py
            val_examples   : Validation set for early stopping
            output_dir     : Where to save checkpoints and the best model
        """
        if self.tokenizer is None:
            self.load()

        cfg = self.slm_cfg
        max_in  = cfg.get("max_input_length", 1024)
        max_out = cfg.get("max_output_length", 256)

        train_ds = DSTDataset(train_examples, self.tokenizer, max_in, max_out)
        val_ds   = DSTDataset(val_examples,   self.tokenizer, max_in, max_out)

        collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        use_bf16 = cfg.get("bf16", True) and torch.cuda.is_available()

        # Compute warmup_steps from warmup_ratio (warmup_ratio is deprecated in v5.2+)
        total_steps = (
            len(train_ds)
            // cfg.get("batch_size", 8)
            // cfg.get("gradient_accumulation_steps", 4)
            * cfg.get("num_epochs", 10)
        )
        warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg.get("num_epochs", 10),
            per_device_train_batch_size=cfg.get("batch_size", 8),
            per_device_eval_batch_size=cfg.get("eval_batch_size", 16),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
            learning_rate=cfg.get("learning_rate", 5e-4),
            weight_decay=cfg.get("weight_decay", 0.01),
            warmup_steps=warmup_steps,
            bf16=use_bf16,
            gradient_checkpointing=True,
            predict_with_generate=True,
            generation_max_length=max_out,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_tlb_jga",
            greater_is_better=True,
            logging_steps=50,
            log_level="warning",               # Suppresses per-step eval logs; keeps progress bar
            report_to="none",  # Set to "wandb" if desired
            dataloader_num_workers=2,
        )

        def compute_metrics(eval_preds):
            preds_str = decode_predictions(eval_preds, self.tokenizer)
            labels = eval_preds.label_ids
            labels_str = decode_predictions(
                type("P", (), {"predictions": labels})(),
                self.tokenizer
            )
            tlb_jga = compute_tlb_jga(preds_str, labels_str)
            return {"tlb_jga": tlb_jga}

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.get("early_stopping_patience", 3)
                )
            ],
        )

        print(f"\n[PromptDST] Starting training — {len(train_ds)} train, {len(val_ds)} val turns")

        # Auto-resume from latest checkpoint if a previous run was interrupted
        resume_ckpt = None
        ckpt_dirs = sorted(Path(output_dir).glob("checkpoint-*"), key=os.path.getmtime)
        if ckpt_dirs:
            resume_ckpt = str(ckpt_dirs[-1])
            print(f"[PromptDST] Resuming from checkpoint: {resume_ckpt}")
        else:
            print("[PromptDST] No existing checkpoints found — training from scratch.")

        trainer.train(resume_from_checkpoint=resume_ckpt)

        # Save best model
        best_path = Path(output_dir) / "best"
        trainer.save_model(str(best_path))
        self.tokenizer.save_pretrained(str(best_path))
        print(f"[PromptDST] Best model saved → {best_path}")
        return trainer

    @torch.inference_mode()
    def predict_turn(
        self,
        input_text: str,
    ) -> str:
        """
        Runs greedy decoding for a single turn input string.

        Implements the inference procedure from the paper:
            "a greedy decoding procedure is directly applied, i.e., only the
             most likely token in the given model vocabulary is predicted at
             each decoding step."

        Args:
            input_text: Formatted input string from format_input().

        Returns:
            Predicted TLB as a string ("slot = value | ...").
        """
        assert self.tokenizer and self.model, "Call .load() first"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.slm_cfg.get("max_input_length", 1024),
            truncation=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.slm_cfg.get("max_output_length", 256),
            do_sample=False,       # Greedy
            num_beams=1,           # Greedy
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.inference_mode()
    def predict_batch(self, input_texts: list[str]) -> list[str]:
        """Batched version of predict_turn for efficient evaluation."""
        assert self.tokenizer and self.model, "Call .load() first"
        cfg = self.slm_cfg
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=cfg.get("max_input_length", 1024),
            truncation=True,
            padding=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=cfg.get("max_output_length", 256),
            do_sample=False,
            num_beams=1,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def predict_dialogue(
        self,
        turns: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        Runs Prompt-DST over all turns of a dialogue, maintaining state.

        Implements the TLB aggregation: DST_t = DST_{t-1} ∪ TLB_t
        (with slot-value replacement for updated slots).

        Args:
            turns: List of turn dicts (from parse_dialogue) for one dialogue.

        Returns:
            - predicted_tlbs  : List of predicted TLB dicts per turn.
            - final_dst       : Final accumulated dialogue state.
        """
        predicted_tlbs: list[dict] = []
        accumulated_dst: dict[str, str] = {}

        for turn in turns:
            # Use accumulated_dst (not ground truth) as prev_dst at inference time
            input_text = format_input(
                prev_dst=accumulated_dst,
                agent_utt=turn["agent_utt"],
                user_utt=turn["user_utt"],
            )
            pred_str = self.predict_turn(input_text)
            pred_tlb = string_to_state(pred_str)

            predicted_tlbs.append(pred_tlb)
            accumulated_dst.update(pred_tlb)  # Replace updated slots

        return predicted_tlbs, accumulated_dst

    def evaluate(
        self,
        examples: list[dict],
        batch_size: int = 16,
    ) -> dict:
        """
        Evaluates Prompt-DST on a set of turn-level examples.

        Computes:
            - TLB JGA : Turn-level belief accuracy (new/changed slots only)
            - DST JGA : Joint goal accuracy on accumulated states

        Note: For DST JGA, we need full dialogue context; here we evaluate
        turn-level DST JGA using gold prev_dst (upper bound for turn isolation).

        Args:
            examples  : List of turn examples from load_jsonl().
            batch_size: Inference batch size.

        Returns:
            Dict with 'tlb_jga' and 'dst_jga' scores.
        """
        assert self.tokenizer and self.model, "Call .load() first"
        self.model.eval()

        all_preds: list[str] = []
        all_targets: list[str] = []
        all_dst_preds: list[str] = []
        all_dst_targets: list[str] = []

        # Group by dialogue for proper DST aggregation
        dialogues: dict[str, list] = {}
        for ex in examples:
            dialogues.setdefault(ex["dialogue_id"], []).append(ex)
        # Sort turns within each dialogue
        for did in dialogues:
            dialogues[did].sort(key=lambda x: x["turn_idx"])

        print(f"[PromptDST] Evaluating {len(examples)} turns across {len(dialogues)} dialogues...")

        for did, turns in dialogues.items():
            accumulated_dst: dict[str, str] = {}
            for turn in turns:
                input_text = format_input(
                    prev_dst=accumulated_dst,
                    agent_utt=turn["agent_utt"],
                    user_utt=turn["user_utt"],
                )
                pred_str = self.predict_turn(input_text)
                pred_tlb = string_to_state(pred_str)

                # Accumulate predicted DST
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
        }
        print(f"[PromptDST] TLB JGA: {results['tlb_jga']:.2f}%  |  DST JGA: {results['dst_jga']:.2f}%")
        return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Prompt-DST with FLAN-T5-large")
    parser.add_argument("mode", choices=["train", "eval", "infer"])
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to saved model checkpoint (eval/infer mode)")
    # Infer-mode args
    parser.add_argument("--agent_utt", default="", help="Previous agent utterance")
    parser.add_argument("--user_utt",  default="", help="Current user utterance")
    parser.add_argument("--prev_dst",  default="", help="Previous DST as 'slot=value|...'")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})  

    model = PromptDST(config)

    if args.mode == "train":
        model.load()
        train_path = Path(paths.get("processed_dir", "./data/processed")) / "train.jsonl"
        val_path   = Path(paths.get("processed_dir", "./data/processed")) / "val.jsonl"
        train_ex   = load_jsonl(str(train_path))
        val_ex     = load_jsonl(str(val_path))
        model.train(train_ex, val_ex, paths.get("model_dir", "./models/prompt_dst"))

    elif args.mode == "eval":
        ckpt = args.checkpoint or str(Path(paths.get("model_dir", "./models/prompt_dst")) / "best")
        model.load(ckpt)
        val_path = Path(paths.get("processed_dir", "./data/processed")) / "val.jsonl"
        val_ex   = load_jsonl(str(val_path))
        results  = model.evaluate(val_ex)
        print(json.dumps(results, indent=2))

    elif args.mode == "infer":
        ckpt = args.checkpoint or str(Path(paths.get("model_dir", "./models/prompt_dst")) / "best")
        model.load(ckpt)
        prev_dst = string_to_state(args.prev_dst) if args.prev_dst else {}
        input_text = format_input(prev_dst, args.agent_utt, args.user_utt)
        print("\n[Input]\n", input_text)
        pred = model.predict_turn(input_text)
        print("\n[Predicted TLB]\n", pred)
        print("\n[Parsed]\n", string_to_state(pred))


if __name__ == "__main__":
    main()
