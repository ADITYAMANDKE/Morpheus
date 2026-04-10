"""
data_preprocessing.py
─────────────────────
Converts raw MultiWOZ 2.4 JSON dialogues into model-ready examples for
Prompt-DST (FLAN-T5) and IC-DST (LLM few-shot).

Each example represents one dialogue turn and contains:
  - schema_prompt : string describing all trackable slots
  - prev_dst      : previous turn's accumulated dialogue state (TLB aggregated)
  - agent_utt     : most recent system utterance A_{t-1}
  - user_utt      : current user utterance U_t
  - tlb_target    : Turn-Level Belief — only NEW or CHANGED slot values at turn t
  - dst_target    : full accumulated dialogue state at turn t

Usage:
    python data_preprocessing.py --data_dir ./data/multiwoz --out_dir ./data/processed
"""

import json
import os
import argparse
import random
from pathlib import Path
from typing import Any
from collections import defaultdict

import yaml

# ─── MultiWOZ 2.4 Schema ──────────────────────────────────────────────────────
# Each domain maps to its informable (semi) and bookable slots.
MULTIWOZ_SCHEMA: dict[str, dict[str, list[str]]] = {
    "hotel": {
        "semi": ["name", "area", "parking", "pricerange", "stars", "internet", "type"],
        "book": ["stay", "day", "people"],
    },
    "restaurant": {
        "semi": ["name", "area", "food", "pricerange"],
        "book": ["time", "day", "people"],
    },
    "attraction": {
        "semi": ["name", "area", "type"],
        "book": [],
    },
    "train": {
        "semi": ["leaveAt", "destination", "day", "arriveBy", "departure"],
        "book": ["people"],
    },
    "taxi": {
        "semi": ["leaveAt", "destination", "departure", "arriveBy"],
        "book": [],
    },
    "hospital": {
        "semi": ["department"],
        "book": [],
    },
    "police": {
        "semi": [],
        "book": [],
    },
}

# Canonical slot names used in flattened representation
# Format: "{domain}-{group}{slot}"  e.g. "hotel-semiarea", "hotel-bookstay"
SLOT_SEPARATOR = "-"
NULL_VALUES = {"", "not mentioned", "none"}


# ─── Schema Prompt Builder ────────────────────────────────────────────────────

def build_schema_prompt(schema: dict[str, dict[str, list[str]]] = MULTIWOZ_SCHEMA) -> str:
    """
    Builds a textual schema table T (as in the paper) describing all
    trackable domains and slots.

    Example output:
        [domain: hotel] [slots: name, area, parking, pricerange, stars,
                                internet, type, book-stay, book-day, book-people]
        [domain: restaurant] ...
    """
    lines = ["Task: Extract dialogue state updates from the conversation.\n",
             "Schema (domain → slots to track):\n"]
    for domain, groups in schema.items():
        all_slots = groups.get("semi", []) + [f"book-{s}" for s in groups.get("book", [])]
        if all_slots:
            lines.append(f"  [{domain}]: {', '.join(all_slots)}")
    lines.append("")
    return "\n".join(lines)


SCHEMA_PROMPT: str = build_schema_prompt()


# ─── State Helpers ────────────────────────────────────────────────────────────

def extract_flat_state(metadata: dict) -> dict[str, str]:
    """
    Converts MultiWOZ metadata dict into a flat {slot_key: value} dict.
    Slot key format: "{domain}-{semi|book}-{slot_name}"
    Null/empty values are excluded.

    Args:
        metadata: The 'metadata' field from a system turn in MultiWOZ JSON.

    Returns:
        Flat dict of active (non-null) slot values.
    """
    state: dict[str, str] = {}
    for domain, groups in metadata.items():
        for group in ("semi", "book"):
            slots = groups.get(group, {})
            if isinstance(slots, dict):
                for slot, value in slots.items():
                    # MultiWOZ sometimes stores values as lists, e.g. ["monday"]
                    if isinstance(value, list):
                        value = " ".join(v for v in value if isinstance(v, str) and v.strip())
                    if not isinstance(value, str):
                        continue
                    if value and value.lower() not in NULL_VALUES:
                        key = f"{domain}-{group}-{slot}"
                        state[key] = value
    return state


def compute_tlb(prev_state: dict[str, str], curr_state: dict[str, str]) -> dict[str, str]:
    """
    Computes the Turn-Level Belief (TLB): only slots that are NEW or have
    CHANGED values compared to the previous state.

    Per paper footnote 1: "We replace previous values with updated ones for
    slots present in prior TLBs."

    Args:
        prev_state: Flat dialogue state from the previous system turn.
        curr_state: Flat dialogue state from the current system turn.

    Returns:
        Dict of only the changed/new slot-value pairs.
    """
    tlb: dict[str, str] = {}
    for slot, value in curr_state.items():
        if prev_state.get(slot) != value:
            tlb[slot] = value
    return tlb


def state_to_string(state: dict[str, str]) -> str:
    """
    Serialises a flat state dict to a readable string for model input/output.
    Format: "domain-group-slot = value | domain-group-slot = value | ..."
    Empty state returns "[none]".
    """
    if not state:
        return "[none]"
    return " | ".join(f"{k} = {v}" for k, v in sorted(state.items()))


def string_to_state(s: str) -> dict[str, str]:
    """
    Inverse of state_to_string. Parses model output back into a flat state dict.
    Handles malformed output gracefully.
    """
    if not s or s.strip() == "[none]":
        return {}
    state: dict[str, str] = {}
    for part in s.split("|"):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            k, v = k.strip(), v.strip()
            if k and v:
                state[k] = v
    return state


# ─── Input Formatter ──────────────────────────────────────────────────────────

def format_input(
    prev_dst: dict[str, str],
    agent_utt: str,
    user_utt: str,
    schema_prompt: str = SCHEMA_PROMPT,
) -> str:
    """
    Builds the full input string for Prompt-DST (FLAN-T5).

    Structure mirrors equation (2) from the paper:
        TLB_t = SLM(T, DST_{t-1}, A_{t-1}, U_t)

    Args:
        prev_dst   : Accumulated dialogue state from the previous turn.
        agent_utt  : Previous system/agent utterance A_{t-1}.
        user_utt   : Current user utterance U_t.
        schema_prompt: Textual schema table T.

    Returns:
        Formatted string ready for tokenisation.
    """
    prev_dst_str = state_to_string(prev_dst)
    return (
        f"{schema_prompt}"
        f"Previous dialogue state: {prev_dst_str}\n"
        f"Agent: {agent_utt.strip()}\n"
        f"User: {user_utt.strip()}\n"
        f"State update (new or changed slots only):"
    )


# ─── Dialogue Parser ──────────────────────────────────────────────────────────

def parse_dialogue(
    dialogue_id: str,
    dialogue: dict,
) -> list[dict[str, Any]]:
    """
    Parses a single MultiWOZ dialogue into a list of turn-level examples.

    MultiWOZ log structure (alternating):
        log[0]  = user turn   (metadata = {})
        log[1]  = system turn (metadata = full state)
        log[2]  = user turn   (metadata = {})
        log[3]  = system turn (metadata = full state)
        ...

    Args:
        dialogue_id: e.g. "SNG01856.json"
        dialogue   : The dialogue dict from the raw JSON.

    Returns:
        List of example dicts, one per user turn.
    """
    log = dialogue.get("log", [])
    examples: list[dict[str, Any]] = []

    prev_state: dict[str, str] = {}
    prev_agent_utt: str = ""  # A_{t-1}; empty at turn 1

    turn_idx = 0
    i = 0
    while i < len(log) - 1:
        user_entry = log[i]
        system_entry = log[i + 1]

        # Sanity: user turns have empty metadata
        if system_entry.get("metadata") is None:
            i += 2
            continue

        user_utt: str = user_entry.get("text", "")
        agent_utt_response: str = system_entry.get("text", "")
        curr_state = extract_flat_state(system_entry.get("metadata", {}))
        tlb = compute_tlb(prev_state, curr_state)

        example: dict[str, Any] = {
            "dialogue_id": dialogue_id,
            "turn_idx": turn_idx,
            # Raw text
            "agent_utt": prev_agent_utt,
            "user_utt": user_utt,
            # State representations (flat dicts)
            "prev_dst": dict(prev_state),
            "curr_dst": dict(curr_state),
            "tlb": dict(tlb),
            # Serialised strings (model input/output)
            "input_text": format_input(prev_state, prev_agent_utt, user_utt),
            "target_text": state_to_string(tlb),   # Model must predict this
            "dst_text": state_to_string(curr_state),
        }
        examples.append(example)

        # Advance
        prev_state = curr_state
        prev_agent_utt = agent_utt_response
        turn_idx += 1
        i += 2

    return examples


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def load_multiwoz(data_dir: str) -> dict[str, dict]:
    """
    Loads MultiWOZ JSON files from data_dir.
    Expects either:
      - A single 'data.json' file (original MultiWOZ format), OR
      - Individual per-dialogue JSON files.

    Returns:
        Dict mapping dialogue_id -> dialogue dict.
    """
    data_path = Path(data_dir)
    combined_file = data_path / "data.json"

    if combined_file.exists():
        print(f"Loading from {combined_file}")
        with open(combined_file) as f:
            return json.load(f)

    # Try loading individual files
    dialogues: dict[str, dict] = {}
    json_files = list(data_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {data_dir}. "
            "Download MultiWOZ 2.4 and place data.json there."
        )
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
            if isinstance(data, dict) and "log" in data:
                dialogues[jf.name] = data
            elif isinstance(data, dict):
                dialogues.update(data)
    return dialogues


def load_split_ids(data_dir: str) -> tuple[list[str], list[str]]:
    """
    Loads official MultiWOZ validation and test dialogue IDs from split files.

    MultiWOZ provides valListFile.json and testListFile.json as plain-text
    files with one dialogue ID per line (e.g. "PMUL0698.json").

    Args:
        data_dir: Path to the directory containing MultiWOZ data files.

    Returns:
        Tuple of (val_ids, test_ids).

    Raises:
        FileNotFoundError: If split files are not found in data_dir.
    """
    data_path = Path(data_dir)
    val_file = data_path / "valListFile.json"
    test_file = data_path / "testListFile.json"

    if not val_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Official split files not found in {data_dir}. "
            "Expected valListFile.json and testListFile.json. "
            "Download MultiWOZ 2.4 with split files."
        )

    with open(val_file) as f:
        val_ids = [line.strip() for line in f if line.strip()]
    with open(test_file) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    print(f"[Splits] Official val: {len(val_ids)}, test: {len(test_ids)} dialogue IDs loaded")
    return val_ids, test_ids


def split_dialogues(
    dialogues: dict[str, dict],
    data_dir: str,
    few_shot_ratio: float = 0.05,
    holdout_count: int = 100,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Splits dialogue IDs into train / holdout / val / test sets using official
    MultiWOZ split files, matching the Morpheus paper (Section 4.2).

    Paper methodology:
    - Use official train/val/test splits from MultiWOZ
    - train    : few_shot_ratio of official TRAIN dialogues (paper: 5%)
    - holdout  : holdout_count dialogues from official VAL set (paper: 100)
    - val      : remaining official VAL dialogues (for evaluation during training)
    - test     : official TEST set (untouched)

    Args:
        dialogues     : Full dialogue dict (all IDs from data.json).
        data_dir      : Path to MultiWOZ data directory (contains split files).
        few_shot_ratio: Fraction of TRAIN dialogues for few-shot training.
        holdout_count : Number of VAL dialogues for expert pool construction.
        seed          : Random seed for reproducibility.

    Returns:
        Dict with keys "train", "holdout", "val", "test".
    """
    rng = random.Random(seed)

    # Load official val/test IDs; derive train by exclusion
    official_val_ids, official_test_ids = load_split_ids(data_dir)
    val_set = set(official_val_ids)
    test_set = set(official_test_ids)
    all_ids = set(dialogues.keys())

    official_train_ids = sorted(all_ids - val_set - test_set)
    rng.shuffle(official_train_ids)

    # 5% of official training set → few-shot fine-tuning
    n_train = max(1, int(len(official_train_ids) * few_shot_ratio))
    train_ids = official_train_ids[:n_train]

    # From official val set: 100 → holdout, rest → val
    official_val_ids_copy = list(official_val_ids)
    rng.shuffle(official_val_ids_copy)
    holdout_ids = official_val_ids_copy[:holdout_count]
    val_ids = official_val_ids_copy[holdout_count:]

    # Official test set, untouched
    test_ids = list(official_test_ids)

    print(
        f"Split — train: {len(train_ids)} (5% of {len(official_train_ids)}), "
        f"holdout: {len(holdout_ids)}, val: {len(val_ids)}, test: {len(test_ids)}"
    )
    return {"train": train_ids, "holdout": holdout_ids, "val": val_ids, "test": test_ids}


def build_dataset(
    dialogues: dict[str, dict],
    dialogue_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Builds a flat list of turn-level examples from a set of dialogue IDs.
    """
    examples: list[dict[str, Any]] = []
    for did in dialogue_ids:
        if did in dialogues:
            examples.extend(parse_dialogue(did, dialogues[did]))
    return examples


def save_jsonl(examples: list[dict], path: str) -> None:
    """Saves examples to a .jsonl file (one JSON object per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples → {path}")


def load_jsonl(path: str) -> list[dict]:
    """Loads examples from a .jsonl file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ─── Statistics ───────────────────────────────────────────────────────────────

def print_stats(split_name: str, examples: list[dict]) -> None:
    """Prints basic dataset statistics for a split."""
    n_turns = len(examples)
    n_dialogues = len({e["dialogue_id"] for e in examples})
    n_with_tlb = sum(1 for e in examples if e["tlb"])

    slot_counts: dict[str, int] = defaultdict(int)
    for e in examples:
        for slot in e["tlb"]:
            slot_counts[slot] += 1

    top_slots = sorted(slot_counts.items(), key=lambda x: -x[1])[:5]
    print(f"\n[{split_name}]")
    print(f"  Dialogues : {n_dialogues}")
    print(f"  Turns     : {n_turns}")
    print(f"  Turns with non-empty TLB : {n_with_tlb} ({100*n_with_tlb/max(n_turns,1):.1f}%)")
    print(f"  Top-5 slots: {top_slots}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess MultiWOZ for Morpheus")
    parser.add_argument("--data_dir", default="./data/multiwoz",
                        help="Directory containing MultiWOZ JSON data")
    parser.add_argument("--out_dir", default="./data/processed",
                        help="Output directory for processed .jsonl files")
    parser.add_argument("--config", default="./config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    cfg: dict = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})

    few_shot_ratio = data_cfg.get("few_shot_ratio", 0.05)
    holdout_count  = data_cfg.get("holdout_dialogues", 100)

    # Load raw data
    dialogues = load_multiwoz(args.data_dir)
    print(f"Loaded {len(dialogues)} dialogues from {args.data_dir}")

    # Split using official MultiWOZ val/test split files
    splits = split_dialogues(
        dialogues, args.data_dir, few_shot_ratio, holdout_count, seed=args.seed
    )

    # Build and save each split
    out = Path(args.out_dir)
    for split_name, ids in splits.items():
        examples = build_dataset(dialogues, ids)
        print_stats(split_name, examples)
        save_jsonl(examples, str(out / f"{split_name}.jsonl"))

    # Also save schema prompt for reference
    schema_path = out / "schema_prompt.txt"
    schema_path.write_text(SCHEMA_PROMPT)
    print(f"\nSchema prompt saved → {schema_path}")
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
