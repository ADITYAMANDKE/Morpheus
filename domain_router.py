"""
domain_router.py
────────────────
Domain-aware routing overlay for OrchestraLLM.

Detects domain switches in dialogue turns and overrides KNN routing
when the SLM is historically weak on newly activated domains.

Offline:
    compute_domain_reliability() → per-domain SLM reliability scores

Online (per turn):
    detect_domain_switch() → newly activated domains via keyword matching
    should_override_to_llm() → True if min reliability < threshold
"""

import re
from typing import Optional

from data_preprocessing import string_to_state


# ─── Default Domain Keywords (MultiWOZ 2.4) ──────────────────────────────────

DEFAULT_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "hotel": [
        "hotel", "room", "stay", "accommodation", "lodge",
        "inn", "guesthouse",
    ],
    "restaurant": [
        "restaurant", "food", "eat", "dining", "cuisine",
        "meal", "lunch", "dinner",
    ],
    "attraction": [
        "attraction", "museum", "park", "theatre", "cinema",
        "gallery", "visit", "sightseeing",
    ],
    "train": [
        "train", "railway", "rail", "depart", "arrive",
    ],
    "taxi": [
        "taxi", "cab", "ride", "pick up", "pickup", "drop off",
    ],
    "hospital": [
        "hospital", "department",
    ],
}


# ─── Offline: Domain Reliability Computation ─────────────────────────────────

def extract_domains_from_slots(slot_string: str) -> set[str]:
    """
    Extracts domain names from a TLB/DST slot string.

    Example:
        "hotel-semi-area = south | train-semi-day = friday"
        → {"hotel", "train"}

    Args:
        slot_string: Pipe-separated slot=value string.

    Returns:
        Set of domain names found in slot keys.
    """
    state = string_to_state(slot_string)
    domains = set()
    for key in state.keys():
        # Keys are like "hotel-semi-area", "train-book-people"
        parts = key.split("-")
        if parts:
            domains.add(parts[0])
    return domains


def extract_domains_from_dict(state_dict: dict[str, str]) -> set[str]:
    """
    Extracts domain names from a DST state dictionary.

    Example:
        {"hotel-semi-area": "south", "train-semi-day": "friday"}
        → {"hotel", "train"}
    """
    domains = set()
    for key in state_dict.keys():
        parts = key.split("-")
        if parts:
            domains.add(parts[0])
    return domains


def compute_domain_reliability(
    slm_pool: list[dict],
    llm_pool: list[dict],
) -> dict[str, float]:
    """
    Computes per-domain SLM reliability from expert pool distributions.

    For each domain, counts how many pool entries involve that domain
    in the SLM pool vs the LLM pool. Higher ratio = SLM is more reliable.

    Args:
        slm_pool: SLM expert pool entries (each has 'target_text' field).
        llm_pool: LLM expert pool entries.

    Returns:
        Dict mapping domain → reliability score in [0, 1].
    """
    slm_counts: dict[str, int] = {}
    llm_counts: dict[str, int] = {}

    for entry in slm_pool:
        domains = extract_domains_from_slots(entry.get("target_text", ""))
        for d in domains:
            slm_counts[d] = slm_counts.get(d, 0) + 1

    for entry in llm_pool:
        domains = extract_domains_from_slots(entry.get("target_text", ""))
        for d in domains:
            llm_counts[d] = llm_counts.get(d, 0) + 1

    all_domains = set(slm_counts.keys()) | set(llm_counts.keys())
    reliability = {}
    for domain in all_domains:
        s = slm_counts.get(domain, 0)
        l = llm_counts.get(domain, 0)
        reliability[domain] = s / (s + l) if (s + l) > 0 else 0.5

    return reliability


# ─── Online: Domain Switch Detection ─────────────────────────────────────────

def detect_domain_switch(
    prev_dst: dict[str, str],
    user_utt: str,
    domain_keywords: Optional[dict[str, list[str]]] = None,
) -> set[str]:
    """
    Detects newly activated domains in the current turn.

    Compares domains already present in prev_dst against domains
    detected via keyword matching in the user utterance.

    Args:
        prev_dst       : Previous accumulated dialogue state.
        user_utt       : Current user utterance.
        domain_keywords: Mapping of domain → list of keywords.

    Returns:
        Set of newly activated domain names (empty if no switch).
    """
    if domain_keywords is None:
        domain_keywords = DEFAULT_DOMAIN_KEYWORDS

    # Domains already active in the dialogue state
    active_domains = extract_domains_from_dict(prev_dst)

    # Detect candidate domains from user utterance via keyword matching
    text = user_utt.lower()
    candidate_domains: set[str] = set()

    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            # Multi-word keywords: simple substring match
            if " " in keyword:
                if keyword in text:
                    candidate_domains.add(domain)
                    break
            else:
                # Single-word keywords: word-boundary match
                if re.search(r"\b" + re.escape(keyword) + r"\b", text):
                    candidate_domains.add(domain)
                    break

    # New domains = candidates that aren't already in the dialogue state
    new_domains = candidate_domains - active_domains
    return new_domains


def should_override_to_llm(
    new_domains: set[str],
    reliability: dict[str, float],
    threshold: float = 0.5,
) -> bool:
    """
    Decides whether to hard-route to LLM based on domain reliability.

    If any newly activated domain has SLM reliability below the threshold,
    override the KNN decision and send to LLM.

    Args:
        new_domains : Set of newly detected domains.
        reliability : Per-domain SLM reliability scores.
        threshold   : Minimum reliability to trust SLM.

    Returns:
        True if should override to LLM, False otherwise.
    """
    if not new_domains:
        return False

    min_reliability = min(
        reliability.get(d, 0.5) for d in new_domains
    )
    return min_reliability < threshold
