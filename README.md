# CADRE

> **CADRE: A Context- and Domain-Aware Routing Engine for Dialogue State Tracking**

CADRE is a retrieval-based routing system for Dialogue State Tracking (DST) on task-oriented dialogues. Each turn is routed to either a cheap fine-tuned **SLM** (FLAN-T5-large) or an **LLM** with in-context examples (Gemini 2.5 Flash). The system is:

- **Context-aware** — every routing decision is made over the full dialogue triplet (DST<sub>t−1</sub>, A<sub>t−1</sub>, U<sub>t</sub>), embedded with a SenBERT bi-encoder and matched against balanced SLM/LLM expert pools by KNN majority vote.
- **Domain-aware** — a reliability overlay detects domain switches in real time and escalates to the LLM whenever the user activates a domain on which the SLM's offline reliability r<sub>d</sub> falls below a calibrated threshold τ.

On MultiWOZ 2.4 the deployed configuration reaches 79.4% TLB JGA / 48.0% DST JGA while sending ~60% of turns to the SLM, outperforming both the always-LLM baseline and the base KNN router. This repository contains the complete implementation, including the two negative results reported in the paper (a learned margin-ranked classifier head and semantic in-context exemplar retrieval).

## Architecture

```
                                                     ┌──────────────────────────┐
Turn (DST_{t-1}, A_{t-1}, U_t) ──► SenBERT ──► KNN vote (K=10) ──►│  SLM expert  (Prompt-DST) │
                                    │                             ├──────────────────────────┤──► TLB_t ──► DST_t
                                    └─► domain-switch override ──►│  LLM expert  (IC-DST)     │
                                        (if r_d < τ, force LLM)   └──────────────────────────┘
```

| Paper section | Component | File |
|---|---|---|
| 2.1 | SLM expert — FLAN-T5-large fine-tuned on 5% of MultiWOZ train | `prompt_dst.py` |
| 2.2 | LLM expert — Gemini 2.5 Flash, K=10 exemplars, temperature 0 | `ic_dst.py` |
| 2.3 | Retrieval router — triplet encoding, expert pools, KNN majority vote | `router.py` |
| 2.4 | Domain-aware reliability overlay — r<sub>d</sub> (Eq. 1), keyword domain-switch detector, override rule | `domain_router.py` |
| 2.5 | Offline threshold calibration — τ sweep on cached holdout predictions, 5–15% band rule | `calibrate_threshold.py` |
| 3.1 | Data — MultiWOZ 2.4, official splits, 100-dialogue holdout | `data_preprocessing.py` |
| 3.1 | Metrics — TLB JGA, DST JGA, slot F1, per-domain TLB JGA | `evaluate.py` |
| 3 | Orchestrator — all configurations of Tables 1 and 3 | `cadre.py` |
| 4.1 | Margin-ranked learned router (negative result) | `mrl_router.py` |
| 4.2 | Semantic in-context exemplars (negative result) | `ic_dst.py` (`exemplar_selection: semantic`) |
| Fig. 2–3 | Per-domain gain and cost–quality plots | `results/plot_paper_figures.py` |
| App. B | SLM training curves | `results/plot_separate_training.py`, `results/trainer_state_epoch10.json` |
| App. C | Domain keyword list | `config.yaml` → `router.domain_keywords` |

## Quick Start

### Google Colab (recommended)

1. Upload `notebooks/CADRE_Colab.ipynb` to Google Colab.
2. Connect to a **T4 GPU** runtime.
3. Follow the cells in order — setup, data, SLM training, expert pools, calibration, and every evaluation configuration of the paper.

### Local setup

```bash
git clone https://github.com/ADITYAMANDKE/Morpheus.git cadre
cd cadre
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
```

**Data:** download [MultiWOZ 2.4](https://github.com/smartyfh/MultiWOZ2.4) and place `data.json`, `valListFile.json` and `testListFile.json` in `data/multiwoz/`.

## Reproducing the paper

```bash
# 1. Preprocess MultiWOZ into train (5%) / holdout (100 dialogues) / val / test
python data_preprocessing.py --config config.yaml

# 2. Fine-tune the SLM expert (Section 2.1, Appendix B)
python prompt_dst.py train --config config.yaml

# 3. Run both experts on the holdout set and save the predictions
#    (see notebook Cell 10) to data/expert_pools/{slm,llm}_holdout_preds.json

# 4. Build the balanced 100 + 100 expert pools (Section 2.3)
python router.py build_pools --config config.yaml

# 5. Calibrate τ on cached predictions — no API calls (Section 2.5)
python calibrate_threshold.py --config config.yaml --write

# 6. Table 1 — the four main configurations on the 204-turn headline slice
python cadre.py eval --max_turns 204 --force_expert slm       # SLM only
python cadre.py eval --max_turns 204 --force_expert llm       # LLM only
python cadre.py eval --max_turns 204 --no_domain_aware        # base KNN router
python cadre.py eval --max_turns 204                          # CADRE (full)

# 7. Table 3 — negative results (Section 4)
python mrl_router.py train                                    # 4.1 train the classifier head
python cadre.py eval --max_turns 204 --no_domain_aware --router mrl
python cadre.py eval --max_turns 204 --no_domain_aware --exemplar_selection semantic   # 4.2

# 8. Wide 736-turn run for per-domain trends (Section 3.3)
python cadre.py eval --max_turns 736

# 9. Figures 2 and 3 (+ markdown Tables 1/3) from the result files
python results/plot_paper_figures.py --turns 204
```

Every `cadre.py eval` run writes `results/<configuration>_<n>turns.json` containing TLB JGA, DST JGA, average slot F1, per-domain TLB JGA, routing share, domain-override rate, the reliability table r<sub>d</sub> and wall-clock time.

`--max_turns N` takes whole dialogues from the validation set in file order until N turns are covered, so every configuration is scored on exactly the same slice.

### Interactive inference

```bash
python cadre.py infer --config config.yaml
```

## Results (paper, MultiWOZ 2.4, 204-turn slice)

| | SLM only | LLM only | Base KNN router | **CADRE** |
|---|---|---|---|---|
| TLB JGA (%) | 72.06 | 77.94 | 75.00 | **79.41** |
| DST JGA (%) | 25.49 | 40.20 | 38.24 | **48.04** |
| Slot F1 (%) | 75.20 | 73.03 | 74.86 | **79.44** |
| Routing % SLM / LLM | 100 / 0 | 0 / 100 | 64.7 / 35.3 | 59.8 / 40.2 |
| Wall-clock (s) | 139.6 | 316.7 | 155.4 | 184.9 |

| Design alternative (Section 4) | TLB | DST | F1 | % SLM |
|---|---|---|---|---|
| + Semantic exemplars | 79.90 | 47.06 | 80.77 | 59.8 |
| MRL classifier head | 75.98 | 36.27 | 74.57 | 45.1 |

`results/base_knn_router_736_turns.json` is the wide base-KNN-router run used for per-domain trends in Section 3.3 (taxi 31.3%, train 74.1%, attraction 59.8%, hotel 57.5%, restaurant 63.3%).

## Configuration

All hyperparameters live in `config.yaml` and match Appendix A (Table 4) of the paper.

| Parameter | Value | Paper |
|---|---|---|
| `prompt_dst.backbone` | `google/flan-t5-large` | §2.1 |
| `data.few_shot_ratio` | 0.05 | §2.1 |
| `prompt_dst.batch_size` × `gradient_accumulation_steps` | 2 × 16 = 32 | §2.1 |
| `prompt_dst.num_epochs` / `early_stopping_patience` | 10 / 3 | §2.1 |
| `ic_dst.model` / `num_exemplars` / `temperature` | `gemini-2.5-flash` / 10 / 0.0 | §2.2 |
| `ic_dst.exemplar_selection` | `random` (`semantic` = §4.2) | §2.2 / §4.2 |
| `router.router_type` | `knn` (`mrl` = §4.1) | §2.3 / §4.1 |
| `router.retriever_backbone` | `sentence-transformers/all-mpnet-base-v2` | §2.3 |
| `router.top_k` / `tie_break` | 10 / `slm` | §2.3 |
| `router.slm_pool_size` / `llm_pool_size` | 100 / 100 | §2.3 |
| `data.holdout_dialogues` | 100 | §2.3 |
| `router.domain_aware` / `reliability_threshold` | true / 0.40 | §2.4 / §2.5 |
| `router.calibration.thresholds` / `override_rate_band` | {0.3 … 0.8} / [0.05, 0.15] | §2.5 |
| `mrl_router.*` | lr 2e-5, 2 + 3 epochs, batch 16, hidden 128, BCE + pos_weight | §4.1 |

**Not part of the paper.** `router.py train_retriever` (contrastive fine-tuning of the SenBERT retriever, inherited from OrchestraLLM) is kept for experimentation but disabled by default. CADRE uses the off-the-shelf encoder; set `router.use_fine_tuned_retriever: true` to load a fine-tuned one.

## Metrics

| Metric | Description |
|---|---|
| **TLB JGA** | Turn-Level Belief Joint Goal Accuracy — a turn is correct only if *all* predicted slot updates match gold. |
| **DST JGA** | Joint Goal Accuracy on the accumulated dialogue state at every turn. |
| **Slot F1** | Average per-turn slot-value F1 over the TLB. |
| **Per-domain TLB JGA** | TLB JGA restricted to turns whose gold or predicted TLB touches a domain (Table 2). |
| **Routing share / override rate** | % of turns routed to the SLM / LLM, and % escalated by the domain-switch override. |

## Acknowledgements

CADRE builds on [OrchestraLLM](https://aclanthology.org/2024.naacl-long.79/) (Lee et al., 2024) for expert-pool construction and KNN routing, and on [IC-DST](https://aclanthology.org/2022.findings-emnlp.193/) (Hu et al., 2022) for the LLM expert.

## License

MIT
