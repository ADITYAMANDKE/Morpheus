# Morpheus

> **Efficient Orchestration of Language Models for Dialogue State Tracking**

Morpheus is a retrieval-based routing system for Dialogue State Tracking (DST) on task-oriented dialogues, built on top of the [OrchestraLLM paper](https://arxiv.org/pdf/2311.09758). It uses a SenBERT bi-encoder to dynamically route each conversation turn to either a fine-tuned **SLM** (FLAN-T5-large) or an **LLM** (Gemini) via KNN majority vote over expert pools. On top of the base routing, Morpheus adds **domain-aware routing** — the SLM is first evaluated on a holdout set to produce per-domain predictions, and per-domain reliability scores are then computed from the expert pool distributions (the ratio of turns where the SLM was correct vs where the LLM was correct for each domain). At inference time, when the user switches to a new domain whose SLM reliability score falls below the `reliability_threshold`, the KNN decision is overridden and the turn is escalated to the LLM. The threshold itself is calibrated offline using `calibrate_threshold.py`, which sweeps candidate values (0.3–0.8) on the holdout set and selects the one that overrides roughly 5–15% of turns for the best accuracy-cost trade-off (default: 0.40).

## How It Works

At each dialogue turn, Morpheus:

1. **Encodes** the current context — previous dialogue state, agent utterance, and user utterance — into a dense vector using a SenBERT bi-encoder.
2. **Retrieves** the top-K nearest neighbours from pre-built SLM and LLM expert pools.
3. **Votes** — majority vote decides which expert handles the turn.
4. **(Optional) Domain-aware override** — if the user switches to a new domain where the SLM has historically low reliability, the turn is automatically routed to the LLM.
5. **Predicts** the Turn-Level Belief (TLB) update using the chosen expert.
6. **Aggregates** the TLB into the running dialogue state.

```
                              ┌─────────────────────────────────────┐
                              │         SLM Expert (Prompt-DST)     │
User Turn ──► SenBERT ──► KNN ├─────────────────────────────────────┤──► TLB ──► DST
              Encoder    Vote │         LLM Expert (IC-DST)         │
                              └─────────────────────────────────────┘
                                  ▲ domain-switch override (optional)
```

## Components

| Module | File | Description |
|--------|------|-------------|
| **Morpheus** | `morpheus.py` | Full pipeline orchestrator — loads all components and runs end-to-end evaluation or interactive inference. |
| **Prompt-DST** | `prompt_dst.py` | Fine-tunes FLAN-T5-large on a small subset (5%) of MultiWOZ data to predict turn-level belief updates. |
| **IC-DST** | `ic_dst.py` | Uses Gemini with K in-context exemplars for few-shot dialogue state tracking — no fine-tuning required. |
| **Router** | `router.py` | SenBERT bi-encoder that retrieves nearest neighbours from expert pools and assigns turns via majority vote. Includes contrastive fine-tuning for improved routing. |
| **Domain Router** | `domain_router.py` | Domain-aware routing overlay — detects domain switches and overrides KNN routing when the SLM is weak on newly activated domains. |
| **Data Preprocessing** | `data_preprocessing.py` | Converts raw MultiWOZ 2.4 JSON into model-ready turn-level JSONL files with train/holdout/val splits. |
| **Evaluation** | `evaluate.py` | Computes Joint Goal Accuracy (JGA) at both turn-level (TLB) and dialogue-level (DST). |
| **Threshold Calibration** | `calibrate_threshold.py` | Calibrates the domain reliability threshold on a holdout set for optimal routing. |

## Quick Start

### Google Colab (Recommended)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload the notebook from `notebooks/` to Google Colab.
2. Connect to a **T4 GPU** runtime.
3. Follow the cells in order — the notebook handles setup, data, training, and evaluation.

### Local Setup

```bash
git clone https://github.com/<your-username>/morpheus.git
cd morpheus
pip install -r requirements.txt
```

**Data:** Download [MultiWOZ 2.4](https://github.com/smartyfh/MultiWOZ2.4) and place `data.json` in `data/multiwoz/`.

### Running the Pipeline

```bash
# 1. Preprocess MultiWOZ data into train/val/holdout splits
python data_preprocessing.py --config config.yaml

# 2. Fine-tune the SLM expert (Prompt-DST on FLAN-T5-large)
python prompt_dst.py train --config config.yaml

# 3. Evaluate the SLM expert standalone
python prompt_dst.py eval --config config.yaml --checkpoint ./models/prompt_dst/best

# 4. Build expert pools for the router
python router.py build_pools --config config.yaml

# 5. Run full Morpheus evaluation (requires Gemini API key)
export GEMINI_API_KEY="your-key"
python morpheus.py eval --config config.yaml

# 6. Interactive single-turn inference
python morpheus.py infer --config config.yaml
```

## Project Structure

```
morpheus/
├── config.yaml              # All hyperparameters and paths
├── requirements.txt         # Python >= 3.10 dependencies
├── morpheus.py              # Full pipeline orchestrator
├── prompt_dst.py            # SLM expert — FLAN-T5-large fine-tuning & inference
├── ic_dst.py                # LLM expert — Gemini few-shot DST
├── router.py                # SenBERT retriever, expert pools, contrastive training
├── domain_router.py         # Domain-switch detection & override logic
├── data_preprocessing.py    # MultiWOZ → model-ready JSONL
├── evaluate.py              # JGA / TLB metrics
├── calibrate_threshold.py   # Holdout-based threshold calibration
├── notebooks/               # Google Colab notebook(s)
├── data/                    # Raw & processed data (gitignored)
├── models/                  # Checkpoints (gitignored)
└── results/                 # Evaluation outputs (gitignored)
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| **TLB JGA** | Turn-Level Belief Joint Goal Accuracy — measures whether all slot changes predicted for a single turn exactly match the gold standard. |
| **DST JGA** | Dialogue State Tracking Joint Goal Accuracy — measures whether the full accumulated dialogue state matches the gold state at each turn. |

## Configuration

All hyperparameters live in `config.yaml`. Key settings:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `few_shot_ratio` | 0.05 | 5% of MultiWOZ training data for SLM fine-tuning |
| `batch_size` | 2 | Fits T4 GPU (15 GB VRAM); increase for larger GPUs |
| `gradient_accumulation_steps` | 16 | Effective batch size = 2 × 16 = 32 |
| `num_epochs` | 10 | With early stopping (patience = 3) |
| `top_k` | 10 | Majority vote over K neighbours for routing |
| `domain_aware` | true | Enable domain-switch override routing |
| `reliability_threshold` | 0.40 | Minimum SLM reliability to trust; below this routes to LLM |

## License

MIT
