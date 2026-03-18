# OrchestraLLM

> **Efficient Orchestration of Language Models for Dialogue State Tracking**

An implementation of the OrchestraLLM paper — a retrieval-based routing system that dynamically dispatches dialogue turns to either a fine-tuned SLM (FLAN-T5-large) or an LLM (Claude) for dialogue state tracking, achieving strong accuracy while minimising compute cost.

## Architecture

```
User Turn → [SenBERT Router] → SLM Expert (Prompt-DST / FLAN-T5-large)
                              → LLM Expert (IC-DST / Claude)
```

- **Prompt-DST** (`prompt_dst.py`): Fine-tunes FLAN-T5-large on 5% of MultiWOZ to predict turn-level belief updates.
- **IC-DST** (`ic_dst.py`): Uses Claude with K in-context exemplars for few-shot DST.
- **Router** (`router.py`): SenBERT bi-encoder retrieves nearest neighbours from expert pools and assigns turns via majority vote.
- **OrchestraLLM** (`orchestrallm.py`): Full pipeline orchestrating all components.

## Quick Start (Google Colab)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload the notebook from `notebooks/` to Google Colab.
2. Connect to a **T4 GPU** runtime.
3. Follow the cells in order — the notebook handles setup, data, training, and evaluation.

## Local Setup

```bash
git clone https://github.com/<your-username>/orchestrallm.git
cd orchestrallm
pip install -r requirements.txt
```

### Data

Download [MultiWOZ 2.4](https://github.com/smartyfh/MultiWOZ2.4) and place `data.json` in `data/multiwoz/`:

```bash
mkdir -p data/multiwoz
# Place data.json here
```

### Pipeline

```bash
# 1. Preprocess
python data_preprocessing.py --config config.yaml

# 2. Fine-tune Prompt-DST
python prompt_dst.py train --config config.yaml

# 3. Evaluate
python prompt_dst.py eval --config config.yaml --checkpoint ./models/prompt_dst/best

# 4. Full OrchestraLLM pipeline (requires expert pools + API key)
export ANTHROPIC_API_KEY="your-key"
python orchestrallm.py eval --config config.yaml
```

## Project Structure

```
orchestrallm/
├── config.yaml              # Hyperparameters and paths
├── requirements.txt         # Python dependencies
├── data_preprocessing.py    # MultiWOZ → model-ready examples
├── prompt_dst.py            # FLAN-T5-large fine-tuning & inference
├── ic_dst.py                # Claude few-shot DST
├── evaluate.py              # JGA / TLB metrics
├── router.py                # SenBERT retriever & expert pools
├── orchestrallm.py          # Full pipeline orchestrator
├── notebooks/               # Colab notebook(s)
├── data/                    # Raw & processed data (gitignored)
├── models/                  # Checkpoints (gitignored)
└── results/                 # Evaluation outputs (gitignored)
```

## Key Metrics

| Metric  | Description |
|---------|-------------|
| TLB JGA | Turn-Level Belief Joint Goal Accuracy — accuracy on per-turn slot changes |
| DST JGA | Dialogue State Tracking JGA — accuracy on full accumulated states |

## Configuration

All hyperparameters are in `config.yaml`. Key settings:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch_size` | 2 | Fits T4 GPU (15GB); increase for larger GPUs |
| `gradient_accumulation_steps` | 16 | Effective batch = batch_size × this |
| `num_epochs` | 10 | With early stopping (patience=3) |
| `few_shot_ratio` | 0.05 | 5% of data for training |

## License

MIT
