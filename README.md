## **CS263 Final Project**

A multi-approach content moderation system that uses LLMs to detect and block harmful content in multi-turn conversations. The project implements and evaluates several supervisor strategies — from zero-shot classification to multi-model ensembles — and includes a custom data generation pipeline for escalation detection research.

## Overview

The system acts as a real-time supervisor that sits between users and a chatbot, evaluating each message for harmful content before allowing a response. Three supervision approaches are implemented and compared:

1. **Zero-shot Supervisor** — A single LLM classifies user input with no examples, outputting `PASS` or `BLOCK` verdicts.
2. **Multi-shot Supervisor** — Few-shot learning with explicit examples of hate speech categories improves classification accuracy.
3. **Model Scalability** — Compares the same zero-shot prompting strategy across model scales (1B vs. 8B), evaluating whether larger models are more robust to adversarial escalation and better calibrated for early detection without additional prompt engineering.
4. **HateBERT** — Replicates the core idea behind GuardReasoner using a fine-tuned BERT encoder instead of an expensive chain-of-thought safety model. HateBERT receives the full cumulative conversation prefix at each turn, enabling context-aware escalation detection at a fraction of the compute cost.

## Project Structure

```
CS263_FinalProject_ContentModeration/
├── README.md
├── full_pipeline_v7.ipynb                 # Main end-to-end pipeline (start here)
├── content-moderation-supervisor.ipynb   # Individual supervisor implementations
├── data_engineering.ipynb                # Synthetic escalation data generation
└── conversation_to_eval.ipynb            # Evaluation framework & metrics
```

## Notebooks

### `full_pipeline.ipynb`
**Start here.** The main notebook that integrates all components into a complete end-to-end content moderation pipeline — from data ingestion through supervisor evaluation and metrics.

### `content-moderation-supervisor.ipynb`
Individual supervisor implementations developed during the project. Contains the zero-shot, multi-shot, ensemble, and GuardReasoner approaches with an interactive chatbot interface.

### `data_engineering.ipynb`
Generates synthetic escalation data using an uncensored LLM (Llama-3.3-8B). Features a multi-stage pipeline: initial draft generation, similarity grading, agreeability checking, iterative refinement, and transition blurring to create realistic radicalization progressions.

### `conversation_to_eval.ipynb`
Evaluation framework using the Kaggle Hate Speech & Offensive Language Dataset (24,783 labeled tweets). Generates multi-turn conversations with ground-truth labels and computes turn-level metrics (precision, recall, F1, FNR), conversation-level metrics (detection latency, miss rate), and safety-critical metrics (post-onset pass rate).

## Tech Stack

- **Models**: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.3-8B-Instruct, Hermes-3-Llama-3.1-8B, HateBERT
- **Inference**: `llama-cpp-python` (GGUF quantized models)
- **ML/NLP**: `transformers`, `accelerate`, `bitsandbytes`, `torch`
- **Evaluation**: `scikit-learn`, `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`

## Setup

### Requirements

```bash
pip install -U "numpy<2.0" "pandas==2.2.2" kagglehub huggingface_hub \
    transformers accelerate "bitsandbytes>0.37.0" trl==0.12.0 peft \
    llama-cpp-python scikit-learn matplotlib seaborn
```

### Hardware

- **GPU recommended** — models use full GPU offloading (`n_gpu_layers=-1`) for fast inference
- CPU fallback is supported but significantly slower

### Data & Models

- Models are automatically downloaded from Hugging Face Hub on first run (set `HF_TOKEN` for gated repos)
- The evaluation dataset is fetched via `kagglehub` (requires Kaggle credentials)
- Our generated dataset of contextual hate speech conversations is available on Hugging Face: [sherinechally/contextual-hate-speech-conversations](https://huggingface.co/datasets/sherinechally/contextual-hate-speech-conversations)

## Usage

1. Open any notebook in Jupyter or VS Code
2. Run cells sequentially — model downloads and dataset fetching are handled automatically
3. For interactive moderation, use the chatbot cells in `content-moderation-supervisor.ipynb`
4. For batch evaluation, run `conversation_to_eval.ipynb` end-to-end

## Evaluation Metrics

| Category | Metrics |
|---|---|
| **Turn-level** | Latency Detection, Recall, FP |
| **Conversation-level** | Macro Precision/Recall/F1, Detection Latency (turns to first block) |
| **Safety-critical** | Post-onset Pass Rate, % of harmful conversations never flagged |
| **Synthetic quality** | Type-Token Ratio, Distinct-1 & Distinct-2 (n-gram diversity) |

## Key Findings

- The **ensemble approach** is more robust than single-model supervision
- **Few-shot examples** significantly improve classification accuracy over zero-shot
- **Context-aware multi-turn analysis** detects escalation patterns better than isolated turn-level checks
- **Post-onset pass rate** is a critical safety metric for real-time deployment
