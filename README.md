# FUC-RAG
FAITHFUL UNDER CONFLICT: SELECTIVE CONFLICT CONSTRAINTS AND ADAPTIVEPARAMETRIC SUPPRESSION FOR RETRIEVAL-AUGMENTED GENERATION

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{TODO,
  title     = {TODO},
  author    = {TODO},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

> The citation entry will be updated after the paper metadata is finalized.

## Overview
This repository provides the core training and evaluation code used in our work on knowledge conflicts in retrieval-augmented generation:
- **Selective Conflict Constraints (SCC)** introduces a training-side conflict constraint that selectively penalizes competitive parametric answers at divergent answer tokens. 
- **Adaptive Parametric Suppression (APS)** adjusts the inference-time suppression strength of knowledge-critical feed-forward layers according to dataset-level conflict and reasoning characteristics.

The method is evaluated on **CoFaithfulQA**, which contains six subsets:
- HotpotQA
- NaturalQuestions-Short
- NewsQA
- SearchQA
- SQuAD
- TriviaQA-web

The main evaluation metrics are:
- **ConR ↑**: contextual response rate
- **MemR ↓**: parametric-memory response rate
- **MR ↓**: memory-dominance rate

## Project Structure
The public repository contains the code and metadata required to understand and reproduce the released experimental pipeline. 

```text
.
├── README.md
├── requirements.txt
│
├── configs/
│   ├── rgdu_train_config.yaml
│   └── inference_ratio_config.yaml
│
├── examples/
│   ├── train_example.jsonl
│   └── CoFaithfulQA_example.jsonl
│
├── src/
│   ├── training/
│   │   ├── train.py
│   │   ├── arguments.py
│   │   └── utils.py
│   │
│   ├── evaluation/
│   │   └── eval_CoFaithfulQA.py
│   │
│   └── rgdu/
│       └── rgdu_loss.py
│
├── scripts/
│   ├── train_rgdu.sh
│   ├── eval_fixed_ratio.sh
│   └── eval_ratio_sweep.sh
│
├── results/
│   ├── rgdu_per_dataset_results.csv
│   ├── rgdu_best_by_mr.csv
│   ├── rgdu_macro_by_ratio.csv
│   └── official_vs_rgdu_best_mr_ratio.csv
│
└── DATA_FORMAT.md
```

## Requirements
We recommend using Python 3.10 or later.
Install the released dependencies with:

```bash
pip install -r requirements.txt
```

### Modified Transformers Source
The original experimental implementation uses a locally modified Transformers source tree to support activation suppression and the RGDU training objective.

If the public release includes the modified model implementation, make sure the repository source is placed before the system-installed Transformers package in `PYTHONPATH`.

For example:
```bash
export PYTHONPATH="$PWD/src/transformers/src:$PWD/src/training${PYTHONPATH:+:$PYTHONPATH}"
```

Do not replace the modified implementation with an unmodified PyPI Transformers package when reproducing the activation-suppression experiments.

## Environment
The main experiments were conducted with:
```text
Base model: Meta-Llama-3-8B-Instruct
Training precision: BF16
Training GPUs: 2 × NVIDIA RTX 4090 D
Maximum sequence length: 1024
Effective batch size: 32
Training steps: 2100
```

Exact Python package versions used in the public release are listed in `requirements.txt`.

## Datasets
### CoFaithfulQA
Evaluation is performed on the public CoFaithfulQA test set, covering six QA subsets:
```text
HotpotQA
NaturalQuestions-Short
NewsQA
SearchQA
SQuAD
TriviaQA-web
```

Please obtain the benchmark from its original source and place the processed test files under your local data directory.
A CoFaithfulQA-style example is provided in:
```text
examples/CoFaithfulQA_example.jsonl
```

### Training Data
The full training data used in our experiments are not redistributed in this repository.
A synthetic example showing the expected training format is provided in:
```text
examples/train_example.jsonl
```

The training record contains the following main fields:
```text
source_index
rag_input
raw_input
output
parametric_answer
parametric_answer_freq
conflict_valid_v31
```

See `DATA_FORMAT.md` for detailed field descriptions.

## Quick Start
### 1. Prepare the Base Model
Download or prepare:
```text
Meta-Llama-3-8B-Instruct
```

Then update the model path in:
```text
configs/rgdu_train_config.yaml
```

or pass the corresponding path through the training script.

### 2. Prepare the Data
Prepare your training and evaluation data according to:
```text
examples/train_example.jsonl
examples/CoFaithfulQA_example.jsonl
DATA_FORMAT.md
```

The full datasets are not included in this repository.

### 3. Train with SCC
The reference training configuration is provided in:
```text
configs/rgdu_train_config.yaml
```
Key settings include:
```text
max_steps                  = 2100
max_length                 = 1024
effective_batch_size       = 32
learning_rate              = 1e-4
warmup_ratio               = 0.1
weight_decay               = 1e-5
LoRA rank                  = 64
LoRA alpha                 = 64
RGDU eta                   = 0.1
RGDU gate margin           = 1.0
inhibition layers          = 19-26
training inhibit strength  = 0.0
```

Run:
```bash
bash scripts/train_rgdu.sh
```

> `configs/rgdu_train_config.yaml` is provided as a reproducibility reference. If the released training script uses command-line arguments, update the paths in the script before running.

### 4. Evaluate a Fixed Suppression Ratio
To evaluate a trained model using a fixed inference-time suppression coefficient:
```bash
bash scripts/eval_fixed_ratio.sh
```

The suppression coefficient follows the convention:
```text
lambda = 0.0  -> strongest / full activation suppression
lambda = 1.0  -> no activation suppression
```

### 5. Run the Inference-Ratio Sweep
The inference sweep configuration is documented in:
```text
configs/inference_ratio_config.yaml
```

Run:
```bash
bash scripts/eval_ratio_sweep.sh
```

The default controlled evaluation protocol uses:
```text
seed             = 42
max_new_tokens   = 32
do_sample        = True
temperature      = 0.6
top_p            = 0.9
inhibition layers = 19-26
```

### 6. Select Dataset-Specific Ratios
For the released SCC sweep, the best ratio is selected primarily by minimizing **MR**.
The controlled produced the following dataset-specific ratios:
| Dataset | Selected ratio |
|---|---:|
| NaturalQuestions-Short | 0.10 |
| NewsQA | 1.00 |
| SQuAD | 0.25 |
| SearchQA | 0.25 |
| TriviaQA-web | 0.25 |
| HotpotQA | 0.20 |

## Released Results
The public release includes summary-level results from the inference-ratio experiments.
```text
results/rgdu_per_dataset_results.csv
results/rgdu_best_by_mr.csv
results/rgdu_macro_by_ratio.csv
results/official_vs_rgdu_best_mr_ratio.csv
```

For the SCC model, dataset-specific ratio selection improved the six-dataset macro average relative to fixed `lambda = 0`:

| Setting | ConR ↑ | MemR ↓ | MR ↓ | EM ↑ |
|---|---:|---:|---:|---:|
| Fixed `lambda = 0` | 68.9350 | 6.2605 | 8.5100 | 64.4950 |
| Dataset-specific ratio | 69.8683 | 6.0643 | 8.1617 | 65.1333 |

The released CSV files contain the corresponding per-dataset and per-ratio results.

## Reproducibility Notes
For SCC training comparisons:
- keep the same base model;
- keep the same training data and optimization settings;
- keep the same suppression layers and suppression strength;
- change only the training objective being evaluated.

For controlled comparison of suppression ratios:
- keep the trained checkpoint fixed;
- keep the decoding seed fixed;
- keep generation settings unchanged;
- vary only the inference-time suppression coefficient;
- use the same suppressed layer set across compared runs.


## Get Involved
If you have questions about the released code or encounter reproducibility issues, please open a GitHub issue.
More files and documentation may be added after publication.
