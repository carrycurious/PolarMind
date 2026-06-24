# PolarMind

Multilingual polarization detection for competition-style NLP tasks.

| Task | Description | Output |
|------|-------------|--------|
| **Task 1** | Binary classification across 22 languages | `polarization` ∈ {0, 1} |
| **Task 2** | Fine-grained multi-label (7 languages) | five binary category columns |

## Project structure

```
PolarMind/
├── polarmind/           # Installable Python package
│   ├── cli/             # Command-line interface
│   ├── data/            # Dataset loading and path resolution
│   ├── inference/       # Prediction runners
│   ├── models/          # Model architectures
│   ├── pipelines/       # End-to-end train / infer workflows
│   ├── prompts/         # Task prompt templates
│   └── training/        # Training loops and custom trainers
├── configs/             # Default hyperparameters (YAML)
├── data/                # Competition datasets (not committed)
│   ├── task1/raw/       # Place task1.zip here
│   └── task2/raw/       # Place task2.zip here
├── artifacts/           # Checkpoints and submissions (gitignored)
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -e .
```

GPU recommended for Qwen QLoRA. Encoder models (LaBSE, XLM-R) run on CPU but train slowly.

> On Windows, `bitsandbytes` (4-bit QLoRA) is excluded from dependencies. Use WSL2 or Linux for Qwen training.

## Data

1. Obtain the official task archives.
2. Place them at:
   - `data/task1/raw/task1.zip`
   - `data/task2/raw/task2.zip`
3. The CLI extracts archives to `data/taskN/processed/` on first use.

Expected layout after extraction:

```
data/task1/processed/train/{lang}.csv
data/task1/processed/dev/{lang}.csv
data/task1/processed/test/{lang}.csv
data/task2/processed/train/{lang}.csv
data/task2/processed/dev/{lang}.csv
```

## Usage

All workflows are exposed through a single CLI:

```bash
python -m polarmind --help
```

### Train

```bash
# Task 1 — LaBSE encoder
python -m polarmind train task1 --model labse --epochs 3

# Task 1 — Qwen QLoRA (multilingual)
python -m polarmind train task1 --model qwen

# Task 2 — Qwen multi-label (custom delimiter-token loss)
python -m polarmind train task2 --delimiter " ::"
```

### Infer

```bash
python -m polarmind infer task1 --model labse --checkpoint artifacts/task1/labse/model.pt --split test
python -m polarmind infer task1 --model qwen --checkpoint artifacts/task1/qwen --split test
python -m polarmind infer task2 --checkpoint artifacts/task2/qwen --split dev
```

Submission zip files are written next to the prediction directory.

### Curriculum learning

Score training examples by loss and export easy / medium / hard splits:

```bash
python -m polarmind curriculum --epochs 1
```

## Configuration

Default hyperparameters live in `configs/`. Override any setting via CLI flags (see `--help` on each subcommand).

## Python API

```python
from polarmind.data import ensure_extracted, get_task_paths, load_multilingual_data
from polarmind.models import LaBSEMultiLayerClassifier
from polarmind.pipelines import BinaryLabseConfig, run_train_binary_labse

paths = get_task_paths(1)
ensure_extracted(paths["raw_archive"], paths["processed"])
checkpoint = run_train_binary_labse(BinaryLabseConfig(epochs=1))
```

## Models

| Model | Task | Method |
|-------|------|--------|
| LaBSE multi-layer | 1 | Weighted pooling over last N encoder layers |
| LaBSE + attention | 1 | Per-layer attention pooling with language tokens |
| Qwen 2.5 + LoRA | 1 | Chat-template SFT, binary output |
| Qwen 2.5 + LoRA | 2 | Custom token-level loss at category delimiters |
| XLM-RoBERTa | curriculum | Probe model for difficulty scoring |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face model download / upload |
| `WANDB_API_KEY` | Optional experiment logging |

## License

Add your project license here.
