# PolarMind

Multilingual polarization detection for competition-style NLP tasks.

## Tasks

| Task | Question | Languages | Output |
|------|----------|-----------|--------|
| **Task 1** | Is this text polarized? | 22 | `polarization` ∈ {0, 1} |
| **Task 2** | What kind of polarization? | 7 | five binary columns |

**Task 2 labels:** `political`, `racial/ethnic`, `religious`, `gender/sexual`, `other`

**Task 1 languages:** `amh`, `arb`, `ben`, `mya`, `eng`, `deu`, `hau`, `hin`, `ita`, `khm`, `nep`, `ori`, `fas`, `pol`, `pan`, `rus`, `spa`, `swa`, `tel`, `tur`, `urd`, `zho`  
**Task 2 languages:** `eng`, `deu`, `ita`, `spa`, `pol`, `rus`, `tur`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows (PowerShell)
# source .venv/bin/activate  # Linux/macOS

pip install -e .
```

## Data

Place the official archives here:

- `data/task1/raw/task1.zip`
- `data/task2/raw/task2.zip`

The CLI will extract them to `data/taskN/processed/` on first use.

## Usage

```bash
python -m polarmind --help
python -m polarmind train --help
python -m polarmind infer --help
python -m polarmind curriculum --help
```

### Train

```bash
# Task 1 — LaBSE encoder baseline
python -m polarmind train task1 --model labse --epochs 3 --output-dir artifacts/task1/labse

# Task 1 — Qwen QLoRA
python -m polarmind train task1 --model qwen --output-dir artifacts/task1/qwen

# Task 2 — Qwen multi-label (delimiter-token loss)
python -m polarmind train task2 --delimiter " ::" --output-dir artifacts/task2/qwen
```

### Infer

```bash
python -m polarmind infer task1 --model labse --checkpoint artifacts/task1/labse/model.pt --split test --output-dir artifacts/task1/predictions
python -m polarmind infer task1 --model qwen --checkpoint artifacts/task1/qwen --split test
python -m polarmind infer task2 --checkpoint artifacts/task2/qwen --split dev --delimiter " ::"
```

### Curriculum

```bash
python -m polarmind curriculum --epochs 1 --output-dir data/task1/curriculum
```

## Repository layout

```
PolarMind/
├── polarmind/        # Package (CLI, pipelines, training, inference)
├── configs/          # Default YAML configs
├── data/             # Raw zips + extracted CSVs
├── artifacts/        # Outputs (checkpoints, predictions, submissions)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## License

Add your project or competition license here.
