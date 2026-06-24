from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import torch

from polarmind.constants import TASK2_LANGUAGES
from polarmind.data import ensure_extracted, get_task_paths
from polarmind.inference.multilabel import load_qwen_adapter, predict_multilabel


@dataclass
class InferMultilabelConfig:
    checkpoint: Path = Path("artifacts/task2/qwen")
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    split: str = "dev"
    delimiter: str = " ::"
    output_dir: Path = Path("artifacts/task2/predictions")
    languages: list[str] = field(default_factory=lambda: list(TASK2_LANGUAGES))


def run(config: InferMultilabelConfig) -> Path:
    paths = get_task_paths(2)
    ensure_extracted(paths["raw_archive"], paths["processed"])

    split_dir = paths["dev_dir"] if config.split == "dev" else paths["test_dir"]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_qwen_adapter(config.checkpoint, config.model_id)
    for lang in config.languages:
        input_csv = split_dir / f"{lang}.csv"
        if not input_csv.exists():
            print(f"Skipping missing file: {input_csv}")
            continue
        predict_multilabel(
            model, tokenizer, input_csv, output_dir / f"{lang}.csv", config.delimiter, device
        )

    archive = output_dir.parent / f"task2_{config.split}_submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_file in sorted(output_dir.glob("*.csv")):
            zf.write(csv_file, arcname=csv_file.name)
    return archive
