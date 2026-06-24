from __future__ import annotations

import glob
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from polarmind.data import (
    ensure_extracted,
    get_csv_patterns,
    get_difficulty_scores,
    get_task_paths,
    load_multilingual_data,
    split_curriculum_buckets,
)
from polarmind.models import XLMRobertaMultiLayerClassifier
from polarmind.training.encoder import EncoderTrainConfig, train_encoder_classifier


@dataclass
class CurriculumConfig:
    model_name: str = "FacebookAI/xlm-roberta-base"
    epochs: int = 1
    batch_size: int = 16
    max_length: int = 256
    output_dir: Path | None = None


def run(config: CurriculumConfig) -> Path:
    paths = get_task_paths(1)
    ensure_extracted(paths["raw_archive"], paths["processed"])
    output_dir = Path(config.output_dir or paths["curriculum_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pattern, dev_pattern = get_csv_patterns(paths)
    train_dataset, test_dataset = load_multilingual_data(train_pattern, dev_pattern)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.max_length)

    cols = [c for c in train_dataset.column_names if c != "polarization"]
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=cols)
    test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=cols)
    train_dataset = train_dataset.rename_column("polarization", "labels")
    test_dataset = test_dataset.rename_column("polarization", "labels")
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XLMRobertaMultiLayerClassifier(config.model_name).to(device)
    train_encoder_classifier(
        model, train_loader, test_loader, device,
        EncoderTrainConfig(epochs=config.epochs, batch_size=config.batch_size),
    )

    frames = [pd.read_csv(path) for path in sorted(glob.glob(train_pattern, recursive=True))]
    full_df = pd.concat(frames, ignore_index=True)

    scoring = Dataset.from_pandas(full_df)
    scoring = scoring.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in scoring.column_names if c not in ["id", "text", "polarization"]],
    )
    scoring = scoring.rename_column("polarization", "labels")
    scoring.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    scoring_loader = DataLoader(scoring, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

    full_df["difficulty_score"] = get_difficulty_scores(model, scoring_loader, device)
    easy, medium, hard = split_curriculum_buckets(full_df)

    archive = output_dir / "curriculum_splits.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, frame in {"easy.csv": easy, "medium.csv": medium, "hard.csv": hard}.items():
            temp = output_dir / name
            frame[["id", "text", "polarization", "difficulty_score"]].to_csv(temp, index=False)
            zf.write(temp, arcname=name)
    return archive
