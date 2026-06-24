from __future__ import annotations

import glob
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def ensure_extracted(zip_path: Path, extract_to: Path) -> Path:
    """Extract a zip archive when the target directory is empty."""
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    if not any(extract_to.iterdir()):
        if not zip_path.exists():
            raise FileNotFoundError(
                f"Dataset zip not found: {zip_path}. "
                "Place the competition archive under data/taskN/raw/."
            )
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_to)

    return extract_to


def load_multilingual_data(train_pattern: str, dev_pattern: str) -> tuple[Dataset, Dataset]:
    """Load and concatenate multilingual CSV files from glob patterns."""
    train_files = sorted(glob.glob(train_pattern, recursive=True))
    dev_files = sorted(glob.glob(dev_pattern, recursive=True))

    if not train_files:
        raise FileNotFoundError(f"No train CSV files matched: {train_pattern}")
    if not dev_files:
        raise FileNotFoundError(f"No dev CSV files matched: {dev_pattern}")

    print(f"Found {len(train_files)} train files")
    print(f"Found {len(dev_files)} dev files")

    train_datasets = [Dataset.from_pandas(pd.read_csv(path)) for path in train_files]
    dev_datasets = [Dataset.from_pandas(pd.read_csv(path)) for path in dev_files]

    full_train = concatenate_datasets(train_datasets).shuffle(seed=42)
    full_dev = concatenate_datasets(dev_datasets).shuffle(seed=42)
    return full_train, full_dev


def load_language_csvs(
    data_dir: Path,
    languages: Iterable[str],
    split_name: str = "train",
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset, dict[str, Dataset]]:
    """Load per-language CSVs, concatenate, and keep per-language eval splits."""
    train_parts: list[Dataset] = []
    eval_parts: list[Dataset] = []
    eval_by_lang: dict[str, Dataset] = {}

    for lang in languages:
        path = data_dir / f"{lang}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing language file: {path}")

        split = Dataset.from_pandas(pd.read_csv(path)).train_test_split(
            test_size=test_size, seed=seed
        )
        train_parts.append(split["train"])
        eval_parts.append(split["test"])
        eval_by_lang[lang] = split["test"]

    return (
        concatenate_datasets(train_parts),
        concatenate_datasets(eval_parts),
        eval_by_lang,
    )


def load_csv_split(csv_path: Path, test_size: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    split = Dataset.from_pandas(pd.read_csv(csv_path)).train_test_split(
        test_size=test_size, seed=seed
    )
    return split["train"], split["test"]


def get_difficulty_scores(model, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """Per-example cross-entropy loss used as curriculum difficulty score."""
    model.eval()
    scores: list[float] = []
    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scoring difficulty"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            if isinstance(logits, tuple):
                logits = logits[1] if len(logits) > 1 else logits[0]

            loss = criterion(logits, labels)
            scores.extend(loss.cpu().numpy().tolist())

    return np.array(scores)


def split_curriculum_buckets(df: pd.DataFrame, score_col: str = "difficulty_score"):
    """Split a dataframe into equal easy / medium / hard thirds by score."""
    ordered = df.sort_values(by=score_col).reset_index(drop=True)
    n = len(ordered)
    third = n // 3
    return (
        ordered.iloc[:third],
        ordered.iloc[third : 2 * third],
        ordered.iloc[2 * third :],
    )
