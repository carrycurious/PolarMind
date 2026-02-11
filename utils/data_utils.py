from pathlib import Path
import glob
import zipfile
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def ensure_extracted(zip_path: Path, extract_to: Path) -> Path:
    """
    Ensure that a zip archive is extracted under the given directory.

    The directory is created if needed and extraction is only performed
    if the directory is currently empty. Returns the extraction directory.
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    # Only extract if directory appears empty (no files or subdirs)
    if not any(extract_to.iterdir()):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

    return extract_to


def load_multilingual_data(train_pattern: str, dev_pattern: str):
    """
    Load multilingual train and dev CSV files given glob patterns.

    Returns:
        full_train, full_dev: shuffled Hugging Face Datasets.
    """
    train_files = glob.glob(train_pattern, recursive=True)
    dev_files = glob.glob(dev_pattern, recursive=True)

    print(f"Found {len(train_files)} train language files: {train_files}")
    print(f"Found {len(dev_files)} dev language files: {dev_files}")

    train_datasets = [Dataset.from_pandas(pd.read_csv(file)) for file in train_files]
    dev_datasets = [Dataset.from_pandas(pd.read_csv(file)) for file in dev_files]

    full_train = concatenate_datasets(train_datasets).shuffle(seed=42)
    full_dev = concatenate_datasets(dev_datasets).shuffle(seed=42)

    return full_train, full_dev


def get_difficulty_scores(model, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Compute per-example cross-entropy loss as a difficulty score for each sample.

    Args:
        model: Trained classification model.
        dataloader: DataLoader yielding batches with 'input_ids', 'attention_mask', 'labels'.
        device: Torch device to run evaluation on.

    Returns:
        np.ndarray of shape (num_examples,) with difficulty scores.
    """
    model.eval()
    scores = []
    criterion_no_reduce = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scoring dataset"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion_no_reduce(logits, labels)
            scores.extend(loss.cpu().numpy())

    return np.array(scores)

