from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from polarmind.data import ensure_extracted, get_csv_patterns, get_task_paths, load_multilingual_data
from polarmind.models import LaBSEMultiLayerClassifier
from polarmind.training.encoder import EncoderTrainConfig, prepare_tokenized_datasets, train_encoder_classifier


@dataclass
class BinaryLabseConfig:
    model_name: str = "setu4993/LaBSE"
    epochs: int = 3
    batch_size: int = 32
    max_length: int = 128
    num_layers: int = 6
    freeze_layers: int = 2
    output_dir: Path = Path("artifacts/task1/labse")


def run(config: BinaryLabseConfig) -> Path:
    paths = get_task_paths(1)
    ensure_extracted(paths["raw_archive"], paths["processed"])

    train_pattern, dev_pattern = get_csv_patterns(paths)
    train_dataset, eval_dataset = load_multilingual_data(train_pattern, dev_pattern)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    train_dataset, eval_dataset, collator = prepare_tokenized_datasets(
        train_dataset, eval_dataset, tokenizer, config.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, collate_fn=collator)

    model = LaBSEMultiLayerClassifier(
        config.model_name,
        num_labels=2,
        num_layers_to_aggregate=config.num_layers,
    ).to(device)

    train_encoder_classifier(
        model,
        train_loader,
        eval_loader,
        device,
        EncoderTrainConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            freeze_embedding_layers=config.freeze_layers,
        ),
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "model_name": config.model_name, "num_layers": config.num_layers},
        checkpoint,
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    return checkpoint
