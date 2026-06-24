from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding, get_cosine_schedule_with_warmup


@dataclass
class EncoderTrainConfig:
    epochs: int = 3
    batch_size: int = 32
    encoder_lr: float = 1e-5
    head_lr: float = 1e-4
    layer_weight_lr: float = 1e-3
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    freeze_embedding_layers: int = 0


def build_optimizer_groups(model: nn.Module, config: EncoderTrainConfig) -> list[dict]:
    groups: list[dict] = [
        {"params": model.encoder.parameters(), "lr": config.encoder_lr, "weight_decay": 0.01},
        {"params": model.classifier.parameters(), "lr": config.head_lr, "weight_decay": 0.01},
    ]
    if hasattr(model, "layer_weights"):
        groups.append({"params": [model.layer_weights], "lr": config.layer_weight_lr, "weight_decay": 0.0})
    return groups


def freeze_encoder_layers(model: nn.Module, num_layers: int) -> None:
    if num_layers <= 0:
        return

    if hasattr(model.encoder, "embeddings"):
        for param in model.encoder.embeddings.parameters():
            param.requires_grad = False

    if hasattr(model.encoder, "encoder") and hasattr(model.encoder.encoder, "layer"):
        for layer_idx in range(min(num_layers, len(model.encoder.encoder.layer))):
            for param in model.encoder.encoder.layer[layer_idx].parameters():
                param.requires_grad = False


def train_encoder_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    config: EncoderTrainConfig | None = None,
    log_every: int = 100,
) -> dict[str, list[float]]:
    config = config or EncoderTrainConfig()
    freeze_encoder_layers(model, config.freeze_embedding_layers)

    optimizer = torch.optim.AdamW(build_optimizer_groups(model, config))
    num_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * num_steps),
        num_training_steps=num_steps,
    )
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]")):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            if isinstance(logits, tuple):
                logits = logits[1] if len(logits) > 1 else logits[0]
            loss = criterion(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            if log_every and (batch_idx + 1) % log_every == 0:
                print(f"  step {batch_idx + 1}: loss={loss.item():.4f}")

        val_loss, val_f1, val_acc = evaluate_encoder(model, eval_loader, device, criterion)
        avg_train = running_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch + 1}: train_loss={avg_train:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f}"
        )

    return history


@torch.no_grad()
def evaluate_encoder(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float, float, float]:
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    losses: list[float] = []
    preds: list[int] = []
    labels: list[int] = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        if isinstance(logits, tuple):
            logits = logits[1] if len(logits) > 1 else logits[0]
        losses.append(criterion(logits, batch["labels"]).item())
        prediction = torch.argmax(logits, dim=1)
        preds.extend(prediction.cpu().numpy().tolist())
        labels.extend(batch["labels"].cpu().numpy().tolist())

    val_loss = sum(losses) / max(len(losses), 1)
    val_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    val_acc = accuracy_score(labels, preds)
    return val_loss, val_f1, val_acc


def prepare_tokenized_datasets(
    train_dataset,
    eval_dataset,
    tokenizer,
    max_length: int,
    label_col: str = "polarization",
):
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    cols_to_remove = [c for c in train_dataset.column_names if c not in [label_col]]
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    train_dataset = train_dataset.rename_column(label_col, "labels")
    eval_dataset = eval_dataset.rename_column(label_col, "labels")
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    collator = DataCollatorWithPadding(tokenizer)
    return train_dataset, eval_dataset, collator
