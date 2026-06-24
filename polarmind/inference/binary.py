from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding


@torch.no_grad()
def predict_encoder(
    model,
    tokenizer,
    csv_path: Path,
    output_path: Path,
    batch_size: int = 32,
    max_length: int = 128,
    device: torch.device | None = None,
) -> pd.DataFrame:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )
    dataset.set_format("torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))

    predictions: list[int] = []
    for batch in tqdm(loader, desc=f"Inference {csv_path.name}"):
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        if isinstance(logits, tuple):
            logits = logits[1] if len(logits) > 1 else logits[0]
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    output = df[["id"]].copy()
    output["polarization"] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


@torch.no_grad()
def predict_qwen(
    model,
    tokenizer,
    csv_path: Path,
    output_path: Path,
    device: torch.device | None = None,
) -> pd.DataFrame:
    from polarmind.prompts.binary import SYSTEM_BINARY

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = pd.read_csv(csv_path)
    results = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Inference {csv_path.name}"):
        messages = [
            {"role": "system", "content": SYSTEM_BINARY},
            {"role": "user", "content": row.text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
        label = generated[0] if generated and generated[0] in {"0", "1"} else "0"
        results.append({"id": row.id, "polarization": label})

    output = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output
