from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from polarmind.constants import MULTILABEL_CATEGORIES
from polarmind.prompts.multilabel import format_multilabel_instruction

DEFAULT_THRESHOLDS = {
    "political": 0.4,
    "racial/ethnic": 0.5,
    "religious": 0.5,
    "gender/sexual": 0.4,
    "other": 0.4,
}


def load_qwen_adapter(checkpoint: Path, model_id: str):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_multilabel(
    model,
    tokenizer,
    csv_path: Path,
    output_path: Path,
    delimiter: str = " ::",
    device: torch.device | None = None,
) -> pd.DataFrame:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda sample: format_multilabel_instruction(sample, delimiter=delimiter))
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512),
        batched=True,
    )
    dataset = dataset.remove_columns(["text"])

    id_0 = tokenizer.encode("0", add_special_tokens=False)[-1]
    id_1 = tokenizer.encode("1", add_special_tokens=False)[-1]
    delimiter_id = tokenizer.encode(delimiter, add_special_tokens=False)[-1]

    predictions = {cat: [] for cat in MULTILABEL_CATEGORIES}
    for row in dataset:
        input_ids = torch.tensor([row["input_ids"]]).to(device)
        attention_mask = torch.tensor([row["attention_mask"]]).to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        delimiter_indices = (input_ids[0] == delimiter_id).nonzero(as_tuple=True)[0][-5:]
        if len(delimiter_indices) != 5:
            for cat in MULTILABEL_CATEGORIES:
                predictions[cat].append(0)
            continue

        for idx, position in enumerate(delimiter_indices):
            category = MULTILABEL_CATEGORIES[idx]
            rel_logits = logits[0, position, [id_0, id_1]]
            prob_1 = F.softmax(rel_logits, dim=0)[1].item()
            predictions[category].append(int(prob_1 > DEFAULT_THRESHOLDS[category]))

    output = df[["id"]].copy()
    for category in MULTILABEL_CATEGORIES:
        output[category] = predictions[category]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output
