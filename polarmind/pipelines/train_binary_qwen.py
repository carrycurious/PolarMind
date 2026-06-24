from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from polarmind.constants import TASK1_LANGUAGES
from polarmind.data import ensure_extracted, get_task_paths
from polarmind.prompts.binary import QWEN_CHAT_TEMPLATE, format_binary_messages


@dataclass
class BinaryQwenConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: Path = Path("artifacts/task1/qwen")
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 8
    learning_rate: float = 5e-5
    max_length: int = 128
    test_size: float = 0.02
    languages: list[str] = field(default_factory=lambda: list(TASK1_LANGUAGES))


def _load_messages(train_dir: Path, languages: list[str], test_size: float, max_words: int = 400):
    frames = []
    for lang in languages:
        path = train_dir / f"{lang}.csv"
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        df = pd.read_csv(path)
        df["lang"] = lang
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No language CSV files found in {train_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["text"].str.split().str.len() < max_words]
    split = Dataset.from_pandas(combined).train_test_split(test_size=test_size, seed=42, shuffle=True)
    train = split["train"].map(format_binary_messages, remove_columns=split["train"].column_names)
    eval_set = split["test"].map(format_binary_messages, remove_columns=split["test"].column_names)
    return train, eval_set


def run(config: BinaryQwenConfig) -> Path:
    paths = get_task_paths(1)
    ensure_extracted(paths["raw_archive"], paths["processed"])
    train_dataset, eval_dataset = _load_messages(paths["train_dir"], config.languages, config.test_size)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        optim="paged_adamw_32bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        assistant_only_loss=True,
        max_length=config.max_length,
        seed=42,
        data_seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(config.output_dir)
    return Path(config.output_dir)
