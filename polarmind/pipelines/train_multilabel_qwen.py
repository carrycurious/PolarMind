from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig

from polarmind.constants import MULTILABEL_CATEGORIES, TASK2_LANGUAGES
from polarmind.data import ensure_extracted, get_task_paths, load_language_csvs
from polarmind.prompts.multilabel import format_multilabel_direct, format_multilabel_instruction
from polarmind.training import F1EvaluationCallback, LabelPreservingCollator, PolarizationTrainer


@dataclass
class MultilabelQwenConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: Path = Path("artifacts/task2/qwen")
    languages: list[str] = field(default_factory=lambda: list(TASK2_LANGUAGES))
    delimiter: str = " ::"
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 1e-4
    max_length: int = 512
    eval_steps: int = 60


def _pos_weight(train_dataset: Dataset, device: torch.device) -> torch.Tensor:
    pos_counts, neg_counts = [], []
    for category in MULTILABEL_CATEGORIES:
        values = train_dataset[category]
        pos = sum(values)
        pos_counts.append(pos)
        neg_counts.append(len(values) - pos)
    pos = torch.tensor(pos_counts, dtype=torch.float)
    neg = torch.tensor(neg_counts, dtype=torch.float)
    return (neg / (pos + 1e-8)).to(device)


def run(config: MultilabelQwenConfig) -> Path:
    paths = get_task_paths(2)
    ensure_extracted(paths["raw_archive"], paths["processed"])
    train_raw, eval_raw, _ = load_language_csvs(paths["train_dir"], config.languages, test_size=0.05)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    lang_tokens = [f"<lang={lang}>" for lang in config.languages]
    tokenizer.add_special_tokens({"additional_special_tokens": lang_tokens})
    model.resize_token_embeddings(len(tokenizer))

    format_fn = lambda sample: format_multilabel_instruction(sample, delimiter=config.delimiter)
    train_dataset = train_raw.map(format_fn)
    eval_dataset = eval_raw.map(format_multilabel_direct)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=config.max_length)

    train_dataset = train_dataset.map(tokenize_fn, batched=True).remove_columns(["id", "text"]).shuffle(seed=42)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True).remove_columns(["id", "text"])

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=config.epochs,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        logging_steps=config.eval_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        packing=False,
        max_length=config.max_length,
    )

    trainer = PolarizationTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        args=sft_config,
        data_collator=LabelPreservingCollator(tokenizer=tokenizer, padding=True),
        callbacks=[F1EvaluationCallback(eval_dataset, tokenizer, delimiter_token=config.delimiter)],
        pos_weight=_pos_weight(train_dataset, next(model.parameters()).device),
        delimiter_token=config.delimiter,
    )
    trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(config.output_dir)
    return Path(config.output_dir)
