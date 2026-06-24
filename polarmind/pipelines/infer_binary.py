from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from polarmind.constants import TASK1_LANGUAGES
from polarmind.data import ensure_extracted, get_task_paths
from polarmind.inference.binary import predict_encoder, predict_qwen
from polarmind.models import LaBSEMultiLayerClassifier


@dataclass
class InferBinaryConfig:
    backend: str = "labse"
    checkpoint: Path = Path("artifacts/task1/labse/model.pt")
    encoder_model_name: str = "setu4993/LaBSE"
    qwen_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    split: str = "test"
    output_dir: Path = Path("artifacts/task1/predictions")
    languages: list[str] = field(default_factory=lambda: list(TASK1_LANGUAGES))


def _load_labse(checkpoint: Path, model_name: str, device: torch.device):
    payload = torch.load(checkpoint, map_location=device)
    model = LaBSEMultiLayerClassifier(
        model_name, num_layers_to_aggregate=payload.get("num_layers", 6)
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.parent / "tokenizer", trust_remote_code=True)
    return model, tokenizer


def _load_qwen(adapter: Path, model_id: str):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config, device_map="auto", trust_remote_code=True
    )
    return PeftModel.from_pretrained(base, adapter), tokenizer


def run(config: InferBinaryConfig) -> Path:
    paths = get_task_paths(1)
    ensure_extracted(paths["raw_archive"], paths["processed"])

    split_dir = paths["dev_dir"] if config.split == "dev" else paths["test_dir"]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.backend == "labse":
        model, tokenizer = _load_labse(config.checkpoint, config.encoder_model_name, device)
        predict = lambda src, dst: predict_encoder(model, tokenizer, src, dst, device=device)
    else:
        model, tokenizer = _load_qwen(config.checkpoint, config.qwen_model_id)
        predict = lambda src, dst: predict_qwen(model, tokenizer, src, dst, device=device)

    for lang in config.languages:
        input_csv = split_dir / f"{lang}.csv"
        if not input_csv.exists():
            print(f"Skipping missing file: {input_csv}")
            continue
        predict(input_csv, output_dir / f"pred_{lang}.csv")

    archive = output_dir.parent / f"task1_{config.split}_submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_file in sorted(output_dir.glob("pred_*.csv")):
            zf.write(csv_file, arcname=csv_file.name)
    return archive
