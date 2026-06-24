"""Prompt templates for multi-label polarization classification (task 2)."""

from polarmind.constants import MULTILABEL_CATEGORIES


def format_multilabel_instruction(sample: dict, delimiter: str = " ::") -> dict:
    lines = "\n".join(f"{name} polarisation (1/0){delimiter} \n" for name in _display_names())
    prompt = (
        "<|system|>\n"
        "You are a classifier. Output 1 or 0 for each label if it is that type of polarisation.\n"
        "<|user|>\n"
        f"Text: {sample['text']}\n"
        "<|assistant|>\n"
        "LABELS ARE GIVEN BELOW\n"
        f"{lines}"
    )
    return {"text": prompt}


def format_multilabel_direct(sample: dict) -> dict:
    prompt = (
        "<|system|>\n"
        "You are a strict classifier. For the categories below, answer only with a comma-separated list of YES or NO.\n"
        "<|user|>\n"
        f"Text: {sample['text']}\n"
        "<|assistant|>\n"
        "Labels (political, racial, religious, gender, other): "
    )
    return {"text": prompt}


def _display_names() -> list[str]:
    mapping = {
        "political": "political",
        "racial/ethnic": "racial",
        "religious": "religious",
        "gender/sexual": "gender/sexual",
        "other": "other",
    }
    return [mapping[cat] for cat in MULTILABEL_CATEGORIES]
