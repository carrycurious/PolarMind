"""Shared task and label definitions."""

# Task 1: binary polarization detection (22 languages)
TASK1_LANGUAGES = [
    "amh", "arb", "ben", "mya", "eng", "deu", "hau", "hin", "ita", "khm",
    "nep", "ori", "fas", "pol", "pan", "rus", "spa", "swa", "tel", "tur", "urd", "zho",
]

# Task 2: fine-grained multi-label (7 languages)
TASK2_LANGUAGES = ["eng", "deu", "ita", "spa", "pol", "rus", "tur"]

MULTILABEL_CATEGORIES = [
    "political",
    "racial/ethnic",
    "religious",
    "gender/sexual",
    "other",
]

BINARY_LABEL_COLUMN = "polarization"
