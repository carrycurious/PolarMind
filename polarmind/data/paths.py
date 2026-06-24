from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_split_dir(extracted: Path, split: str) -> Path:
    """Locate train/dev/test whether the archive extracts flat or nested."""
    extracted = Path(extracted)
    direct = extracted / split
    if direct.is_dir() and any(direct.glob("*.csv")):
        return direct

    for candidate in sorted(p for p in extracted.glob(f"**/{split}") if p.is_dir()):
        if any(candidate.glob("*.csv")):
            return candidate

    return direct


def get_task_paths(task: int) -> dict[str, Path]:
    """Return canonical paths for task 1 (binary) or task 2 (multi-label)."""
    if task not in (1, 2):
        raise ValueError("task must be 1 or 2")

    root = project_root() / "data" / f"task{task}"
    extracted = root / "processed"
    return {
        "root": root,
        "raw_archive": root / "raw" / f"task{task}.zip",
        "processed": extracted,
        "train_dir": resolve_split_dir(extracted, "train"),
        "dev_dir": resolve_split_dir(extracted, "dev"),
        "test_dir": resolve_split_dir(extracted, "test"),
        "curriculum_dir": root / "curriculum",
    }


def get_csv_patterns(paths: dict[str, Path]) -> tuple[str, str]:
    """Return glob patterns for train and dev CSV files."""
    train_dir = paths["train_dir"]
    dev_dir = paths["dev_dir"]
    if train_dir.exists() and dev_dir.exists():
        return str(train_dir / "*.csv"), str(dev_dir / "*.csv")

    processed = paths["processed"]
    return (
        str(processed / "**" / "train" / "*.csv"),
        str(processed / "**" / "dev" / "*.csv"),
    )
