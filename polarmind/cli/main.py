from __future__ import annotations

import argparse
from pathlib import Path

from polarmind.pipelines import (
    BinaryLabseConfig,
    BinaryQwenConfig,
    CurriculumConfig,
    InferBinaryConfig,
    InferMultilabelConfig,
    MultilabelQwenConfig,
    run_curriculum,
    run_infer_binary,
    run_infer_multilabel,
    run_train_binary_labse,
    run_train_binary_qwen,
    run_train_multilabel_qwen,
)


def _add_train_parser(subparsers) -> None:
    train = subparsers.add_parser("train", help="Train a model")
    train_sub = train.add_subparsers(dest="task", required=True)

    t1 = train_sub.add_parser("task1", help="Binary polarization (task 1)")
    t1.add_argument("--model", choices=["labse", "qwen"], default="labse")
    t1.add_argument("--output-dir", type=Path)
    t1.add_argument("--epochs", type=int)
    t1.add_argument("--batch-size", type=int)
    t1.add_argument("--model-name", help="Encoder or Qwen model ID")

    t2 = train_sub.add_parser("task2", help="Multi-label polarization (task 2)")
    t2.add_argument("--output-dir", type=Path)
    t2.add_argument("--delimiter", choices=[" ::", ")%"], default=" ::")
    t2.add_argument("--epochs", type=int)
    t2.add_argument("--batch-size", type=int)
    t2.add_argument("--model-name", dest="model_id")


def _add_infer_parser(subparsers) -> None:
    infer = subparsers.add_parser("infer", help="Run inference and package submissions")
    infer_sub = infer.add_subparsers(dest="task", required=True)

    t1 = infer_sub.add_parser("task1", help="Binary predictions")
    t1.add_argument("--model", choices=["labse", "qwen"], default="labse")
    t1.add_argument("--checkpoint", type=Path, required=True)
    t1.add_argument("--split", choices=["dev", "test"], default="test")
    t1.add_argument("--output-dir", type=Path)
    t1.add_argument("--model-name")

    t2 = infer_sub.add_parser("task2", help="Multi-label predictions")
    t2.add_argument("--checkpoint", type=Path, required=True)
    t2.add_argument("--split", choices=["dev", "test"], default="dev")
    t2.add_argument("--delimiter", choices=[" ::", ")%"], default=" ::")
    t2.add_argument("--output-dir", type=Path)
    t2.add_argument("--model-id")


def _add_curriculum_parser(subparsers) -> None:
    curriculum = subparsers.add_parser("curriculum", help="Build curriculum learning splits")
    curriculum.add_argument("--epochs", type=int, default=1)
    curriculum.add_argument("--batch-size", type=int, default=16)
    curriculum.add_argument("--output-dir", type=Path)
    curriculum.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")


def _run_train(args) -> None:
    if args.task == "task1":
        if args.model == "labse":
            config = BinaryLabseConfig()
            if args.output_dir:
                config.output_dir = args.output_dir
            if args.epochs is not None:
                config.epochs = args.epochs
            if args.batch_size is not None:
                config.batch_size = args.batch_size
            if args.model_name:
                config.model_name = args.model_name
            path = run_train_binary_labse(config)
        else:
            config = BinaryQwenConfig()
            if args.output_dir:
                config.output_dir = args.output_dir
            if args.epochs is not None:
                config.epochs = args.epochs
            if args.batch_size is not None:
                config.batch_size = args.batch_size
            if args.model_name:
                config.model_id = args.model_name
            path = run_train_binary_qwen(config)
    else:
        config = MultilabelQwenConfig()
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.delimiter:
            config.delimiter = args.delimiter
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.model_id:
            config.model_id = args.model_id
        path = run_train_multilabel_qwen(config)
    print(f"Saved model to {path}")


def _run_infer(args) -> None:
    if args.task == "task1":
        config = InferBinaryConfig(checkpoint=args.checkpoint, split=args.split, backend=args.model)
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.model_name:
            if args.model == "labse":
                config.encoder_model_name = args.model_name
            else:
                config.qwen_model_id = args.model_name
        archive = run_infer_binary(config)
    else:
        config = InferMultilabelConfig(checkpoint=args.checkpoint, split=args.split, delimiter=args.delimiter)
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.model_id:
            config.model_id = args.model_id
        archive = run_infer_multilabel(config)
    print(f"Submission archive: {archive}")


def _run_curriculum(args) -> None:
    config = CurriculumConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    archive = run_curriculum(config)
    print(f"Curriculum archive: {archive}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polarmind",
        description="Multilingual polarization detection: training, inference, and curriculum tooling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers)
    _add_infer_parser(subparsers)
    _add_curriculum_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        _run_train(args)
    elif args.command == "infer":
        _run_infer(args)
    elif args.command == "curriculum":
        _run_curriculum(args)


if __name__ == "__main__":
    main()
