"""Public command line; runs locally on the machine invoking it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run self-contained cipher experiments locally.")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "train"):
        item = sub.add_parser(command, help=("Validate config/device without training" if command
                                           == "check" else "Train on this machine"))
        item.add_argument("--config", type=Path, required=True)
        if command == "train":
            item.add_argument("--output", type=Path, required=True, help="New run directory")
    item = sub.add_parser("evaluate", help="Evaluate a SHA-bound checkpoint on saved test recipe")
    item.add_argument("--checkpoint", type=Path, required=True)
    item.add_argument("--sha256", required=True)
    item.add_argument("--device", default="cpu", choices=("cpu", "cuda:0", "cuda:1"))
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--experiment", default="prediction", choices=("prediction", "innovation1"))
    args = parser.parse_args(argv)
    try:
        from .artifacts import write_json
        from .training import datasets, device_for, train
        if args.command in ("check", "train"):
            config = load_config(args.config)
            device_for(config.device)
            if args.command == "check":
                print(json.dumps({"status": "valid", "config": config.to_dict(),
                                  "config_sha256": config.sha256()}, indent=2))
            else:
                if args.output.exists():
                    raise FileExistsError(f"output already exists: {args.output}")
                if config.experiment == "innovation1":
                    from .innovation1 import train as train_shared_m
                    result = train_shared_m(config, args.output)
                else:
                    result = train(config, args.output)
                print(json.dumps(result, indent=2))
        else:
            if args.experiment == "innovation1":
                from .innovation1 import evaluate_checkpoint
                result = evaluate_checkpoint(args.checkpoint, args.sha256, args.device, args.output)
                print(json.dumps(result, indent=2))
                return 0
            from .evaluation import evaluate_model, load_checkpoint
            device = device_for(args.device)
            model, config, majority, probability = load_checkpoint(
                args.checkpoint, args.sha256, device=device)
            args.output.mkdir(parents=True, exist_ok=False)
            metrics = evaluate_model(model, *datasets(config)["test"], device=device,
                                     batch_size=config.batch_size,
                                     majority=majority, probability=probability)
            result = {"checkpoint_sha256": args.sha256, "split": "test",
                      "config_sha256": config.sha256(), "metrics": metrics,
                      "note": "explicit evaluation regenerates configured test data"}
            write_json(args.output / "evaluation.json", result)
            print(json.dumps(result, indent=2))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
