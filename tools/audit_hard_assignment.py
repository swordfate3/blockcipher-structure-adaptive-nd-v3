"""Post-hoc audit: are learned hard constants identifiable, or only equivalent?

Reads a finished ``simon-hard-search``/``simon-hard-sat`` run directory, compares
the stored assignment with the true round-key schedule, and writes a small JSON
report. It deliberately does not retrain, reload the checkpoint or touch labels.

    python tools/audit_hard_assignment.py --run runs/<run-id> --output audit.json
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from cipher_experiments.ciphers.simon import encrypt, round_keys
from cipher_experiments.data.prediction import key_for_seed
from cipher_experiments.fitting.ambiguity import (
    assignment_bits,
    audit_assignment,
    evaluate_cone_bit,
    schedule_bits,
)
from cipher_experiments.representations.simon_cone import simon_cone


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit hard-constant identifiability")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.probes < 1:
        raise SystemExit("--probes must be positive")

    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    fit = json.loads((args.run / "fit.json").read_text(encoding="utf-8"))
    if fit.get("assignment") is None:
        raise SystemExit("run has no fitted assignment; nothing to audit")
    if config["label_mode"] != "cipher":
        raise SystemExit("audit needs cipher labels to define a true schedule")

    rounds, target_bit = int(config["rounds"]), int(config["target_bit"])
    cone = simon_cone(rounds, target_bit)
    coordinates = cone.round_key_coordinates
    learned = assignment_bits(int(fit["assignment"]), coordinates)
    reference = schedule_bits(round_keys(key_for_seed(int(config["key_seed"])), rounds),
                              coordinates)

    generator = random.Random(int(config["data_seed"]))
    probes = tuple(generator.getrandbits(64) for _ in range(args.probes))
    witness = audit_assignment(cone, probes, learned, reference)

    # Independent check against the real cipher, not just against the schedule.
    true_key = key_for_seed(int(config["key_seed"]))
    mask = 1 << (63 - target_bit)
    learned_vs_cipher = sum(
        evaluate_cone_bit(cone, plaintext, learned)
        != int(bool(encrypt(plaintext, true_key, rounds) & mask))
        for plaintext in probes
    )
    report = {
        "run": str(args.run), "rounds": rounds, "target_bit": target_bit,
        "coordinates": len(coordinates), "probes": args.probes,
        "learned_vs_true_schedule_output_mismatches": witness.output_mismatches_vs_reference,
        "learned_vs_real_cipher_mismatches": learned_vs_cipher,
        "differing_coordinates_vs_true_schedule": [list(c) for c in witness.differing],
        "differing_set_flip_preserves_function": (
            witness.differing_set_flip_preserves_function),
        "individually_sensitive": [list(c) for c in witness.individually_sensitive],
        "individually_inert": [list(c) for c in witness.individually_inert],
        "identifiable": not witness.differing,
        "note": ("Prediction can be exact while constants are not unique; this report "
                 "does not establish key recovery."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**report, "witness": asdict(witness)}, indent=2)
                           + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
