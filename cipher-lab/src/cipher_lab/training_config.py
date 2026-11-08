"""Pure public training/task/model specifications; no files, secrets or execution.

Models may consume these typed specifications without reading experiment JSON.
TASK_FAMILIES is the sole ordered declaration of supported public cipher families;
rounds and differences belong to each recipe, not to a training registry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast


def _integer(value: object, name: str, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: expected integer in supported range")


def _object(value: object, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name}: missing or unknown fields")
    return cast("dict[str, object]", value)


@dataclass(frozen=True, slots=True)
class TaskFamily:
    cipher: str
    max_rounds: int
    key_bits: int
    block_bits: int = 64


TASK_FAMILIES = (
    TaskFamily("SIMECK", 44, 128), TaskFamily("PRESENT", 31, 80),
    TaskFamily("SIMON", 44, 128), TaskFamily("SKINNY", 32, 64),
)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    cipher: str
    rounds: int
    difference: int

    def __post_init__(self) -> None:
        family = next((item for item in TASK_FAMILIES if item.cipher == self.cipher), None)
        if family is None:
            raise ValueError("cipher: unsupported family")
        _integer(self.rounds, "rounds", 1, family.max_rounds)
        _integer(self.difference, "difference", 1, 2**64 - 1)

    def to_dict(self) -> dict[str, object]:
        return {"cipher": self.cipher, "rounds": self.rounds, "difference": self.difference}


@dataclass(frozen=True, slots=True)
class TrainSpec:
    train: int
    dev: int
    test: int
    batch_size: int
    epochs: int
    learning_rate: float
    init_seed: int
    sample_seed: int
    shuffle_seed: int

    def __post_init__(self) -> None:
        for name in ("train", "dev", "test", "batch_size"):
            _integer(getattr(self, name), name, 1, 2000000)
        if self.train % self.batch_size:
            raise ValueError("train: must be divisible by batch_size")
        _integer(self.epochs, "epochs", 1, 100000)
        lr = self.learning_rate
        if type(lr) not in (int, float) or not 0 < lr <= 1 or not math.isfinite(lr):
            raise ValueError("learning_rate: expected finite number in (0, 1]")
        for name in ("init_seed", "sample_seed", "shuffle_seed"):
            _integer(getattr(self, name), name, 0, 2**32 - 1)

    def to_dict(self) -> dict[str, object]:
        return {"train": self.train, "dev": self.dev, "test": self.test,
                "batch_size": self.batch_size, "epochs": self.epochs,
                "learning_rate": self.learning_rate, "init_seed": self.init_seed,
                "sample_seed": self.sample_seed, "shuffle_seed": self.shuffle_seed}


@dataclass(frozen=True, slots=True)
class MLPSpec:
    hidden_width: int

    def __post_init__(self) -> None:
        _integer(self.hidden_width, "hidden_width", 1, 4096)

    def to_dict(self) -> dict[str, object]:
        return {"hidden_width": self.hidden_width}


@dataclass(frozen=True, slots=True)
class SharedMSpec:
    stem_width: int
    channels: int
    norm_groups: int
    suffix_kernels: tuple[int, ...]
    head_hidden: int
    router_hidden: int

    def __post_init__(self) -> None:
        for name in ("stem_width", "channels", "norm_groups", "head_hidden", "router_hidden"):
            _integer(getattr(self, name), name, 1, 1024)
        if self.stem_width % self.norm_groups or self.channels % self.norm_groups:
            raise ValueError("norm_groups: must divide stem_width and channels")
        if type(self.suffix_kernels) is not tuple or not 1 <= len(self.suffix_kernels) <= 16:
            raise ValueError("suffix_kernels: expected tuple of 1..16 kernels")
        for kernel in self.suffix_kernels:
            _integer(kernel, "suffix kernel", 1, 31)
            if kernel % 2 != 1:
                raise ValueError("suffix kernel: expected odd integer")

    def to_dict(self) -> dict[str, object]:
        return {"stem_width": self.stem_width, "channels": self.channels,
                "norm_groups": self.norm_groups, "suffix_kernels": list(self.suffix_kernels),
                "head_hidden": self.head_hidden, "router_hidden": self.router_hidden}


@dataclass(frozen=True, slots=True)
class PredictionRecipe:
    schema: int
    method: str
    model: MLPSpec
    training: TrainSpec
    rounds: int
    target_bit: int
    label_mode: str

    def __post_init__(self) -> None:
        _integer(self.schema, "schema", 1, 1)
        if self.method != "bit-prediction-v1":
            raise ValueError("method: unsupported prediction method")
        if not isinstance(self.model, MLPSpec) or not isinstance(self.training, TrainSpec):
            raise ValueError("model/training: expected typed prediction specifications")
        _integer(self.rounds, "rounds", 1, 44)
        _integer(self.target_bit, "target_bit", 0, 63)
        if self.label_mode not in ("cipher", "null"):
            raise ValueError("label_mode: expected cipher or null")

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "method": self.method, "model": self.model.to_dict(),
                "training": self.training.to_dict(), "rounds": self.rounds,
                "target_bit": self.target_bit, "label_mode": self.label_mode}


@dataclass(frozen=True, slots=True)
class SharedMRecipe:
    schema: int
    method: str
    model: SharedMSpec
    training: TrainSpec
    tasks: tuple[TaskSpec, ...]

    def __post_init__(self) -> None:
        _integer(self.schema, "schema", 1, 1)
        if self.method != "shared-m-four-task-v1":
            raise ValueError("method: unsupported shared-M method")
        if not isinstance(self.model, SharedMSpec) or not isinstance(self.training, TrainSpec):
            raise ValueError("model/training: expected typed shared-M specifications")
        if (type(self.tasks) is not tuple or len(self.tasks) != len(TASK_FAMILIES)
                or any(not isinstance(task, TaskSpec) for task in self.tasks)
                or tuple(task.cipher for task in self.tasks)
                != tuple(family.cipher for family in TASK_FAMILIES)):
            raise ValueError("tasks: expected the four supported families in fixed order")
        for name in ("train", "dev", "test"):
            rows = getattr(self.training, name)
            if not 2 <= rows <= 65536 or rows % 2:
                raise ValueError(f"{name}: shared-M requires 2..65536 even groups")

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "method": self.method, "model": self.model.to_dict(),
                "training": self.training.to_dict(),
                "tasks": [task.to_dict() for task in self.tasks]}


def _training(value: object) -> TrainSpec:
    node = _object(value, {"train", "dev", "test", "batch_size", "epochs", "learning_rate",
                           "init_seed", "sample_seed", "shuffle_seed"}, "training")
    return TrainSpec(
        train=cast(int, node["train"]), dev=cast(int, node["dev"]), test=cast(int, node["test"]),
        batch_size=cast(int, node["batch_size"]), epochs=cast(int, node["epochs"]),
        learning_rate=cast(float, node["learning_rate"]), init_seed=cast(int, node["init_seed"]),
        sample_seed=cast(int, node["sample_seed"]), shuffle_seed=cast(int, node["shuffle_seed"]),
    )


def parse_prediction(value: object) -> PredictionRecipe:
    """Validate an already-decoded public object; constructors enforce value types."""
    node = _object(value, {"schema", "method", "model", "training", "rounds", "target_bit",
                           "label_mode"}, "prediction recipe")
    model = _object(node["model"], {"hidden_width"}, "MLP model")
    return PredictionRecipe(
        schema=cast(int, node["schema"]), method=cast(str, node["method"]),
        model=MLPSpec(cast(int, model["hidden_width"])), training=_training(node["training"]),
        rounds=cast(int, node["rounds"]), target_bit=cast(int, node["target_bit"]),
        label_mode=cast(str, node["label_mode"]),
    )


def parse_shared_m(value: object) -> SharedMRecipe:
    """Validate one explicit shared-M recipe, without a registry or file access."""
    node = _object(value, {"schema", "method", "model", "training", "tasks"}, "shared-M recipe")
    model = _object(node["model"], {"stem_width", "channels", "norm_groups", "suffix_kernels",
                                  "head_hidden", "router_hidden"}, "shared-M model")
    kernels = model["suffix_kernels"]
    if not isinstance(kernels, list):
        raise ValueError("suffix_kernels: expected array")
    spec = SharedMSpec(
        stem_width=cast(int, model["stem_width"]), channels=cast(int, model["channels"]),
        norm_groups=cast(int, model["norm_groups"]), suffix_kernels=tuple(cast(list[int], kernels)),
        head_hidden=cast(int, model["head_hidden"]),
        router_hidden=cast(int, model["router_hidden"]),
    )
    raw_tasks = node["tasks"]
    if not isinstance(raw_tasks, list):
        raise ValueError("tasks: expected array")
    tasks: list[TaskSpec] = []
    for raw_task in raw_tasks:
        task = _object(raw_task, {"cipher", "rounds", "difference"}, "task")
        tasks.append(TaskSpec(cast(str, task["cipher"]), cast(int, task["rounds"]),
                              cast(int, task["difference"])))
    return SharedMRecipe(
        schema=cast(int, node["schema"]), method=cast(str, node["method"]), model=spec,
        training=_training(node["training"]), tasks=tuple(tasks),
    )
