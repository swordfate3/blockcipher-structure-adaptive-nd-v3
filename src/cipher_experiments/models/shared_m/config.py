"""Small, explicit architecture configuration (not an experiment configuration)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import cast

TOP2_PAIRS = ((0, 1), (0, 2), (1, 2))
STEM_NAMES = ("local", "word", "nibble")


def _check_fields(raw: Mapping[str, object], cls: type[object]) -> None:
    allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown, key=str)}")


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_pair(pair: object) -> tuple[int, int]:
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(type(i) is not int for i in pair)
        or pair not in TOP2_PAIRS
    ):
        raise ValueError("an ordered, distinct top-2 pair (0,1), (0,2) or (1,2) is required")
    return pair


@dataclass(frozen=True)
class RouterConfig:
    mode: str = "learned"
    hidden_width: int = 16
    fixed_pair: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("learned", "constant", "fixed"):
            raise ValueError("router.mode must be learned, constant or fixed")
        _integer(self.hidden_width, "router.hidden_width")
        if self.mode == "fixed":
            validate_pair(self.fixed_pair)
        elif self.fixed_pair is not None:
            raise ValueError("router.fixed_pair is only valid with fixed mode")

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> RouterConfig:
        _check_fields(raw, cls)
        pair = raw.get("fixed_pair")
        if isinstance(pair, list):
            pair = tuple(pair)
        return cls(
            mode=cast(str, raw.get("mode", "learned")),
            hidden_width=cast(int, raw.get("hidden_width", 16)),
            fixed_pair=cast(tuple[int, int] | None, pair),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Exactly four 64-bit pairs; widths/kernels are explicit architecture controls.

    The default is state-and-function compatible with SkinnyPairMoE. Only the
    reviewed SIMECK64, SIMON64, PRESENT80 and SKINNY64 public cards are supported;
    this is not a variable-block-width cipher framework.
    """

    block_bits: int = 64
    pairs: int = 4
    stem_width: int = 96
    channels: int = 96
    norm_groups: int = 8
    suffix_kernels: tuple[int, ...] = (3, 5, 7, 9, 11)
    head_hidden: int = 64
    head: str = "pair"
    representation: str = "relation"
    router: RouterConfig = field(default_factory=RouterConfig)
    view_mode: str = "single"

    def __post_init__(self) -> None:
        if type(self.block_bits) is not int or self.block_bits != 64:
            raise ValueError("only block_bits=64 is supported")
        if type(self.pairs) is not int or self.pairs != 4:
            raise ValueError("exactly pairs=4 is required")
        for name in ("stem_width", "channels", "norm_groups", "head_hidden"):
            _integer(getattr(self, name), name)
        if self.stem_width % self.norm_groups or self.channels % self.norm_groups:
            raise ValueError("stem_width and channels must be divisible by norm_groups")
        if not isinstance(self.suffix_kernels, tuple) or not self.suffix_kernels:
            raise ValueError("suffix_kernels must be a nonempty tuple of odd positive integers")
        for kernel in self.suffix_kernels:
            if _integer(kernel, "suffix kernel") % 2 == 0:
                raise ValueError("suffix kernels must be odd to preserve bit alignment")
        if self.head not in ("pair", "group"):
            raise ValueError("head must be pair or group")
        if self.representation not in ("relation", "raw"):
            raise ValueError("representation must be relation or raw")
        if self.view_mode not in ("single", "tt", "ct"):
            raise ValueError("view_mode must be single, tt or ct")
        if not isinstance(self.router, RouterConfig):
            raise ValueError("router must be a RouterConfig")

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> ModelConfig:
        """Decode JSON-friendly lists, rejecting unknown fields at every level."""
        _check_fields(raw, cls)
        router = raw.get("router", {})
        if not isinstance(router, Mapping):
            raise ValueError("router must be a mapping")
        kernels = raw.get("suffix_kernels", (3, 5, 7, 9, 11))
        if isinstance(kernels, list):
            kernels = tuple(kernels)
        return cls(
            block_bits=cast(int, raw.get("block_bits", 64)),
            pairs=cast(int, raw.get("pairs", 4)),
            stem_width=cast(int, raw.get("stem_width", 96)),
            channels=cast(int, raw.get("channels", 96)),
            norm_groups=cast(int, raw.get("norm_groups", 8)),
            suffix_kernels=cast(tuple[int, ...], kernels),
            head_hidden=cast(int, raw.get("head_hidden", 64)),
            head=cast(str, raw.get("head", "pair")),
            representation=cast(str, raw.get("representation", "relation")),
            router=RouterConfig.from_dict(router),
            view_mode=cast(str, raw.get("view_mode", "single")),
        )
