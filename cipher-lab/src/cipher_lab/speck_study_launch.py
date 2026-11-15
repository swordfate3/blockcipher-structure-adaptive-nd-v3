"""Stdlib-only launch metadata validation; never claims or probes GPU/runtime.

Pinned weights are metadata, not bundled assets. Missing Windows actual-memory
or durability proof cannot be bypassed by passing valid settings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

OFFICIAL_REVISION = '96344da116ff624be30f503038b22f26c7f7f693'
OFFICIAL_WEIGHTS_SHA256 = '15c58fff86855e882be1f977e42006761eeb5b9af28739d7a7fc8f44834902c0'


@dataclass(frozen=True, slots=True)
class LaunchSettings:
    teacher_python: Path
    weights: Path
    revision: str
    weights_sha256: str
    gpu_uuid: str
    gpu_pci: str
    source_manifest: Path

    def __post_init__(self) -> None:
        # Same address language as platform_resources.gpu.pci, without importing
        # machine observation here: only a zero-extended 16-bit domain is valid.
        pci = (re.fullmatch(
            r'(?:0000)?([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])',
            self.gpu_pci) if type(self.gpu_pci) is str else None)
        if (self.revision != OFFICIAL_REVISION or self.weights_sha256 != OFFICIAL_WEIGHTS_SHA256
                or re.fullmatch(r'GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}',
                                self.gpu_uuid) is None
                or pci is None):
            raise ValueError('invalid frozen study metadata')
        object.__setattr__(self, 'gpu_pci', pci[1].lower())
        for path in (self.teacher_python, self.weights, self.source_manifest):
            if not path.is_absolute() or not path.is_file() or path.is_symlink():
                raise ValueError('missing explicit launch file')
