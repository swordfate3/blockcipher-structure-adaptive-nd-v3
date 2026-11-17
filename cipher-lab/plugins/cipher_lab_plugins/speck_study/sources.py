"""Fresh source supplier shared by claim and the study's saved-context guards."""
from __future__ import annotations

from cipher_lab_plugins.speck_study.paths import source_roots
from cipher_lab_plugins.speck_study.runtime.source import package_snapshot


def source_identity() -> dict[str, str]:
    return package_snapshot(*source_roots())
