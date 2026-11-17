"""Fail-closed scientific test isolation."""
import pytest

from cipher_lab_plugins.prediction_common.tests.guards import science_guard as science_guard


@pytest.fixture(autouse=True)
def isolated_science(science_guard: None) -> None:
    pass
