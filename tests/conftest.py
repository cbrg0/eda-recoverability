"""Fixtures shared across the suite."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from edarecover.config import Settings
from edarecover.dalia import FS_ACCELERATION, FS_CONDUCTANCE


@pytest.fixture
def settings() -> Settings:
    """Default parameters."""
    return Settings()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A two-subject corpus laid out the way PPG-DaLiA files are.

    Written rather than downloaded so the reader is tested without the real
    corpus, which is gigabytes and not redistributable.
    """
    rng = np.random.default_rng(0)
    ratio = int(FS_ACCELERATION / FS_CONDUCTANCE)
    n = int(600 * FS_CONDUCTANCE)

    for number in (1, 2):
        folder = tmp_path / f"S{number}"
        folder.mkdir()
        payload = {
            "subject": f"S{number}",
            "signal": {
                "wrist": {
                    "EDA": (3.0 + 0.1 * rng.standard_normal((n, 1))),
                    "ACC": rng.standard_normal((n * ratio, 3)),
                }
            },
            "activity": rng.integers(0, 9, size=(n, 1)),
        }
        with (folder / f"S{number}.pkl").open("wb") as handle:
            pickle.dump(payload, handle)

    return tmp_path
