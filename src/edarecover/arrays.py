"""Array type aliases shared by the numerical modules.

Kept in one place so the annotations stay identical everywhere and a change of
precision is a single edit.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Array: TypeAlias = npt.NDArray[np.float64]
"""Floating-point signal or result vector."""

IntArray: TypeAlias = npt.NDArray[np.int_]
"""Integer labels, seeds or activity codes."""
