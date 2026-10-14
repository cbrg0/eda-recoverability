"""Reading wrist recordings from the PPG-DaLiA corpus.

Each subject file holds conductance at 4 Hz and three-axis acceleration at
32 Hz from an Empatica E4, together with an activity code on the conductance
grid (Reiss et al., 2019). Acceleration is reduced to a magnitude and brought
down to the conductance rate through an anti-aliasing filter, which is what
keeps the band coherence comparable with the calibrated values.
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sps

from edarecover.arrays import Array, IntArray
from edarecover.calibration import needs_correction
from edarecover.config import Settings
from edarecover.features import coherence_per_window, motion_envelope

log = logging.getLogger(__name__)

FS_CONDUCTANCE = 4.0
FS_ACCELERATION = 32.0
ACTIVITY_NAMES = {
    0: "unlabelled",
    1: "sitting",
    2: "stairs",
    3: "soccer",
    4: "cycling",
    5: "driving",
    6: "lunch",
    7: "walking",
    8: "working",
}


@dataclass(frozen=True, slots=True)
class Subject:
    """One subject's wrist recording on a single time grid.

    Attributes:
        identifier: Subject label as stored in the file.
        conductance: Skin conductance at 4 Hz, microsiemens.
        envelope: Movement envelope on the same grid.
        activity: Activity code per sample.
    """

    identifier: str
    conductance: Array
    envelope: Array
    activity: IntArray


@dataclass(frozen=True, slots=True)
class WindowReport:
    """Classification of one analysis window.

    Attributes:
        subject: Subject label.
        index: Window position in the recording.
        centre_s: Time at the middle of the window, s.
        coherence: Band coherence of the window.
        activity: Name of the activity occupying most of the window.
        correct: Whether the window falls above the threshold.
    """

    subject: str
    index: int
    centre_s: float
    coherence: float
    activity: str
    correct: bool


def settings_for_dalia(settings: Settings) -> Settings:
    """Return the calibration settings moved to the corpus sampling rate.

    The window and segment lengths stay in seconds, so the frequency resolution
    and the number of averages behind each estimate are unchanged.
    """
    return settings.at_sampling_rate(FS_CONDUCTANCE)


def load_subject(path: Path, settings: Settings) -> Subject:
    """Read one subject file.

    Args:
        path: Path to ``S{i}.pkl``.
        settings: Supplies the envelope smoothing length.

    Returns:
        Conductance, envelope and activity on the 4 Hz grid.

    Raises:
        ValueError: If the acceleration length does not match the conductance
            length at the documented ratio, which would mean the file layout
            differs from the one this reader expects.

    Warning:
        The corpus is distributed as pickle files, and unpickling executes
        arbitrary code. Pass only files obtained from the published corpus.
    """
    # The corpus ships as pickles. Unpickling executes code, so this is safe
    # only for files the caller obtained from the published corpus.
    with path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")  # noqa: S301

    conductance = raw["signal"]["wrist"]["EDA"].astype(np.float64).ravel()
    acceleration = raw["signal"]["wrist"]["ACC"].astype(np.float64)
    activity = raw["activity"].astype(int).ravel()
    identifier = str(raw.get("subject", path.stem))

    ratio = int(FS_ACCELERATION / FS_CONDUCTANCE)
    expected = conductance.size * ratio
    if abs(acceleration.shape[0] - expected) > ratio:
        raise ValueError(
            f"{identifier}: acceleration has {acceleration.shape[0]} samples, "
            f"expected about {expected} for {conductance.size} conductance samples"
        )

    magnitude = np.linalg.norm(acceleration, axis=1)
    reduced = sps.decimate(magnitude, ratio, ftype="fir", zero_phase=True)

    n = min(conductance.size, reduced.size, activity.size)
    return Subject(
        identifier=identifier,
        conductance=conductance[:n],
        envelope=motion_envelope(
            reduced[:n], FS_CONDUCTANCE, settings.envelope_smooth_s
        ),
        activity=activity[:n],
    )


def dominant_activity(codes: IntArray) -> str:
    """Name the activity occupying most of a window."""
    code, _ = Counter(codes.tolist()).most_common(1)[0]
    return ACTIVITY_NAMES.get(code, f"code {code}")


def report_subject(subject: Subject, settings: Settings) -> list[WindowReport]:
    """Classify every complete window of one subject.

    Args:
        subject: Recording to analyse.
        settings: Calibration settings at the corpus sampling rate.

    Returns:
        One report per window.
    """
    width = settings.samples_per_window
    windows = coherence_per_window(subject.conductance, subject.envelope, settings)

    return [
        WindowReport(
            subject=subject.identifier,
            index=window.number,
            centre_s=window.centre_s,
            coherence=window.value,
            activity=dominant_activity(
                subject.activity[window.number * width : (window.number + 1) * width]
            ),
            correct=needs_correction(window.value, settings.threshold),
        )
        for window in windows
    ]


def iter_subject_files(root: Path) -> list[Path]:
    """List subject files under a corpus directory, in numerical order.

    Args:
        root: Directory holding ``S1/S1.pkl`` and so on.

    Returns:
        Existing paths, sorted by subject number.
    """
    found = []
    for number in range(1, 16):
        candidate = root / f"S{number}" / f"S{number}.pkl"
        if candidate.exists():
            found.append(candidate)
        else:
            log.warning("missing %s", candidate)
    return found
