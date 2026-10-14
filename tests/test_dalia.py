"""Reading and classifying a PPG-DaLiA style corpus."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from edarecover.config import Settings
from edarecover.dalia import (
    FS_CONDUCTANCE,
    dominant_activity,
    iter_subject_files,
    load_subject,
    report_subject,
    settings_for_dalia,
)


def test_corpus_settings_keep_the_frequency_resolution(settings: Settings):
    corpus = settings_for_dalia(settings)
    assert corpus.fs == FS_CONDUCTANCE
    assert corpus.frequency_resolution_hz == pytest.approx(
        settings.frequency_resolution_hz
    )
    assert corpus.window_s == settings.window_s


def test_subject_files_are_listed_in_order(corpus: Path):
    found = iter_subject_files(corpus)
    assert [path.parent.name for path in found] == ["S1", "S2"]


def test_a_missing_corpus_yields_nothing(tmp_path: Path):
    assert iter_subject_files(tmp_path) == []


def test_loading_puts_every_channel_on_one_grid(corpus: Path, settings: Settings):
    subject = load_subject(corpus / "S1" / "S1.pkl", settings_for_dalia(settings))
    assert subject.identifier == "S1"
    assert subject.conductance.shape == subject.envelope.shape
    assert subject.conductance.shape == subject.activity.shape
    assert subject.envelope.min() >= 0.0


def test_acceleration_of_the_wrong_length_is_rejected(
    tmp_path: Path, settings: Settings
):
    folder = tmp_path / "S9"
    folder.mkdir()
    payload = {
        "subject": "S9",
        "signal": {"wrist": {"EDA": np.ones((400, 1)), "ACC": np.ones((123, 3))}},
        "activity": np.zeros((400, 1), dtype=int),
    }
    path = folder / "S9.pkl"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(ValueError, match="expected about"):
        load_subject(path, settings_for_dalia(settings))


def test_windows_are_reported_with_their_activity(corpus: Path, settings: Settings):
    corpus_settings = settings_for_dalia(settings)
    subject = load_subject(corpus / "S1" / "S1.pkl", corpus_settings)
    reports = report_subject(subject, corpus_settings)

    expected = len(subject.conductance) // corpus_settings.samples_per_window
    assert len(reports) == expected
    assert [r.index for r in reports] == list(range(expected))
    assert all(0.0 <= r.coherence <= 1.0 for r in reports)
    assert all(isinstance(r.correct, bool) for r in reports)


def test_activity_names_fall_back_to_the_raw_code():
    assert dominant_activity(np.array([7, 7, 1])) == "walking"
    assert dominant_activity(np.array([42, 42])) == "code 42"
