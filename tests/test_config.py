"""Parameter validation and the derived window arithmetic."""

import pytest

from edarecover.config import Settings


def test_resolution_follows_segment_length_not_sampling_rate():
    base = Settings()
    assert base.at_sampling_rate(4.0).frequency_resolution_hz == pytest.approx(
        base.frequency_resolution_hz
    )


def test_segment_longer_than_window_is_rejected():
    with pytest.raises(ValueError, match="exceeds window_s"):
        Settings(segment_s=300.0, window_s=240.0)


def test_band_above_nyquist_is_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        Settings(fs=0.4)


def test_band_narrower_than_one_bin_is_rejected():
    with pytest.raises(ValueError, match="coarser than"):
        Settings(segment_s=2.0)


def test_settings_are_immutable():
    # setattr rather than a plain assignment so the static checkers do not
    # reject the line they are meant to prove is rejected at runtime.
    with pytest.raises(AttributeError):
        setattr(Settings(), "fs", 32.0)  # noqa: B010
