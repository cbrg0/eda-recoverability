"""Parameters of the recoverability experiment.

All numbers that control the model live here. Nothing elsewhere in the package
reads a bare literal, so a change of sampling rate or analysis band is a change
in one object rather than a search through the modules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable parameter set for one experiment.

    Attributes:
        fs: Sampling rate of the synthetic recordings, Hz.
        window_s: Length of one analysis window, s.
        segment_s: Welch segment length inside a window, s.
        band_hz: Frequency band over which coherence is averaged, Hz.
        noise_std: Standard deviation of measurement noise, microsiemens.
        envelope_smooth_s: Moving-average length of the motion envelope, s.
        beta_per_blur: Additive-trace amplitude per second of blur,
            microsiemens per unit of envelope.
        rmse_tolerance: Error above which a window is said to need
            correction, microsiemens.
        threshold: Calibrated coherence threshold, dimensionless.
        blur_levels: Blur widths swept during calibration, s.
        n_seeds: Number of random seeds per blur level.
        lambda_grid_size: Number of regularisation parameters on the L-curve.
        lambda_log_bounds: Base-10 exponents bounding the lambda grid.
    """

    fs: float = 16.0
    window_s: float = 240.0
    segment_s: float = 20.0
    band_hz: tuple[float, float] = (0.045, 0.25)
    noise_std: float = 0.01
    envelope_smooth_s: float = 2.0
    beta_per_blur: float = 0.3
    rmse_tolerance: float = 0.08
    threshold: float = 0.156
    blur_levels: tuple[float, ...] = (
        0.0,
        0.1,
        0.2,
        0.4,
        0.7,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
    )
    n_seeds: int = 6
    lambda_grid_size: int = 7
    lambda_log_bounds: tuple[float, float] = (-3.0, 2.0)

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        if self.segment_s > self.window_s:
            raise ValueError(
                f"segment_s ({self.segment_s}) exceeds window_s ({self.window_s})"
            )
        low, high = self.band_hz
        if not 0 < low < high:
            raise ValueError(f"band_hz must satisfy 0 < low < high, got {self.band_hz}")
        if high >= self.fs / 2:
            raise ValueError(
                f"band upper edge {high} Hz is at or above Nyquist {self.fs / 2} Hz"
            )
        if self.frequency_resolution_hz > high - low:
            raise ValueError(
                f"resolution {self.frequency_resolution_hz:.4f} Hz is coarser "
                f"than the band width {high - low:.4f} Hz; lengthen segment_s"
            )

    @property
    def samples_per_segment(self) -> int:
        """Welch segment length in samples."""
        return round(self.segment_s * self.fs)

    @property
    def samples_per_window(self) -> int:
        """Analysis window length in samples."""
        return round(self.window_s * self.fs)

    @property
    def frequency_resolution_hz(self) -> float:
        """Spacing between adjacent Welch frequency bins."""
        return self.fs / self.samples_per_segment

    def at_sampling_rate(self, fs: float) -> Settings:
        """Return a copy at a different sampling rate, validating the band."""
        return replace(self, fs=fs)


DEFAULT = Settings()
