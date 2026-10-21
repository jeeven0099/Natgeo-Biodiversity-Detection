"""A synthetic soundscape with known events, for the demo and the tests.

No real recording ships with this repository. The soundscape is pink background noise
plus low-frequency wind, with five kinds of call-like events at known times and bands:

  downslur   8 -> 5 kHz frequency sweep, 0.3 s (bird-like)
  whistle    steady 6 kHz tone, 0.5 s
  pulses     train of 12 kHz pulses, 15 ms each, 0.4 s (insect-like)
  chirp      2.2 kHz fundamental + 4.4 kHz harmonic, 0.15 s (frog-like)
  click      broadband 3 ms click (rain- or handling-like)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ["downslur", "whistle", "pulses", "chirp", "click"]


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    x = np.fft.irfft(spec / np.sqrt(f), n=n)
    return x / x.std()


def _envelope(n: int) -> np.ndarray:
    return np.hanning(n) if n > 2 else np.ones(n)


def _event(kind: str, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """(waveform, low_hz, high_hz) of one event at unit peak amplitude."""
    if kind == "downslur":
        d = 0.3
        t = np.arange(int(d * sr)) / sr
        f0, f1 = 8000.0, 5000.0
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * d))
        return np.sin(phase) * _envelope(len(t)), f1, f0
    if kind == "whistle":
        t = np.arange(int(0.5 * sr)) / sr
        return np.sin(2 * np.pi * 6000 * t) * _envelope(len(t)), 6000.0, 6000.0
    if kind == "pulses":
        n, pulse, gap = int(0.4 * sr), int(0.015 * sr), int(0.035 * sr)
        x = np.zeros(n)
        t = np.arange(pulse) / sr
        for a in range(0, n - pulse, pulse + gap):
            x[a:a + pulse] = np.sin(2 * np.pi * 12000 * t) * _envelope(pulse)
        return x, 12000.0, 12000.0
    if kind == "chirp":
        t = np.arange(int(0.15 * sr)) / sr
        x = np.sin(2 * np.pi * 2200 * t) + 0.5 * np.sin(2 * np.pi * 4400 * t)
        return x / 1.5 * _envelope(len(t)), 2200.0, 4400.0
    if kind == "click":
        n = int(0.003 * sr)
        return rng.standard_normal(n) * _envelope(n), 0.0, sr / 2
    raise ValueError(kind)


def soundscape(duration_s: float = 60.0, sr: int = 48000, seed: int = 0, n_events: int = 30,
               snr_db: tuple[float, float] = (12.0, 24.0)) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (samples in [-1, 1], one row per event: kind, begin_s, end_s, low_hz, high_hz, snr_db)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    bg = pink_noise(n, rng)
    wind = np.convolve(rng.standard_normal(n), np.ones(400) / 400, mode="same")
    bg = 0.01 * bg + 0.02 * wind / wind.std()
    x = bg.copy()
    rows = []
    slots = np.sort(rng.choice(np.arange(1.0, duration_s - 1.0, 0.75), size=n_events, replace=False))
    for k, t0 in enumerate(slots):
        kind = KINDS[k % len(KINDS)]
        w, lo, hi = _event(kind, sr, rng)
        snr = rng.uniform(*snr_db)
        a = int(t0 * sr)
        w = w[: n - a]
        x[a:a + len(w)] += w * 0.01 * 10 ** (snr / 20)
        rows.append({"kind": kind, "begin_s": t0, "end_s": t0 + len(w) / sr, "low_hz": lo, "high_hz": hi,
                     "snr_db": snr})
    x = 0.9 * x / np.abs(x).max()
    return x, pd.DataFrame(rows)


def write(folder: str | Path, n_files: int = 3, duration_s: float = 120.0, sr: int = 48000,
          seed: int = 0) -> pd.DataFrame:
    """Write n_files synthetic recordings (16-bit WAV) and events.csv to `folder`; returns the events."""
    import soundfile as sf

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    events = []
    for i in range(n_files):
        name = f"synthetic_{i + 1:02d}"
        x, ev = soundscape(duration_s, sr, seed + i, n_events=int(duration_s / 2))
        sf.write(folder / f"{name}.wav", x, sr, subtype="PCM_16")
        events.append(ev.assign(file=name))
    ev = pd.concat(events, ignore_index=True)
    ev.to_csv(folder / "events.csv", index=False)
    return ev
