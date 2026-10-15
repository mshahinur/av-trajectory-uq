"""Synthetic highway trajectories with sensor noise and dropouts.

The generator mimics the sampling and kinematics of drone-based highway
datasets such as highD (25 Hz, lane width 3.75 m). It is a stand-in so the
pipeline can be tested end to end before real data is plugged in.
"""
import numpy as np

FS = 25.0          # sampling rate (Hz), same as highD
DT = 1.0 / FS
LANE_W = 3.75      # lane width (m)


def _smooth_noise(rng, n, scale, kernel=50):
    """Low-frequency random signal: white noise convolved with a Hann window."""
    w = np.hanning(kernel)
    w /= w.sum()
    x = rng.normal(0, 1, n + kernel)
    return scale * np.convolve(x, w, mode="same")[:n] * np.sqrt(kernel)


def make_trajectory(rng, duration=8.0, aggressive=False):
    """One vehicle: longitudinal (x) and lateral (y) position in metres."""
    n = int(duration * FS)
    t = np.arange(n) * DT
    v0 = rng.uniform(22, 36)                      # 80 to 130 km/h
    acc_scale = 1.4 if aggressive else 0.5
    a = _smooth_noise(rng, n, acc_scale)
    v = np.clip(v0 + np.cumsum(a) * DT, 10, 45)
    x = np.cumsum(v) * DT

    lane = rng.integers(0, 3)
    y = np.full(n, lane * LANE_W + LANE_W / 2)
    p_change = 0.8 if aggressive else 0.4
    if rng.random() < p_change:                   # lane change, sigmoid profile
        direction = rng.choice([-1, 1])
        tc = rng.uniform(2.0, duration - 2.0)
        dur = rng.uniform(1.5, 3.0) / (1.6 if aggressive else 1.0)
        y = y + direction * LANE_W / (1 + np.exp(-(t - tc) * 8 / dur))
    y = y + _smooth_noise(rng, n, 0.03)          # lane-keeping wander
    return t, np.stack([x, y], axis=1)


def observe(rng, traj, noise_std=0.25):
    """Add measurement noise (detector / localisation error)."""
    return traj + rng.normal(0, noise_std, traj.shape)


def make_gap(rng, n, gap_len):
    """Boolean mask, False inside one dropout of gap_len samples."""
    mask = np.ones(n, dtype=bool)
    start = rng.integers(int(0.25 * n), int(0.75 * n) - gap_len)
    mask[start:start + gap_len] = False
    return mask


def make_dataset(n_vehicles, seed=0, aggressive_frac=0.0, noise_std=0.25):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_vehicles):
        agg = rng.random() < aggressive_frac
        t, truth = make_trajectory(rng, aggressive=agg)
        out.append((t, truth, observe(rng, truth, noise_std), agg))
    return out
