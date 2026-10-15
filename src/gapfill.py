"""Reconstructing trajectory segments lost to sensor dropouts.

Methods compared:
  linear   - straight-line interpolation across the gap
  spline   - cubic smoothing spline fitted to the observed samples
  kalman   - constant-acceleration Kalman filter + Rauch-Tung-Striebel smoother
"""
import numpy as np
from scipy.interpolate import make_smoothing_spline


def linear_fill(t, z, mask):
    return np.stack([np.interp(t, t[mask], z[mask, k]) for k in range(z.shape[1])], 1)


def spline_fill(t, z, mask, noise_std=0.25):
    out = []
    for k in range(z.shape[1]):
        # Smoothing parameter chosen by generalised cross validation (lam=None).
        s = make_smoothing_spline(t[mask], z[mask, k])
        out.append(s(t))
    return np.stack(out, 1)


def _rts_1d(t, z, mask, q=3.0, r=0.25):
    """Constant-acceleration model on one axis, RTS smoothed."""
    dt = t[1] - t[0]
    F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])
    G = np.array([[dt ** 3 / 6], [dt ** 2 / 2], [dt]])
    Q = q * (G @ G.T)
    H = np.array([[1.0, 0, 0]])
    R = np.array([[r * r]])
    n = len(t)
    x = np.array([z[mask][0], 0.0, 0.0])
    if mask.sum() > 1:
        i0, i1 = np.flatnonzero(mask)[:2]
        x[1] = (z[i1] - z[i0]) / (t[i1] - t[i0])
    P = np.diag([1.0, 25.0, 4.0])
    xf, Pf, xp, Pp = (np.zeros((n, 3)), np.zeros((n, 3, 3)),
                      np.zeros((n, 3)), np.zeros((n, 3, 3)))
    for i in range(n):
        if i > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xp[i], Pp[i] = x, P
        if mask[i]:
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + (K @ (z[i] - H @ x)).ravel()
            P = (np.eye(3) - K @ H) @ P
        xf[i], Pf[i] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    for i in range(n - 2, -1, -1):
        C = Pf[i] @ F.T @ np.linalg.inv(Pp[i + 1])
        xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Pf[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T
    return xs[:, 0], np.sqrt(Ps[:, 0, 0])


def kalman_fill(t, z, mask, noise_std=0.25):
    cols, sds = zip(*[_rts_1d(t, z[:, k], mask, r=noise_std) for k in range(z.shape[1])])
    return np.stack(cols, 1), np.stack(sds, 1)


METHODS = {
    "Linear": lambda t, z, m: linear_fill(t, z, m),
    "Smoothing spline": lambda t, z, m: spline_fill(t, z, m),
    "Kalman (RTS)": lambda t, z, m: kalman_fill(t, z, m)[0],
}
