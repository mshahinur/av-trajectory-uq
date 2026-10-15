"""Short-horizon trajectory forecasting with calibrated uncertainty.

Predict the position 3 s ahead from 2 s of noisy history.
  - Constant velocity (CV) baseline
  - Gradient-boosted regressor on hand-built kinematic features
Prediction intervals:
  - Split conformal prediction (exchangeable data)
  - Adaptive conformal inference, ACI (Gibbs & Candes, 2021), for drift
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from data import FS

HIST = int(2.0 * FS)
HOR = int(3.0 * FS)


def features(hist):
    """hist: (HIST, 2) noisy positions. Returns features relative to last point."""
    t = np.arange(HIST) / FS
    t = t - t[-1]
    feats = []
    for k in range(2):
        c2, c1, c0 = np.polyfit(t, hist[:, k], 2)   # local quadratic fit
        feats += [c1, 2 * c2]                       # velocity, acceleration
        feats += [hist[-1, k] - c0]                 # residual of last point
    vy_recent = np.polyfit(t[-12:], hist[-12:, 1], 1)[0]
    feats.append(vy_recent)
    feats.append(hist[-1, 1] % 3.75)                # position within lane
    return np.array(feats)


def windows(dataset, stride=10):
    X, Ycv, Y = [], [], []
    for t, truth, obs, _ in dataset:
        for s in range(0, len(t) - HIST - HOR, stride):
            h = obs[s:s + HIST]
            tt = np.arange(HIST) / FS
            v = np.array([np.polyfit(tt, h[:, k], 1)[0] for k in range(2)])
            anchor = np.array([np.polyval(np.polyfit(tt, h[:, k], 1), tt[-1]) for k in range(2)])
            cv = anchor + v * (HOR / FS)
            target = truth[s + HIST + HOR - 1]
            X.append(features(h))
            Ycv.append(cv)
            Y.append(target)
    return np.array(X), np.array(Ycv), np.array(Y)


class ResidualGBR:
    """Learns the correction on top of the CV forecast, one model per axis."""

    def fit(self, X, Ycv, Y):
        self.m = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                random_state=0).fit(X, Y[:, k] - Ycv[:, k])
                  for k in range(2)]
        return self

    def predict(self, X, Ycv):
        return Ycv + np.stack([m.predict(X) for m in self.m], 1)


def split_conformal_radius(pred, y, alpha=0.1):
    """Radius of a disc covering (1 - alpha) of calibration errors."""
    s = np.linalg.norm(pred - y, axis=1)
    n = len(s)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return np.quantile(s, min(q, 1.0), method="higher")


def aci(pred, y, cal_scores, alpha=0.1, gamma=0.01):
    """Adaptive conformal inference over a stream. Returns covered flags, radii."""
    a_t = alpha
    scores = list(cal_scores)
    covered, radii = [], []
    for p, yy in zip(pred, y):
        level = np.clip(1 - a_t, 0.0, 1.0)
        r = np.quantile(scores, level, method="higher") if level < 1 else np.inf
        s = np.linalg.norm(p - yy)
        err = float(s > r)
        covered.append(1 - err)
        radii.append(r if np.isfinite(r) else max(scores))
        a_t = a_t + gamma * (alpha - err)
        scores.append(s)
    return np.array(covered), np.array(radii)
