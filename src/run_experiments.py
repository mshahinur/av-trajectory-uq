"""Runs both experiments and writes results/ and figures/.

Usage:  python src/run_experiments.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data import FS, make_dataset, make_gap
from gapfill import METHODS, kalman_fill
from forecast import windows, ResidualGBR, split_conformal_radius, aci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = {"Linear": "#9a9a9a", "Smoothing spline": "#2f6db3", "Kalman (RTS)": "#d0702b"}


def exp_gapfill(n=300, seed=1):
    data = make_dataset(n, seed=seed, aggressive_frac=0.3)
    rng = np.random.default_rng(seed + 100)
    gaps_s = [0.2, 0.4, 0.8, 1.2, 1.6, 2.0]
    table = {m: [] for m in METHODS}
    table_sd = {m: [] for m in METHODS}
    kf_cov = []
    for g in gaps_s:
        gl = int(g * FS)
        errs = {m: [] for m in METHODS}
        cov = []
        for t, truth, obs, _ in data:
            mask = make_gap(rng, len(t), gl)
            for m, f in METHODS.items():
                est = f(t, obs, mask)
                e = np.linalg.norm(est[~mask] - truth[~mask], axis=1)
                errs[m].append(np.sqrt(np.mean(e ** 2)))
            est, sd = kalman_fill(t, obs, mask)
            z = np.abs(est[~mask] - truth[~mask]) / sd[~mask]
            cov.append(np.mean(z < 1.96))
        for m in METHODS:
            table[m].append(float(np.mean(errs[m])))
            table_sd[m].append(float(np.std(errs[m]) / np.sqrt(n)))
        kf_cov.append(float(np.mean(cov)))

    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for m in METHODS:
        ax.errorbar(gaps_s, table[m], yerr=1.96 * np.array(table_sd[m]), marker="o",
                    ms=4, capsize=3, color=COL[m], label=m)
    ax.set_xlabel("Dropout length (s)")
    ax.set_ylabel("Reconstruction RMSE in gap (m)")
    ax.legend(frameon=False)
    ax.set_title("Gap filling on 300 highway trajectories", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_gapfill_rmse.png"), dpi=200)
    plt.close(fig)

    # Example reconstruction for the figure
    t, truth, obs, _ = data[3]
    rng2 = np.random.default_rng(7)
    mask = make_gap(rng2, len(t), int(1.6 * FS))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(t, truth[:, 1], color="k", lw=1.2, label="Ground truth")
    ax.scatter(t[mask], obs[mask, 1], s=4, color="#bbbbbb", label="Noisy observations")
    for m, f in METHODS.items():
        est = f(t, obs, mask)
        ax.plot(t[~mask], est[~mask, 1], color=COL[m], lw=2, label=m)
    ax.axvspan(t[~mask][0], t[~mask][-1], color="#f2e3d5", alpha=0.6, lw=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lateral position (m)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_gapfill_example.png"), dpi=200)
    plt.close(fig)

    return {"gaps_s": gaps_s, "rmse": table, "rmse_se": table_sd,
            "kalman_95_band_coverage": kf_cov}


def exp_forecast(alpha=0.1):
    train = make_dataset(400, seed=10, aggressive_frac=0.2)
    cal = make_dataset(150, seed=11, aggressive_frac=0.2)
    test_iid = make_dataset(150, seed=12, aggressive_frac=0.2)
    test_shift = make_dataset(150, seed=13, aggressive_frac=0.8, noise_std=0.5)

    Xtr, Ctr, Ytr = windows(train)
    Xc, Cc, Yc = windows(cal)
    Xi, Ci, Yi = windows(test_iid)
    Xs, Cs, Ys = windows(test_shift)

    model = ResidualGBR().fit(Xtr, Ctr, Ytr)
    out = {"n_windows": {"train": len(Ytr), "cal": len(Yc), "test_iid": len(Yi),
                         "test_shift": len(Ys)}}

    def ade(p, y):
        return float(np.mean(np.linalg.norm(p - y, axis=1)))

    pc = model.predict(Xc, Cc)
    r_cv = split_conformal_radius(Cc, Yc, alpha)
    r_gb = split_conformal_radius(pc, Yc, alpha)
    out["final_disp_error_m"] = {}
    out["split_conformal"] = {}
    for name, X, C, Y in [("iid", Xi, Ci, Yi), ("shift", Xs, Cs, Ys)]:
        pg = model.predict(X, C)
        out["final_disp_error_m"][name] = {"CV": ade(C, Y), "GBR": ade(pg, Y)}
        out["split_conformal"][name] = {
            "CV": {"coverage": float(np.mean(np.linalg.norm(C - Y, axis=1) <= r_cv)),
                   "radius_m": float(r_cv)},
            "GBR": {"coverage": float(np.mean(np.linalg.norm(pg - Y, axis=1) <= r_gb)),
                    "radius_m": float(r_gb)}}

    # ACI on the shifted stream, interleaving iid then shift to show adaptation
    pg_i, pg_s = model.predict(Xi, Ci), model.predict(Xs, Cs)
    stream_p = np.concatenate([pg_i, pg_s])
    stream_y = np.concatenate([Yi, Ys])
    cal_scores = np.linalg.norm(pc - Yc, axis=1)
    cov_aci, rad_aci = aci(stream_p, stream_y, cal_scores, alpha=alpha, gamma=0.01)
    cov_static = (np.linalg.norm(stream_p - stream_y, axis=1) <= r_gb).astype(float)
    k = len(pg_i)
    out["aci"] = {
        "static_coverage_after_shift": float(cov_static[k:].mean()),
        "aci_coverage_after_shift": float(cov_aci[k:].mean()),
        "aci_mean_radius_after_shift_m": float(rad_aci[k:].mean()),
        "gamma": 0.01,
    }

    w = 150
    roll = lambda v: np.convolve(v, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    xs = np.arange(len(roll(cov_static)))
    ax.plot(xs, roll(cov_static), color="#9a9a9a", label="Split conformal (fixed)")
    ax.plot(xs, roll(cov_aci), color="#2f6db3", label="Adaptive conformal (ACI)")
    ax.axhline(1 - alpha, color="k", ls="--", lw=1, label="Target 90%")
    ax.axvline(k - w / 2, color="#d0702b", lw=1)
    ax.text(k - w / 2 + 15, 0.62, "drift begins:\naggressive drivers,\n2x sensor noise",
            fontsize=8, color="#d0702b")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Forecast index in stream")
    ax.set_ylabel(f"Rolling coverage ({w} windows)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_conformal_coverage.png"), dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    results = {"gapfill": exp_gapfill(), "forecast": exp_forecast()}
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
