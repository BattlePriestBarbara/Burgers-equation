"""
4.4_mode_ablation.py
====================

Section 4.4: Fourier-mode ablation experiment (k_max sweep).

Goal
----
At the nu = 0.001 sharp-feature scenario, study the effect of the number of
Fourier modes k_max on the accuracy / cost of the FNO and the underlying
frequency-domain mechanism.

Procedure
---------
  1. Train six FNOs with k_max in {8, 12, 16, 20, 24, 32} on the 800
     training samples (nu = 0.001, N = 64); all other hyper-parameters are
     identical (Adam lr = 1e-3, cosine schedule, 200 epochs, batch 32).
  2. Evaluate on the 200 test samples: mean rel L2 +/- bootstrap 95% CI,
     parameter count, per-sample inference time, and the convergence epoch.
  3. Frequency-domain error spectrum

        E_k = (1 / N_s) * sum_i | FFT[u_pred_i - u_GT_i](k) |^2

     computed per k_max for k = 0 .. 32 (the N = 64 grid).
  4. Statistics: pairwise bootstrap CIs and Wilcoxon signed-rank tests for
     adjacent k_max values, and Jarque-Bera normality tests for the error
     distributions of k_max = 8 and k_max = 32.

Acceptance criteria
-------------------
  * Error is monotonically non-increasing in k_max with diminishing returns
    (decrease 16->32 smaller than decrease 8->16).
  * Low-k_max error spectra are significantly higher in the band
    k in (k_max, 32].
  * Wilcoxon signed-rank test k_max = 8 vs 32 gives p < 0.01.

Outputs (all with the "4.4" prefix)
-----------------------------------
    data/4.4_ablation_table.csv
    data/4.4_error_spectrum.csv
    data/4.4_fno_kmax{8,12,16,20,24,32}_weights.pt
    figures/4.4_ablation_curve.png
    figures/4.4_error_spectrum.png
    4.4_README.md

All comments and figure labels are in English.
"""

import os
import time
import math

import numpy as np
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# 1. Configuration
# ============================================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NU = 0.001                 # sharp-feature scenario
KMAX_LIST = [8, 12, 16, 20, 24, 32]
N_GRID = 64
N_REF = 2048
N_SAMPLES = 1000
N_TRAIN = 800
N_TEST = 200

FNO_WIDTH = 64
FNO_LAYERS = 4
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
BOOT_ITER = 1000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")

# ============================================================================
# 2. Statistical helpers (implemented without scipy)
# ============================================================================
def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _rankdata(a):
    """Ranks with average ranks for ties (same convention as scipy.stats)."""
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a)
    inv = np.empty(len(a), dtype=int)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return (count[dense] + count[dense - 1] + 1) / 2.0


def wilcoxon_signed_rank(d1, d2):
    """Two-sided Wilcoxon signed-rank test for paired samples.

    Returns (W, p_value).  n = 200 > 50, so the normal approximation with a
    continuity correction is used (equivalent to scipy.stats.wilcoxon with
    method='approx').
    """
    d = np.asarray(d1, dtype=float) - np.asarray(d2, dtype=float)
    d = d[d != 0.0]
    n = len(d)
    r = _rankdata(np.abs(d))
    w = float(np.sum(r[d > 0.0]))
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if w > mu:
        z = (w - mu - 0.5) / sigma
    else:
        z = (w - mu + 0.5) / sigma
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return w, p


def jarque_bera(x):
    """Jarque-Bera normality test.  Returns (JB statistic, p-value).

    p-value uses the chi-square(2) survival function: chi2.sf(jb, 2) =
    exp(-jb / 2).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m2 = np.mean((x - x.mean()) ** 2)
    m3 = np.mean((x - x.mean()) ** 3)
    m4 = np.mean((x - x.mean()) ** 4)
    s = m3 / m2 ** 1.5                     # skewness
    k = m4 / m2 ** 2.0                     # kurtosis
    jb = n / 6.0 * (s ** 2 + (k - 3.0) ** 2 / 4.0)
    p = math.exp(-jb / 2.0)
    return jb, p


def bootstrap_ci(errs, n_iter=BOOT_ITER, alpha=0.05, seed=SEED):
    """95% bootstrap confidence interval of the mean."""
    rng = np.random.default_rng(seed)
    means = np.empty(n_iter)
    for i in range(n_iter):
        means[i] = rng.choice(errs, size=len(errs), replace=True).mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def bootstrap_ci_diff(d1, d2, n_iter=BOOT_ITER, alpha=0.05, seed=SEED):
    """Bootstrap 95% CI of mean(d1) - mean(d2) (paired)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d1) - np.asarray(d2)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, len(d), size=len(d))
        diffs[i] = d[idx].mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


# ============================================================================
# 3. Load modules and data
# ============================================================================
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m42 = _load_module("m42", os.path.join(BASE_DIR, "4.2_resolution_invariance.py"))
FNO1d = m42.FNO1d


def load_data():
    """nu = 0.001 dataset: (u0_all, ref_all, train_idx, test_idx)."""
    u0_all = np.loadtxt(os.path.join(DATA_DIR, "grf_initial_conditions.csv"),
                        delimiter=",", skiprows=1)
    ref_all = np.loadtxt(os.path.join(
        DATA_DIR, "4.3_reference_solutions_nu0p001.csv"),
        delimiter=",", skiprows=1)
    with open(os.path.join(DATA_DIR, "train_test_split.csv")) as fh:
        lines = fh.read().splitlines()[1:]
    train_idx = np.array([i for i, ln in enumerate(lines)
                          if ln.split(",")[1] == "train"])
    test_idx = np.array([i for i, ln in enumerate(lines)
                         if ln.split(",")[1] == "test"])
    return u0_all, ref_all, train_idx, test_idx

# ============================================================================
# 4. Training, evaluation, error spectrum
# ============================================================================
def train_fno_with_history(kmax, X, Y):
    """Train one FNO (with k_max modes) and return (model, loss_history).

    Adam (lr = 1e-3) + cosine schedule, EPOCHS epochs, batch BATCH_SIZE.
    """
    model = FNO1d(modes=kmax, width=FNO_WIDTH, layers=FNO_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = X.shape[0]
    losses = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            pred = model(X[idx])
            loss = F.mse_loss(pred, Y[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        sched.step()
        losses.append(ep_loss / n)
        if ep % 50 == 0:
            print(f"  [k_max={kmax}] epoch {ep:3d}/{EPOCHS}  "
                  f"loss = {losses[-1]:.6e}")
    return model, np.array(losses)


def convergence_epoch(losses, tol=0.02):
    """First epoch where the loss reaches within tol (2%) of its final value."""
    final = float(np.min(losses[-10:]))
    for e, v in enumerate(losses):
        if v <= (1.0 + tol) * final:
            return e + 1
    return int(len(losses))


def evaluate_fno(model, u0_test, ref_test, grid=N_GRID):
    """Per-sample relative L2 errors of the FNO on the test set."""
    step = N_REF // grid
    X = torch.tensor(u0_test[:, ::step], dtype=torch.float32)
    y = ref_test[:, ::step]
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        pred = model(X)
        time_ms = (time.time() - t0) / len(X) * 1000.0
    errs = np.linalg.norm(pred.numpy() - y, axis=1) / np.linalg.norm(y, axis=1)
    return errs, time_ms


def error_spectrum(model, u0_test, ref_test, grid=N_GRID):
    """E_k = mean_i |FFT[u_pred_i - u_GT_i](k)|^2 for k = 0 .. grid/2."""
    step = N_REF // grid
    X = torch.tensor(u0_test[:, ::step], dtype=torch.float32)
    y = ref_test[:, ::step]
    model.eval()
    with torch.no_grad():
        pred = model(X).numpy()
    Fk = np.fft.rfft(pred - y, axis=1)          # (n_test, grid//2 + 1)
    return np.mean(np.abs(Fk) ** 2, axis=0)     # (grid//2 + 1,)


# ============================================================================
# 5. Figures (PNG, English labels)
# ============================================================================
def plot_ablation_curve(rows, path):
    """Error vs k_max with CI band; twin axis for params and inference time."""
    kmax = [r["kmax"] for r in rows]
    err = [r["mean_rel_l2"] for r in rows]
    lo = [r["ci_low"] for r in rows]
    hi = [r["ci_high"] for r in rows]
    params = [r["params"] for r in rows]
    times = [r["time_ms"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.8, 5.2))
    ax1.plot(kmax, err, "o-", lw=1.8, color="#1f77b4", label="Mean rel-L2")
    ax1.fill_between(kmax, lo, hi, color="#1f77b4", alpha=0.2,
                     label="95% CI")
    ax1.set_xlabel(r"Number of Fourier modes  $k_{max}$")
    ax1.set_ylabel("Mean relative L2 error (200 test samples)")
    ax1.set_yscale("log")
    ax1.set_xticks(KMAX_LIST)
    ax1.grid(True, alpha=0.3, which="both")

    ax2 = ax1.twinx()
    ax2.plot(kmax, params, "s--", lw=1.2, color="#2ca02c",
             label="Parameters")
    ax2.plot(kmax, times, "d--", lw=1.2, color="#d62728",
             label="Inference time")
    ax2.set_ylabel("Parameters / inference time (ms)")
    ax2.tick_params(axis="y", labelcolor="#555555")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    ax1.set_title("Fourier-Mode Ablation at $\\nu = 0.001$:\n"
                  "Error, Parameters and Inference Time vs $k_{max}$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_spectrum(kmax_list, spectra, path):
    """Family of log-scale error spectra E_k for every k_max."""
    k = np.arange(N_GRID // 2 + 1)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for i, km in enumerate(kmax_list):
        ax.plot(k, spectra[km], lw=1.5, color=cmap(i / (len(kmax_list) - 1)),
                label=f"$k_{{max}}$ = {km}")
        # mark the k_max boundary
        if km < N_GRID // 2:
            ax.axvline(km, ls=":", lw=0.8, color=cmap(i / (len(kmax_list) - 1)),
                       alpha=0.6)
    ax.set_xlabel(r"Wavenumber  $k$")
    ax.set_ylabel(r"Mean squared Fourier error  $E_k = \langle |\mathcal{F}[u_{pred} - u_{GT}]|^2 \rangle$")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(0, 33, 4))
    ax.set_title("Frequency-Domain Error Spectrum vs $k_{max}$\n"
                 "($\\nu = 0.001$, 200 test samples, N = 64 grid)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================
# 6. Main pipeline
# ============================================================================
def main():
    t_start = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 78)
    print("Section 4.4 - Fourier-mode ablation (k_max sweep, nu = 0.001)")
    print("=" * 78)

    u0_all, ref_all, train_idx, test_idx = load_data()
    step = N_REF // N_GRID
    X_tr = torch.tensor(u0_all[train_idx][:, ::step], dtype=torch.float32)
    Y_tr = torch.tensor(ref_all[train_idx][:, ::step], dtype=torch.float32)
    u0_te = u0_all[test_idx]
    ref_te = ref_all[test_idx]
    print(f"Data: 800 train / 200 test, N = {N_GRID}, nu = {NU:g}")

    # ---------- train / load the six FNOs --------------------------------------
    models, losses, rows = {}, {}, []
    for kmax in KMAX_LIST:
        wpath = os.path.join(DATA_DIR, f"4.4_fno_kmax{kmax}_weights.pt")
        lpath = os.path.join(DATA_DIR, f"4.4_loss_kmax{kmax}.csv")
        model = FNO1d(modes=kmax, width=FNO_WIDTH, layers=FNO_LAYERS)
        if os.path.exists(wpath):
            model.load_state_dict(torch.load(wpath))
            print(f"k_max = {kmax}: loaded weights (resume).")
            loss_hist = (np.loadtxt(lpath) if os.path.exists(lpath) else None)
        else:
            print(f"Training FNO k_max = {kmax} ...")
            model, loss_hist = train_fno_with_history(kmax, X_tr, Y_tr)
            torch.save(model.state_dict(), wpath)
            np.savetxt(lpath, loss_hist, delimiter=",")
        models[kmax] = model
        losses[kmax] = loss_hist

    # ---------- evaluation ------------------------------------------------------
    print("Evaluation on the 200 test samples ...")
    errs = {}
    for kmax in KMAX_LIST:
        e, t_ms = evaluate_fno(models[kmax], u0_te, ref_te)
        errs[kmax] = e
        p_count = m42.count_params(models[kmax])
        conv = (convergence_epoch(losses[kmax]) if losses[kmax] is not None
                else None)
        lo, hi = bootstrap_ci(e)
        rows.append({"kmax": kmax, "mean_rel_l2": e.mean(), "ci_low": lo,
                     "ci_high": hi, "params": p_count, "time_ms": t_ms,
                     "converged_epoch": conv})
        print(f"  k_max={kmax:2d}: rel-L2 = {e.mean():.4e} [{lo:.4e}, {hi:.4e}] "
              f"| params = {p_count} | time = {t_ms:.3f} ms | "
              f"conv. epoch = {conv}")

    # ---------- error spectra ------------------------------------------------------
    print("Frequency-domain error spectra ...")
    spectra = {}
    for kmax in KMAX_LIST:
        spectra[kmax] = error_spectrum(models[kmax], u0_te, ref_te)
    k_axis = np.arange(N_GRID // 2 + 1)

    # ---------- statistical tests --------------------------------------------------
    print("Statistical tests ...")
    stat_rows = []
    for a, b in zip(KMAX_LIST[:-1], KMAX_LIST[1:]):
        dlo, dhi = bootstrap_ci_diff(errs[a], errs[b])
        w, p = wilcoxon_signed_rank(errs[a], errs[b])
        stat_rows.append({"kmax_a": a, "kmax_b": b,
                          "mean_diff": errs[a].mean() - errs[b].mean(),
                          "ci_low_diff": dlo, "ci_high_diff": dhi,
                          "wilcoxon_p": p})
        print(f"  {a} vs {b}: mean diff = {errs[a].mean() - errs[b].mean():.4e} "
              f"[{dlo:.4e}, {dhi:.4e}]  Wilcoxon p = {p:.4e}")
    jb8 = jarque_bera(errs[8])
    jb32 = jarque_bera(errs[32])
    print(f"  Jarque-Bera k_max=8 : JB = {jb8[0]:.3f}, p = {jb8[1]:.4e}")
    print(f"  Jarque-Bera k_max=32: JB = {jb32[0]:.3f}, p = {jb32[1]:.4e}")

    # ---------- acceptance criteria --------------------------------------------------
    e16_32 = errs[16].mean() - errs[32].mean()
    e8_16 = errs[8].mean() - errs[16].mean()
    mono = all(errs[KMAX_LIST[i]].mean() <= errs[KMAX_LIST[i - 1]].mean()
               for i in range(1, len(KMAX_LIST)))
    diminish = e16_32 < e8_16
    w832 = wilcoxon_signed_rank(errs[8], errs[32])[1]
    # spectral energy in the band k in (16, 32] (above k_max = 16)
    band_low16 = spectra[8][17:33].sum()
    band_high16 = spectra[32][17:33].sum()
    checks = {
        "monotonic non-increasing in k_max": (mono, 1.0),
        "diminishing returns (16->32 < 8->16)": (
            diminish, e8_16 - e16_32),
        "Wilcoxon 8 vs 32 p < 0.01": (w832 < 0.01, w832),
        "spectrum(8) > spectrum(32) on k in (16,32]": (
            band_low16 > band_high16, band_low16 / max(band_high16, 1e-30)),
    }
    print("-" * 78)
    for name, (ok, val) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val:.4f}")
    print("-" * 78)

    # ---------- save CSVs --------------------------------------------------------
    with open(os.path.join(DATA_DIR, "4.4_ablation_table.csv"), "w") as fh:
        fh.write("kmax,mean_rel_l2,ci_low,ci_high,params,time_ms,converged_epoch\n")
        for r in rows:
            fh.write(f"{r['kmax']},{r['mean_rel_l2']:.8e},{r['ci_low']:.8e},"
                     f"{r['ci_high']:.8e},{r['params']},{r['time_ms']:.6f},"
                     f"{r['converged_epoch']}\n")
    print(f"Saved: {os.path.join(DATA_DIR, '4.4_ablation_table.csv')}")

    with open(os.path.join(DATA_DIR, "4.4_error_spectrum.csv"), "w") as fh:
        fh.write("wavenumber," + ",".join(f"kmax{k}" for k in KMAX_LIST) + "\n")
        for j, k in enumerate(k_axis):
            fh.write(f"{k}," + ",".join(f"{spectra[km][j]:.8e}"
                                        for km in KMAX_LIST) + "\n")
    print(f"Saved: {os.path.join(DATA_DIR, '4.4_error_spectrum.csv')}")

    with open(os.path.join(DATA_DIR, "4.4_statistics.csv"), "w") as fh:
        fh.write("test,detail,statistic,p_value\n")
        for sr in stat_rows:
            fh.write(f"wilcoxon,{sr['kmax_a']}_vs_{sr['kmax_b']},"
                     f"{sr['mean_diff']:.8e},{sr['wilcoxon_p']:.8e}\n")
        fh.write(f"jarque_bera,kmax_8,{jb8[0]:.6e},{jb8[1]:.6e}\n")
        fh.write(f"jarque_bera,kmax_32,{jb32[0]:.6e},{jb32[1]:.6e}\n")
    print(f"Saved: {os.path.join(DATA_DIR, '4.4_statistics.csv')}")

    # ---------- figures ----------------------------------------------------------
    plot_ablation_curve(rows, os.path.join(FIG_DIR, "4.4_ablation_curve.png"))
    plot_error_spectrum(KMAX_LIST, spectra,
                        os.path.join(FIG_DIR, "4.4_error_spectrum.png"))

    # ---------- README ------------------------------------------------------------
    write_readme(rows, stat_rows, jb8, jb32, checks)

    print("=" * 78)
    print(f"Done. Total wall-clock time: {time.time() - t_start:.1f} s")
    print("=" * 78)


def write_readme(rows, stat_rows, jb8, jb32, checks):
    """Write 4.4_README.md (English)."""
    lines = [
        "# Section 4.4 - Fourier-Mode Ablation (k_max sweep)",
        "",
        "Goal: at the nu = 0.001 sharp-feature scenario, study the effect of",
        "the number of Fourier modes k_max on the accuracy / cost of the FNO",
        "and the frequency-domain mechanism.",
        "",
        "## Setup",
        "",
        "- Dataset: nu = 0.001 Burgers GRF dataset (Section 4.3), 800 train /",
        "  200 test samples, N = 64.",
        "- Six FNOs (L = 4, d_v = 64) with k_max in {8, 12, 16, 20, 24, 32}.",
        "- Identical hyper-parameters: Adam (lr = 1e-3) + cosine schedule,",
        "  200 epochs, batch 32, MSE loss.",
        "",
        "## Ablation table (data/4.4_ablation_table.csv)",
        "",
        "| k_max | mean rel-L2 [95% CI] | params | time (ms) | conv. epoch |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['kmax']} | {r['mean_rel_l2']:.3e} "
            f"[{r['ci_low']:.3e}, {r['ci_high']:.3e}] | {r['params']} | "
            f"{r['time_ms']:.3f} | {r['converged_epoch']} |")
    lines += [
        "",
        "## Statistical tests (data/4.4_statistics.csv)",
        "",
    ]
    for sr in stat_rows:
        lines.append(
            f"- Wilcoxon signed-rank {sr['kmax_a']} vs {sr['kmax_b']}: "
            f"p = {sr['wilcoxon_p']:.4e} "
            f"(bootstrap mean-diff CI [{sr['ci_low_diff']:.3e}, "
            f"{sr['ci_high_diff']:.3e}])")
    lines += [
        f"- Jarque-Bera k_max = 8 : JB = {jb8[0]:.3f}, p = {jb8[1]:.4e}",
        f"- Jarque-Bera k_max = 32: JB = {jb32[0]:.3f}, p = {jb32[1]:.4e}",
        "",
        "## Acceptance criteria",
        "",
    ]
    for name, (ok, val) in checks.items():
        lines.append(f"- **{'PASS' if ok else 'FAIL'}** {name}: {val:.4f}")
    lines += [
        "",
        "## Files",
        "",
        "- `data/4.4_ablation_table.csv` - error / params / time / epoch per "
        "k_max",
        "- `data/4.4_error_spectrum.csv` - E_k per k_max",
        "- `data/4.4_statistics.csv` - Wilcoxon and Jarque-Bera results",
        "- `data/4.4_fno_kmax{8..32}_weights.pt` - trained weights",
        "- `figures/4.4_ablation_curve.png` - error curve + params/time twin "
        "axis",
        "- `figures/4.4_error_spectrum.png` - log-scale error spectra",
        "",
        "## Reproduce",
        "",
        "    .venv_torch\\Scripts\\python.exe 4.4_mode_ablation.py",
    ]
    with open(os.path.join(BASE_DIR, "4.4_README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()





