"""
4.2_resolution_invariance.py
============================

Section 4.2: resolution-invariance (zero-shot super-resolution) verification.

Goal: demonstrate that the Fourier Neural Operator (FNO) is grid-invariant
while a parameter-matched U-Net is grid-sensitive.

Procedure
---------
  1. Train an FNO (L = 4, k_max = 16, d_v = 64) and a U-Net (matched
     parameter count, |params| within 10%) on the nu = 0.01 dataset at
     resolution N = 64 (800 training samples generated in Section 4.1.1).
  2. WITHOUT any fine-tuning, evaluate both models on the 200 test samples
     at six resolutions eval_N = {64, 128, 256, 512, 1024, 2048} (zero-shot
     super-resolution), comparing the predictions against the N = 2048
     spectral reference downsampled to each resolution.
  3. Record per-resolution mean relative L2 error (with bootstrap 95% CI)
     and per-sample inference time.

Acceptance criteria
-------------------
  * FNO: error increase from N = 64 to N = 2048 <= 20% (flat curve).
  * U-Net: error at N = 2048 >= 5x the FNO error at the same resolution.
  * Parameter count difference between the two models <= 10%.

Outputs
-------
  data/4.2_resolution_table.csv
  data/4.2_fno_weights.pt , data/4.2_unet_weights.pt
  figures/4.2_resolution_invariance.png
  figures/4.2_sample_predictions.png
  4.2_README.md

Every file carries a "4.2" prefix; all comments and figure labels are in
English.
"""

import os
import time

import numpy as np
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

NU = 0.01                 # viscosity of the benchmark scenario
N_REF = 2048              # reference resolution (spectral ground truth)
N_TRAIN_GRID = 64         # training resolution (FNO and U-Net)
N_SAMPLES = 1000
N_TRAIN = 800
N_TEST = 200
EVAL_RES = [64, 128, 256, 512, 1024, 2048]     # zero-shot evaluation grids

# FNO architecture (Table 3.1)
FNO_MODES = 16
FNO_WIDTH = 64
FNO_LAYERS = 4

# U-Net architecture (channel widths chosen to match the FNO parameter count
# within 10%)
UNET_C1 = 120
UNET_C2 = 240

EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
BOOT_ITER = 1000          # bootstrap resamples for the 95% confidence interval
BOOT_SEED = 7

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")

# ============================================================================
# 2. Data loading (reuse the Section 4.1.1 GRF dataset)
# ============================================================================
def load_data():
    """Load initial conditions and spectral reference solutions.

    Returns
    -------
    u0_all, ref_all : np.ndarray, (1000, 2048)
        Initial conditions and reference solutions u(x, T) on the 2048 grid.
    train_idx, test_idx : np.ndarray
        The 800/200 train/test partition (identical to Sections 4.1.1/4.1.3).
    """
    u0_all = np.loadtxt(os.path.join(DATA_DIR, "grf_initial_conditions.csv"),
                        delimiter=",", skiprows=1)
    train_sol = np.loadtxt(os.path.join(DATA_DIR, "train", "solutions_final.csv"),
                           delimiter=",", skiprows=1)
    test_sol = np.loadtxt(os.path.join(DATA_DIR, "test", "solutions_final.csv"),
                          delimiter=",", skiprows=1)
    with open(os.path.join(DATA_DIR, "train_test_split.csv")) as fh:
        lines = fh.read().splitlines()[1:]
    train_idx = np.array([i for i, ln in enumerate(lines)
                          if ln.split(",")[1] == "train"])
    test_idx = np.array([i for i, ln in enumerate(lines)
                         if ln.split(",")[1] == "test"])
    ref_all = np.empty_like(u0_all)
    ref_all[train_idx] = train_sol
    ref_all[test_idx] = test_sol
    return u0_all, ref_all, train_idx, test_idx

# ============================================================================
# 3. Metric helpers
# ============================================================================
def rel_l2_batch(pred, ref):
    """Per-sample relative L2 error, pred/ref shape (n, N)."""
    return np.linalg.norm(pred - ref, axis=1) / np.linalg.norm(ref, axis=1)


def bootstrap_ci(errs, n_iter=BOOT_ITER, alpha=0.05, seed=BOOT_SEED):
    """95% bootstrap confidence interval of the mean relative L2 error."""
    rng = np.random.default_rng(seed)
    means = np.empty(n_iter)
    for i in range(n_iter):
        means[i] = rng.choice(errs, size=len(errs), replace=True).mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


# ============================================================================
# 4. Models
# ============================================================================
class SpectralConv1d(nn.Module):
    """1-D Fourier spectral convolution layer (FNO).

    Applies learnable complex weights to the first ``modes`` Fourier
    coefficients of the input and transforms back to physical space.  The
    DC (k = 0) mode uses a real weight.  The layer is resolution-invariant:
    it acts on whatever grid size the input has.
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        # complex weights for modes k = 1 .. modes
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes,
                                    dtype=torch.cfloat))
        # real weights for the DC mode k = 0
        self.weights0 = nn.Parameter(self.scale * torch.rand(in_channels,
                                                             out_channels))

    def forward(self, x):
        # x: (batch, in_channels, N)
        batch = x.shape[0]
        x_ft = torch.fft.rfft(x)                    # (batch, in, N//2+1)
        out_ft = torch.zeros(batch, self.out_channels, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, 1:self.modes + 1] = torch.einsum(
            "bix,iox->box", x_ft[:, :, 1:self.modes + 1], self.weights)
        out_ft[:, :, 0] = torch.einsum(
            "bi,io->bo", x_ft[:, :, 0].real, self.weights0)
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    """Fourier Neural Operator for the 1-D Burgers solution operator.

    Architecture (Chapter 3, Table 3.1): L = 4 Fourier layers, k_max = 16,
    hidden width d_v = 64, GELU activation.  Input u_0 at any resolution,
    output u(x, T) at the same resolution.
    """

    def __init__(self, modes=FNO_MODES, width=FNO_WIDTH, layers=FNO_LAYERS):
        super().__init__()
        self.modes = modes
        self.width = width
        self.layers = layers
        self.fc0 = nn.Linear(1, width)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.local = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, N)
        x = x.unsqueeze(-1)                         # (batch, N, 1)
        x = self.fc0(x)                             # (batch, N, width)
        x = x.permute(0, 2, 1)                      # (batch, width, N)
        for sp, lo in zip(self.spectral, self.local):
            x = F.gelu(sp(x) + lo(x))
        x = x.permute(0, 2, 1)                      # (batch, N, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                             # (batch, N, 1)
        return x.squeeze(-1)


class UNet1d(nn.Module):
    """1-D U-Net (grid-sensitive baseline).

    Classic encoder-decoder with max-pooling downsampling, nearest-neighbour
    upsampling and skip connections.  The convolution kernels are shared
    across resolutions in grid units, so the receptive field does NOT adapt
    when the resolution changes -- this is what makes the U-Net grid
    sensitive (the accuracy degrades away from the training resolution).

    Channel widths (c1, c2) are chosen so that the parameter count matches
    the FNO within 10%.
    """

    def __init__(self, in_ch=1, out_ch=1, c1=UNET_C1, c2=UNET_C2):
        super().__init__()
        self.conv_in = nn.Conv1d(in_ch, c1, 3, padding=1)
        self.enc1 = nn.Conv1d(c1, c1, 3, padding=1)
        self.enc2 = nn.Conv1d(c1, c2, 3, padding=1)
        self.bottle = nn.Conv1d(c2, c2, 3, padding=1)
        self.up2 = nn.Conv1d(2 * c2, c1, 3, padding=1)
        self.up1 = nn.Conv1d(2 * c1, c1, 3, padding=1)
        self.conv_out = nn.Conv1d(c1, out_ch, 1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        # x: (batch, N)  ->  (batch, 1, N)
        x = x.unsqueeze(1)
        x = F.gelu(self.conv_in(x))
        s1 = F.gelu(self.enc1(x))                   # resolution N
        x = self.pool(s1)                           # N/2
        s2 = F.gelu(self.enc2(x))                   # N/2
        x = self.pool(s2)                           # N/4
        x = F.gelu(self.bottle(x))                  # N/4
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.gelu(self.up2(torch.cat([x, s2], dim=1)))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.gelu(self.up1(torch.cat([x, s1], dim=1)))
        return self.conv_out(x).squeeze(1)

def count_params(model):
    """Count trainable parameters as stored float values.

    Complex-valued tensors (used by the FNO spectral weights) are counted
    twice because each complex number stores two real floats.  This makes
    the comparison with the real-valued U-Net fair.
    """
    total = 0
    for p in model.parameters():
        total += p.numel() * (2 if p.dtype.is_complex else 1)
    return total


# ============================================================================
# 5. Training and evaluation
# ============================================================================
def train_model(model, X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                name="model"):
    """Train a model on the N = 64 dataset with Adam + cosine schedule.

    X, Y : torch tensors (n_train, N_TRAIN_GRID)
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.shape[0]
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X[idx])
            loss = F.mse_loss(pred, Y[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        sched.step()
        if ep % 25 == 0:
            print(f"  [{name}] epoch {ep:3d}/{epochs}  "
                  f"loss = {ep_loss / n:.6e}  "
                  f"({time.time() - t0:.0f} s)")
    return model


def evaluate_resolutions(model, u0_test, ref_test, resolutions,
                         model_name, params_count):
    """Zero-shot evaluation at every resolution (no fine-tuning).

    For each resolution N the model receives u_0 sampled on the N-point grid
    and outputs a prediction on the same grid; the N = 2048 reference is
    downsampled to N for the comparison.

    Returns a list of dicts with the per-resolution results.
    """
    rows = []
    for N in resolutions:
        step = N_REF // N
        X = torch.tensor(u0_test[:, ::step], dtype=torch.float32)   # (n, N)
        y_ref = ref_test[:, ::step]
        model.eval()
        with torch.no_grad():
            t0 = time.time()
            pred = model(X)
            time_ms = (time.time() - t0) / len(X) * 1000.0
        errs = rel_l2_batch(pred.numpy(), y_ref)
        lo, hi = bootstrap_ci(errs)
        rows.append({
            "method": model_name, "resolution": N,
            "mean_rel_l2": errs.mean(), "ci_low": lo, "ci_high": hi,
            "time_ms": time_ms, "params": params_count,
        })
        print(f"  [{model_name}] N={N:5d}: mean rel-L2 = {errs.mean():.4e} "
              f"[{lo:.4e}, {hi:.4e}]  time = {time_ms:.3f} ms")
    return rows


# ============================================================================
# 6. Figures (PNG, English labels)
# ============================================================================
def plot_resolution_invariance(rows, path):
    """Error-vs-resolution curves (FNO should be flat, U-Net steep)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for method, color in (("FNO", "#1f77b4"), ("U-Net", "#d62728")):
        sub = [r for r in rows if r["method"] == method]
        sub = sorted(sub, key=lambda r: r["resolution"])
        x = [r["resolution"] for r in sub]
        y = [r["mean_rel_l2"] for r in sub]
        lo = [r["ci_low"] for r in sub]
        hi = [r["ci_high"] for r in sub]
        ax.plot(x, y, "o-", lw=1.8, color=color, label=f"{method} (zero-shot)")
        ax.fill_between(x, lo, hi, color=color, alpha=0.2,
                        label=f"{method} 95% CI")
    ax.axvline(64, ls=":", color="gray", lw=1.2)
    ax.text(66, ax.get_ylim()[1] * 0.95, "training resolution", fontsize=8,
            color="gray")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([64, 128, 256, 512, 1024, 2048])
    ax.set_xticklabels(["64", "128", "256", "512", "1024", "2048"])
    ax.set_xlabel("Evaluation resolution  N")
    ax.set_ylabel("Mean relative L2 error")
    ax.set_title("Zero-shot Super-resolution: Grid Invariance Test\n"
                 r"(Burgers, $\nu = 0.01$, 200 test samples)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sample_predictions(u0_test, ref_test, fno_model, unet_model,
                            sample_ids, path):
    """Prediction snapshots at N = 2048 for a few test samples."""
    n = len(sample_ids)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 6.0),
                             sharex=True, sharey=False)
    x = np.arange(N_REF) / N_REF
    with torch.no_grad():
        for j, s in enumerate(sample_ids):
            X = torch.tensor(u0_test[s], dtype=torch.float32).unsqueeze(0)  # (1, N)
            pred_fno = fno_model(X).squeeze(0).numpy()
            pred_unet = unet_model(X).squeeze(0).numpy()

            # top row: full field
            ax = axes[0, j]
            ax.plot(x, ref_test[s], lw=1.0, color="black",
                    label="Reference (N=2048)")
            ax.plot(x, pred_fno, lw=1.0, color="#1f77b4", label="FNO (zero-shot)")
            ax.plot(x, pred_unet, lw=1.0, color="#d62728", label="U-Net (zero-shot)")
            ax.set_title(f"Test sample {s}")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel(r"$u(x,T)$")
            ax.legend(fontsize=7)

            # bottom row: zoomed region x in [0.35, 0.65]
            ax = axes[1, j]
            m = (x >= 0.35) & (x <= 0.65)
            ax.plot(x[m], ref_test[s][m], lw=1.2, color="black")
            ax.plot(x[m], pred_fno[m], lw=1.2, color="#1f77b4")
            ax.plot(x[m], pred_unet[m], lw=1.2, color="#d62728")
            ax.set_xlim(0.35, 0.65)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel(r"$u(x,T)$")
            ax.set_xlabel("$x$")
    fig.suptitle("Zero-shot Predictions at N = 2048 (models trained at N = 64)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================
# 7. Main pipeline
# ============================================================================
def main():
    t_start = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 78)
    print("Section 4.2 - resolution-invariance (zero-shot super-resolution)")
    print("=" * 78)

    # --- data -------------------------------------------------------------------
    u0_all, ref_all, train_idx, test_idx = load_data()
    step = N_REF // N_TRAIN_GRID
    X_tr = torch.tensor(u0_all[train_idx][:, ::step], dtype=torch.float32)
    Y_tr = torch.tensor(ref_all[train_idx][:, ::step], dtype=torch.float32)
    u0_te = u0_all[test_idx]
    ref_te = ref_all[test_idx]
    print(f"Train pairs: {X_tr.shape[0]} x N={N_TRAIN_GRID}; "
          f"test samples: {u0_te.shape[0]}")

    # --- models and parameter counts ----------------------------------------------
    fno = FNO1d()
    unet = UNet1d()
    p_fno = count_params(fno)
    p_unet = count_params(unet)
    diff = abs(p_fno - p_unet) / p_fno
    print(f"FNO params  : {p_fno} (stored float values)")
    print(f"U-Net params: {p_unet}  (relative difference {diff:.2%})")

    # --- training at N = 64 -------------------------------------------------------
    fno_path = os.path.join(DATA_DIR, "4.2_fno_weights.pt")
    unet_path = os.path.join(DATA_DIR, "4.2_unet_weights.pt")
    if os.path.exists(fno_path) and os.path.exists(unet_path):
        fno.load_state_dict(torch.load(fno_path))
        unet.load_state_dict(torch.load(unet_path))
        print("Found saved weights -> skipped training (resume mode).")
    else:
        print("Training FNO at N = 64 ...")
        train_model(fno, X_tr, Y_tr, name="FNO")
        print("Training U-Net at N = 64 ...")
        train_model(unet, X_tr, Y_tr, name="U-Net")

    # --- zero-shot evaluation at six resolutions ----------------------------------
    print("Zero-shot evaluation (no fine-tuning) ...")
    rows = []
    rows += evaluate_resolutions(fno, u0_te, ref_te, EVAL_RES, "FNO", p_fno)
    rows += evaluate_resolutions(unet, u0_te, ref_te, EVAL_RES, "U-Net", p_unet)

    # --- acceptance criteria --------------------------------------------------------
    e64 = {r["method"]: r for r in rows if r["resolution"] == 64}
    e2048 = {r["method"]: r for r in rows if r["resolution"] == 2048}
    fno_ratio = e2048["FNO"]["mean_rel_l2"] / e64["FNO"]["mean_rel_l2"]
    unet_vs_fno = (e2048["U-Net"]["mean_rel_l2"]
                   / e2048["FNO"]["mean_rel_l2"])
    checks = {
        "FNO error increase N64->N2048 <= 20%": (fno_ratio <= 1.20, fno_ratio),
        "U-Net(N=2048) >= 5x FNO(N=2048)": (unet_vs_fno >= 5.0, unet_vs_fno),
        "|params| difference <= 10%": (diff <= 0.10, diff),
    }
    print("-" * 78)
    for name, (ok, val) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val:.4f}")
    print("-" * 78)

    # --- save CSV ---------------------------------------------------------------------
    csv_path = os.path.join(DATA_DIR, "4.2_resolution_table.csv")
    with open(csv_path, "w") as fh:
        fh.write("method,resolution,mean_rel_l2,ci_low,ci_high,time_ms,params\n")
        for r in rows:
            fh.write(f"{r['method']},{r['resolution']},{r['mean_rel_l2']:.8e},"
                     f"{r['ci_low']:.8e},{r['ci_high']:.8e},"
                     f"{r['time_ms']:.6f},{r['params']}\n")
    print(f"Saved: {csv_path}")

    # --- save model weights ----------------------------------------------------------
    torch.save(fno.state_dict(), os.path.join(DATA_DIR, "4.2_fno_weights.pt"))
    torch.save(unet.state_dict(), os.path.join(DATA_DIR, "4.2_unet_weights.pt"))

    # --- figures ----------------------------------------------------------------------
    plot_resolution_invariance(rows,
                               os.path.join(FIG_DIR,
                                            "4.2_resolution_invariance.png"))
    plot_sample_predictions(u0_te, ref_te, fno, unet, [0, 1],
                            os.path.join(FIG_DIR, "4.2_sample_predictions.png"))

    # --- write the README ---------------------------------------------------------------
    write_readme(rows, checks, p_fno, p_unet)

    print("=" * 78)
    print(f"Done. Total wall-clock time: {time.time() - t_start:.1f} s")
    print("=" * 78)


def write_readme(rows, checks, p_fno, p_unet):
    """Write 4.2_README.md (English)."""
    lines = [
        "# Section 4.2 - Resolution Invariance (Zero-shot Super-resolution)",
        "",
        "Goal: demonstrate that the FNO is **grid-invariant** while a",
        "parameter-matched **U-Net** is **grid-sensitive**.",
        "",
        "## Setup",
        "",
        "- Dataset: nu = 0.01 Burgers GRF dataset (Section 4.1.1).",
        "- Both models trained on 800 training samples at N = 64 (no",
        "  normalization), Adam (lr = 1e-3) + cosine schedule, 200 epochs.",
        "- Zero-shot evaluation on the 200 test samples at",
        "  N in {64, 128, 256, 512, 1024, 2048} without any fine-tuning.",
        "- Reference: N = 2048 spectral solution downsampled to each grid.",
        f"- Parameters: FNO = {p_fno}, U-Net = {p_unet} "
        f"(relative difference {abs(p_fno - p_unet) / p_fno:.2%}).",
        "",
        "## Per-resolution results (data/4.2_resolution_table.csv)",
        "",
        "| method | resolution | mean rel-L2 | 95% CI | time (ms) |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["method"], r["resolution"])):
        lines.append(
            f"| {r['method']} | {r['resolution']} | "
            f"{r['mean_rel_l2']:.3e} | "
            f"[{r['ci_low']:.3e}, {r['ci_high']:.3e}] | "
            f"{r['time_ms']:.3f} |")
    lines += [
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
        "- `data/4.2_resolution_table.csv` - method x resolution x error x CI x "
        "time x params",
        "- `data/4.2_fno_weights.pt`, `data/4.2_unet_weights.pt` - trained weights",
        "- `figures/4.2_resolution_invariance.png` - error vs resolution curves",
        "- `figures/4.2_sample_predictions.png` - zero-shot predictions at N = 2048",
        "",
        "## Reproduce",
        "",
        "    .venv_torch\\Scripts\\python.exe 4.2_resolution_invariance.py",
    ]
    with open(os.path.join(BASE_DIR, "4.2_README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()





