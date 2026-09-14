"""
generate_burgers_dataset.py
===========================

Generation of a dataset for learning the solution operator of the 1D viscous
Burgers equation

        u_t + u u_x = nu * u_xx,        x in [0, 1)   (periodic boundary),
        u(0, x) = u_0(x)                              (random-field initial data).

This script implements Section 4.1.1: initial-condition generation based on a
Gaussian Random Field (GRF).

    * All initial conditions u_0(x) are drawn from a zero-mean Gaussian Random
      Field with covariance operator

            C = (-Delta + tau^2)^(-alpha),          tau = 7,  alpha = 2.

      Under periodic boundary conditions the eigenfunctions of C are the
      Fourier modes exp(2 pi i k x) with eigenvalues

            lambda_k = (4 pi^2 k^2 + tau^2)^(-alpha).

      A Karhunen-Loeve (KL) expansion truncated at K_KL = 128 modes is used:

            u_0(x) ~= sum_{|k| <= K_KL} sqrt(lambda_k) xi_k exp(2 pi i k x),
            xi_k i.i.d. N(0, 1)   (complex draws with Hermitian symmetry,
                                   which makes u_0 real valued).

    * N_SAMPLES = 1000 initial conditions are generated.  For every sample the
      Burgers equation is solved on the very fine grid N_GT = 2048 with a
      Fourier pseudo-spectral method (3/2 dealiasing) and a 4th-order
      Runge-Kutta integrator (integrating-factor variant, IF-RK4) up to the
      target time T = 1.0.  These are the high-accuracy reference
      (ground-truth) solutions.

    * The data are split into 800 training / 200 test samples.

All sampled data are stored as CSV files under ./data and every figure is
saved as a PNG under ./figures.  All comments and figure labels are written
in English.
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend (figures are only saved)
import matplotlib.pyplot as plt

# ============================================================================
# 1. Configuration
# ============================================================================
# --- GRF parameters (Section 4.1.1) -----------------------------------------
K_KL = 128          # KL expansion truncation order
TAU = 7.0           # GRF regularization parameter
ALPHA = 2.0         # GRF smoothness parameter

# --- Spatial / temporal setup ------------------------------------------------
N_GT = 2048         # number of fine-grid points (x_j = j / N_GT, j = 0..N_GT-1)
N_SAMPLES = 1000    # number of GRF initial-condition samples
T_MAX = 1.0         # target final time T
N_SNAP = 101        # number of stored time snapshots (including t = 0)
DT = 1e-3           # IF-RK4 time step (RK4 temporal error ~ O(dt^4) ~ 1e-11)
NU = 0.01           # viscosity of the Burgers equation

# --- Dataset split -----------------------------------------------------------
N_TRAIN = 800       # number of training samples
N_TEST = 200        # number of test samples
SEED = 42           # random seed (reproducible split and draws)
N_TRAJ_EXAMPLES = 3 # number of full trajectories exported as per-sample CSVs

# --- Output folders -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAJ_DIR = os.path.join(DATA_DIR, "trajectories")

# ============================================================================
# 2. Helper: CSV export
# ============================================================================
def save_matrix_csv(path, arr, x):
    """Save a 2-D array as a CSV file.

    Rows = samples (or time snapshots), columns = spatial grid points.
    The first row is a header holding the physical coordinates x of the
    periodic grid on [0, 1).  All values are written in scientific format.
    """
    header = ",".join([f"{xv:.6f}" for xv in x])
    np.savetxt(path, arr, delimiter=",", fmt="%.8e", header=header, comments="")

# ============================================================================
# 3. Gaussian Random Field (GRF): Karhunen-Loeve expansion
# ============================================================================
def grf_eigenvalues(n, k_kl, tau, alpha):
    """Eigenvalues of C = (-Delta + tau^2)^(-alpha) on the FFT wavenumber grid.

    Under periodic boundary conditions the eigenfunctions of C are the Fourier
    modes exp(2 pi i k x) with eigenvalues

        lambda_k = (4 pi^2 k^2 + tau^2)^(-alpha).

    Modes with |k| > K_KL are truncated (their eigenvalue is set to zero).

    Parameters
    ----------
    n : int
        Number of grid points (must be even).
    k_kl : int
        KL truncation order.
    tau, alpha : float
        GRF regularization and smoothness parameters.

    Returns
    -------
    k : np.ndarray, shape (n,)
        Integer wavenumbers in FFT ordering (-n/2, ..., n/2 - 1).
    lam : np.ndarray, shape (n,)
        GRF eigenvalues (zero for the truncated modes).
    """
    k = np.fft.fftfreq(n, d=1.0 / n)                 # integer wavenumbers
    lam = (4.0 * np.pi**2 * k**2 + tau**2) ** (-alpha)
    lam[np.abs(k) > k_kl] = 0.0                      # KL truncation
    return k, lam


def draw_grf_sample(k, lam, n, rng):
    """Draw a single real-valued GRF initial condition u_0 on the grid.

    Truncated Karhunen-Loeve expansion

        u_0(x) = sum_{|k| <= K_KL} sqrt(lambda_k) xi_k exp(2 pi i k x),

    where xi_k are complex standard normals (E|xi_k|^2 = 1).  The draws are
    imposed to be Hermitian symmetric, xi_{-k} = conj(xi_k), which guarantees
    that u_0 is real valued on the periodic grid.

    Parameters
    ----------
    k : np.ndarray
        Wavenumbers from ``grf_eigenvalues``.
    lam : np.ndarray
        Eigenvalues from ``grf_eigenvalues``.
    n : int
        Number of grid points.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    u0 : np.ndarray, shape (n,)
        The sampled initial condition u_0(x_j), x_j = j / n.
    """
    xi = np.zeros(n, dtype=complex)
    xi[0] = rng.standard_normal()                    # k = 0 mode (real xi_0)
    pos = np.where(k > 0)[0]                         # indices of positive modes
    # xi_k = (a + i b) / sqrt(2), a, b ~ N(0, 1)  =>  E|xi_k|^2 = 1
    xi[pos] = (rng.standard_normal(len(pos))
               + 1j * rng.standard_normal(len(pos))) / np.sqrt(2.0)
    xi[n - pos] = np.conj(xi[pos])                   # Hermitian symmetry
    if n % 2 == 0 and abs(k[n // 2]) <= K_KL:
        xi[n // 2] = rng.standard_normal()           # Nyquist mode (real)

    coeff = np.sqrt(lam) * xi                        # KL coefficients c_k
    # u0(x_j) = sum_k c_k exp(2 pi i k x_j) = ifft(n * c)[j]
    u0 = np.fft.ifft(n * coeff).real
    return u0


# ============================================================================
# 4. Burgers equation: Fourier pseudo-spectral solver (IF-RK4)
# ============================================================================
def burgers_nonlinear(X, w, n):
    """Dealiased pseudo-spectral advection term of the Burgers RHS.

    Given the FFT coefficients X of u (so that u = ifft(X)), this returns the
    FFT coefficients of  - u u_x  =  - (1/2) (u^2)_x.

    A 3/2 zero-padding rule is applied so that the quadratic product u^2 is
    computed free of aliasing errors.
    """
    m = 3 * n // 2                                   # padded grid size (3/2 rule)
    Xpad = np.zeros(m, dtype=complex)
    Xpad[: n // 2] = X[: n // 2]                     # positive modes (incl. DC)
    Xpad[n:] = X[n // 2:]                            # negative modes (incl. Nyq.)
    # NOTE: numpy's ifft applies 1/m (m = padded length), which would scale the
    # physical values by n/m.  The factor m/n restores the exact field values
    # on the padded grid (so that u(x) matches the original n-point grid).
    u = np.fft.ifft(Xpad) * (m / n)                  # u on the padded grid
    u2 = u * u                                       # quadratic product in x
    X2 = np.fft.fft(u2) * (n / m)                    # back to the n-FFT convention
    X2r = np.empty_like(X)
    X2r[: n // 2] = X2[: n // 2]                     # truncate to the n-grid
    X2r[n // 2:] = X2[n:]
    X2r[n // 2] = 0.0                                # drop the Nyquist mode
    return -0.5j * w * X2r                           # FFT of -(1/2)(u^2)_x


def solve_burgers_ifrk4(u0, k, w, nu, dt, n_steps, n_snap, n):
    """Integrate the viscous Burgers equation up to T = n_steps * dt.

    Fourier pseudo-spectral space discretization combined with the
    integrating-factor Runge-Kutta 4 scheme (Cox & Matthews, 2002).  The
    linear diffusion is treated exactly through the integrating factor

        V(t) = exp(nu * w^2 * t) * X(t),

    which removes the stiffness, while the advection term is advanced with a
    classical 4th-order Runge-Kutta method (dealiased, see
    ``burgers_nonlinear``).

    Returns
    -------
    snapshots : np.ndarray, shape (n_snap, n)
        The solution sampled at n_snap uniformly spaced times (t = 0 .. T).
    """
    X = np.fft.fft(u0)                               # initial spectrum
    e_half = np.exp(-0.5 * nu * w * w * dt)          # IF factor, half step
    e_full = e_half * e_half                         # IF factor, full step

    snap_stride = n_steps / (n_snap - 1)
    snapshots = np.empty((n_snap, n))
    snapshots[0] = u0
    next_snap = 1

    for step in range(1, n_steps + 1):
        # --- four RK4 stages of the advection term (IF formulation) ---------
        n1 = burgers_nonlinear(X, w, n)
        b = e_half * (X + 0.5 * dt * n1)
        n2 = burgers_nonlinear(b, w, n)
        c = e_half * X + 0.5 * dt * n2
        n3 = burgers_nonlinear(c, w, n)
        d = e_full * X + dt * n3
        n4 = burgers_nonlinear(d, w, n)
        X = (e_full * X
             + (dt / 6.0) * (e_full * n1 + 2.0 * e_half * (n2 + n3) + n4))

        # enforce Hermitian symmetry so that u stays real to machine precision
        X[0] = X[0].real
        X[n // 2] = X[n // 2].real

        # --- store the snapshot if the time level has been crossed ----------
        if step >= next_snap * snap_stride - 1e-12:
            snapshots[next_snap] = np.fft.ifft(X).real
            next_snap += 1
            if next_snap >= n_snap:
                break
    return snapshots


# ============================================================================
# 5. Plotting helpers (all figures saved as PNG, English labels)
# ============================================================================
def plot_eigenvalue_spectrum(k, lam, path):
    """Figure: eigenvalue spectrum of the GRF covariance operator C."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    mask = np.abs(k) <= K_KL
    ax.semilogy(k[mask], lam[mask], ".-", markersize=3.0, color="#1f77b4")
    ax.set_xlabel(r"Wavenumber $k$")
    ax.set_ylabel(r"Eigenvalue $\lambda_k = (4\pi^2 k^2 + \tau^2)^{-\alpha}$")
    ax.set_title(r"Eigenvalue Spectrum of $C = (-\Delta + \tau^2)^{-\alpha}$, "
                 r"$\tau = 7$, $\alpha = 2$, $K_{\mathrm{KL}} = 128$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grf_samples(x, u0, n_show, path):
    """Figure: a few sample GRF initial conditions u_0(x)."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for s in range(n_show):
        ax.plot(x, u0[s], lw=1.2, label=f"Sample {s}")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$u_0(x)$")
    ax.set_title(r"Sample Initial Conditions from the GRF "
                 r"($K_{\mathrm{KL}} = 128$, $\tau = 7$, $\alpha = 2$)")
    ax.legend(ncol=3, fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_burgers_evolution(x, t_snap, sol, sample_id, path):
    """Figure: space-time diagram of the Burgers solution of one sample."""
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    pc = ax.pcolormesh(x, t_snap, sol, shading="auto", cmap="viridis")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_title(f"Burgers Equation Solution $u(x,t)$, Sample {sample_id} "
                 rf"($\nu = {NU:g}$, $N_{{\mathrm{{GT}}}} = {N_GT}$, IF-RK4)")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label(r"$u(x,t)$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_initial_vs_final(x, u0, uT, sample_ids, path):
    """Figure: initial condition vs. solution at the final time T."""
    n = len(sample_ids)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 3.4),
                             sharex=True, sharey=True)
    for ax, s in zip(axes, sample_ids):
        ax.plot(x, u0[s], lw=1.4, label=r"$u_0(x)$", color="#1f77b4")
        ax.plot(x, uT[s], lw=1.4, ls="--", label=r"$u(x,T)$", color="#d62728")
        ax.set_title(f"Sample {s}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$u$")
    fig.suptitle(f"Initial Condition vs. Solution at Final Time $T = {T_MAX:g}$")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_split(x, u0_all, idx_train, idx_test, path):
    """Figure: train/test partition and its statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # left panel: partition labels over the sample index
    part = np.zeros(N_SAMPLES, dtype=int)
    part[idx_test] = 1
    colors = np.where(part == 0, "#1f77b4", "#d62728")
    axes[0].scatter(np.arange(N_SAMPLES), part, s=9, c=colors, edgecolors="none")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["train", "test"])
    axes[0].set_xlabel("Sample index")
    axes[0].set_title(f"Train / Test Partition ({N_TRAIN} / {N_TEST})")
    axes[0].grid(True, alpha=0.3)

    # right panel: mean +/- std of u_0 for train and test subsets
    mu_tr, sd_tr = u0_all[idx_train].mean(0), u0_all[idx_train].std(0)
    mu_te, sd_te = u0_all[idx_test].mean(0), u0_all[idx_test].std(0)
    axes[1].plot(x, mu_tr, color="#1f77b4", label="Train mean")
    axes[1].fill_between(x, mu_tr - sd_tr, mu_tr + sd_tr,
                         color="#1f77b4", alpha=0.25, label="Train $\\pm$ 1 std")
    axes[1].plot(x, mu_te, ls="--", color="#d62728", label="Test mean")
    axes[1].fill_between(x, mu_te - sd_te, mu_te + sd_te,
                         color="#d62728", alpha=0.25, label="Test $\\pm$ 1 std")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel(r"$u_0(x)$")
    axes[1].set_title("Statistics of the Train vs. Test Initial Conditions")
    axes[1].legend(fontsize=7, framealpha=0.9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ============================================================================
# 6. Validation helpers (sanity checks of the solver)
# ============================================================================
def validate_linear_limit(u0, k, w, n_steps):
    """Check the solver against the exact linear diffusion limit.

    For a tiny-amplitude field the advection term is negligible and
    X(T) ~= X(0) * exp(-nu w^2 T) holds exactly in Fourier space.
    """
    tiny = u0 * 1e-6
    X0 = np.fft.fft(tiny)
    X_lin = X0 * np.exp(-NU * w * w * T_MAX)
    u_lin = np.fft.ifft(X_lin).real
    sol = solve_burgers_ifrk4(tiny, k, w, NU, DT, n_steps, 2, N_GT)
    err = np.abs(sol[-1] - u_lin).max()
    scale = np.abs(u_lin).max()
    return err, scale


def validate_temporal_convergence(u0, k, w, n_steps):
    """Estimate the temporal discretization error (RK4: O(dt^4))."""
    sol_dt = solve_burgers_ifrk4(u0, k, w, NU, DT, n_steps, 2, N_GT)
    sol_half = solve_burgers_ifrk4(u0, k, w, NU, DT / 2.0, 2 * n_steps, 2, N_GT)
    return np.abs(sol_dt[-1] - sol_half[-1]).max()

# ============================================================================
# 7. Main pipeline
# ============================================================================
def main():
    t_start = time.time()
    for d in (DATA_DIR, FIG_DIR, TRAIN_DIR, TEST_DIR, TRAJ_DIR):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("1D Burgers equation dataset generation (Section 4.1.1)")
    print("=" * 78)

    # --- grids -----------------------------------------------------------------
    x = np.arange(N_GT) / N_GT                     # x_j = j / N_GT in [0, 1)
    t_snap = np.linspace(0.0, T_MAX, N_SNAP)       # stored snapshot times
    n_steps = int(round(T_MAX / DT))               # total number of RK4 steps

    # --- GRF eigenvalues and wavenumbers ---------------------------------------
    k, lam = grf_eigenvalues(N_GT, K_KL, TAU, ALPHA)
    w = 2.0 * np.pi * k
    theor_std = np.sqrt(lam.sum())
    print(f"GRF: tau = {TAU:g}, alpha = {ALPHA:g}, K_KL = {K_KL}, "
          f"theoretical std(u_0) = {theor_std:.6e}")

    # --- validation of the solver on one sample --------------------------------
    u0_check = draw_grf_sample(k, lam, N_GT, rng)
    lin_err, lin_scale = validate_linear_limit(u0_check, k, w, n_steps)
    print(f"Solver check (linear limit):  max error = {lin_err:.3e} "
          f"(scale = {lin_scale:.3e})")
    conv_err = validate_temporal_convergence(u0_check, k, w, n_steps)
    print(f"Solver check (dt vs dt/2):    max difference = {conv_err:.3e}")

    # --- draw all GRF initial conditions ----------------------------------------
    print(f"Drawing {N_SAMPLES} GRF initial conditions ...")
    u0_all = np.empty((N_SAMPLES, N_GT))
    for s in range(N_SAMPLES):
        u0_all[s] = draw_grf_sample(k, lam, N_GT, rng)
    print(f"  empirical std(u_0) = {u0_all.std():.6e}, "
          f"max |u_0| = {np.abs(u0_all).max():.6e}")

    # --- solve Burgers equation for every sample ---------------------------------
    print(f"Solving Burgers equation (nu = {NU:g}) up to T = {T_MAX:g} ...")
    print(f"  grid: N_GT = {N_GT}, time steps: {n_steps} (dt = {DT:g}), "
          f"snapshots: {N_SNAP}")
    solutions_final = np.empty((N_SAMPLES, N_GT))
    example_traj = {}
    t_solve0 = time.time()
    for s in range(N_SAMPLES):
        snap = solve_burgers_ifrk4(u0_all[s], k, w, NU, DT, n_steps,
                                   N_SNAP, N_GT)
        solutions_final[s] = snap[-1]
        if s < N_TRAJ_EXAMPLES:
            example_traj[s] = snap
        if (s + 1) % 100 == 0:
            elapsed = time.time() - t_solve0
            eta = elapsed / (s + 1) * (N_SAMPLES - s - 1)
            print(f"  [{s + 1:4d}/{N_SAMPLES}] solved  "
                  f"(elapsed {elapsed:6.1f} s, ETA {eta:6.1f} s)")
    print(f"  all samples solved in {time.time() - t_solve0:.1f} s; "
          f"max |u(x,T)| = {np.abs(solutions_final).max():.6e}")

    # --- train / test split (seeded, reproducible) -------------------------------
    idx = rng.permutation(N_SAMPLES)
    idx_train = np.sort(idx[:N_TRAIN])
    idx_test = np.sort(idx[N_TRAIN:N_TRAIN + N_TEST])
    print(f"Split: {N_TRAIN} training / {N_TEST} test samples (seed = {SEED})")

    # --- save CSV data -------------------------------------------------------------
    print("Saving CSV datasets ...")
    save_matrix_csv(os.path.join(DATA_DIR, "x_grid.csv"), x[None, :], x)
    np.savetxt(os.path.join(DATA_DIR, "time_grid.csv"), t_snap,
               delimiter=",", fmt="%.6f", header="t", comments="")
    save_matrix_csv(os.path.join(DATA_DIR, "grf_initial_conditions.csv"),
                    u0_all, x)
    save_matrix_csv(os.path.join(TRAIN_DIR, "initial_conditions.csv"),
                    u0_all[idx_train], x)
    save_matrix_csv(os.path.join(TRAIN_DIR, "solutions_final.csv"),
                    solutions_final[idx_train], x)
    save_matrix_csv(os.path.join(TEST_DIR, "initial_conditions.csv"),
                    u0_all[idx_test], x)
    save_matrix_csv(os.path.join(TEST_DIR, "solutions_final.csv"),
                    solutions_final[idx_test], x)
    for s in example_traj:
        save_matrix_csv(
            os.path.join(TRAJ_DIR, f"sample_{s:03d}_trajectory.csv"),
            example_traj[s], x)

    # train/test partition table
    with open(os.path.join(DATA_DIR, "train_test_split.csv"), "w") as fh:
        fh.write("sample_index,split\n")
        test_set = set(idx_test.tolist())
        for i in range(N_SAMPLES):
            fh.write(f"{i},{'train' if i not in test_set else 'test'}\n")

    # parameter log
    with open(os.path.join(DATA_DIR, "parameters.csv"), "w") as fh:
        fh.write("parameter,value\n")
        for key, val in [
            ("equation", "u_t + u u_x = nu * u_xx, x in [0,1), periodic BC"),
            ("nu", NU), ("T_max", T_MAX), ("N_GT", N_GT),
            ("N_SAMPLES", N_SAMPLES), ("N_TRAIN", N_TRAIN),
            ("N_TEST", N_TEST), ("K_KL", K_KL), ("tau", TAU),
            ("alpha", ALPHA), ("N_SNAPSHOTS", N_SNAP), ("dt", DT),
            ("seed", SEED), ("grid", "uniform, x_j = j / N_GT"),
            ("initial_conditions", "GRF, C = (-Delta + tau^2)^(-alpha), KL"),
            ("solver", "Fourier pseudo-spectral (3/2 dealiasing) + IF-RK4"),
        ]:
            fh.write(f"{key},{val}\n")

    # --- save PNG figures ---------------------------------------------------------
    print("Saving PNG figures ...")
    plot_eigenvalue_spectrum(k, lam, os.path.join(FIG_DIR,
                             "grf_eigenvalue_spectrum.png"))
    plot_grf_samples(x, u0_all, 6, os.path.join(FIG_DIR,
                     "grf_initial_condition_samples.png"))
    plot_burgers_evolution(x, t_snap, example_traj[0], 0,
                           os.path.join(FIG_DIR,
                                        "burgers_evolution_sample_000.png"))
    plot_initial_vs_final(x, u0_all, solutions_final, [0, 1, 2, 3],
                          os.path.join(FIG_DIR, "burgers_initial_vs_final.png"))
    plot_split(x, u0_all, idx_train, idx_test,
               os.path.join(FIG_DIR, "train_test_split.png"))

    print("=" * 78)
    print(f"Done. Total wall-clock time: {time.time() - t_start:.1f} s")
    print(f"CSV data : {DATA_DIR}")
    print(f"Figures  : {FIG_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()



