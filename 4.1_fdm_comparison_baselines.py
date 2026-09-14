"""
4.1_fdm_comparison_baselines.py
===============================

Baseline comparison for Section 4.1.3 of the thesis.

Benchmark scenario: medium viscosity nu = 0.01 on the spatial grid N = 64.
Under this setting the Burgers solution shows moderate nonlinear advection
but no extremely sharp shocks -- the favourable operating regime of the FNO
surrogate model introduced in Chapter 3 (Table 3.1).

Two classical numerical baselines are compared against the FNO framework:

  * FDM (finite difference method)
      - MUSCL-type reconstruction with a TVD (van Leer) slope limiter,
      - Lax-Friedrichs numerical flux,
      - central second-order viscous term,
      - explicit 4th-order Runge-Kutta time stepping,
      - time step constrained by the CFL condition (diffusion-dominated),
      - grid spacing dx = 1/64.

  * Spectral method (reference / ground truth)
      - Fourier pseudo-spectral + IF-RK4 on N_GT = 2048 (Section 4.1.1).
      Its solutions are reused from the GRF dataset of Section 4.1.1; the
      initial conditions are the same GRF draws for every method, which
      makes the comparison fair.

The FNO surrogate (L = 4, k_max = 16, d_v = 64, GELU activation, Adam
optimizer, see Table 3.1) is a PyTorch model trained in Chapter 3; it is not
part of this script.  The error tables written here are formatted so that
the FNO test errors can be appended to the same CSV files afterwards.

Every file produced by this script carries a "4.1" prefix in its file name.
All comments and figure labels are written in English.
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
# --- Scenario (Section 4.1.3) ------------------------------------------------
NU = 0.01           # viscosity of the benchmark scenario
T_MAX = 1.0         # final time
N_FDM = 64          # spatial resolution of the FDM baseline (and of the FNO)
N_REF = 2048        # spectral reference resolution (ground truth, Section 4.1.1)
N_SAMPLES = 1000    # number of GRF initial-condition samples
N_TRAIN = 800       # training samples
N_TEST = 200        # test samples
SEED = 42           # reproducibility seed (identical to Section 4.1.1)

# --- FDM scheme parameters ----------------------------------------------------
CFL_DIFF = 0.25     # CFL safety factor for the explicit diffusion term
CFL_ADV = 0.8       # CFL safety factor for the advective term
N_SNAP = 101        # number of time snapshots used for the trajectory outputs
TRAJ_EXAMPLES = [0, 1, 2]          # samples exported as full trajectories
CONV_GRIDS = [32, 64, 128]         # grid sizes for the convergence study
CONV_SAMPLES = 10                  # samples used in the convergence study

# ============================================================================
# 2. Paths and helpers
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")


def save_matrix_csv(path, arr, x):
    """Save a 2-D array as a CSV (rows = samples/snapshots, cols = grid pts).

    The first row is a header with the physical coordinates of the grid.
    """
    header = ",".join([f"{xv:.6f}" for xv in x])
    np.savetxt(path, arr, delimiter=",", fmt="%.8e", header=header, comments="")


def load_matrix_csv(path):
    """Load a spatial CSV written by ``save_matrix_csv`` (skip the header)."""
    return np.loadtxt(path, delimiter=",", skiprows=1)


def load_reference_data():
    """Load the GRF initial conditions and spectral reference solutions.

    The data are the outputs of Section 4.1.1 (generate_burgers_dataset.py):
    u_0 on the N_REF grid and the reference solutions u(x, T) stored in the
    train/test folders together with the partition table.

    Returns
    -------
    u0_all : np.ndarray, (N_SAMPLES, N_REF)   initial conditions (2048 grid)
    ref_all : np.ndarray, (N_SAMPLES, N_REF)  reference solutions u(x, T)
    train_idx, test_idx : np.ndarray          partition of the 1000 samples
    """
    u0_all = load_matrix_csv(os.path.join(DATA_DIR, "grf_initial_conditions.csv"))
    with open(os.path.join(DATA_DIR, "train_test_split.csv")) as fh:
        lines = fh.read().splitlines()[1:]
    train_idx = np.array([i for i, ln in enumerate(lines)
                          if ln.split(",")[1] == "train"])
    test_idx = np.array([i for i, ln in enumerate(lines)
                         if ln.split(",")[1] == "test"])

    ref_all = np.empty_like(u0_all)
    ref_all[train_idx] = load_matrix_csv(
        os.path.join(TRAIN_DIR, "solutions_final.csv"))
    ref_all[test_idx] = load_matrix_csv(
        os.path.join(TEST_DIR, "solutions_final.csv"))
    return u0_all, ref_all, train_idx, test_idx


# ============================================================================
# 3. FDM solver: Lax-Friedrichs flux + TVD (van Leer) limiter
# ============================================================================
def van_leer_limiter(du_b, du_f):
    """Van Leer TVD slope limiter.

    phi(r) = (r + |r|) / (1 + |r|),   r = du_b / du_f,

    evaluated in a gradient form that never divides by zero:

        phi = 2*du_b / (du_b + du_f)    if du_b*du_f > 0  (same sign slopes),
        phi = 0                         otherwise (local extremum: clip).

    The limiter vanishes at local extrema, which guarantees the total
    variation diminishing (TVD) property of the reconstruction.
    """
    same_sign = du_b * du_f > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(same_sign, 2.0 * du_b / (du_b + du_f + 1e-30), 0.0)
    return phi


def fdm_rhs(u, nu, dx, dx2):
    """Semi-discrete FDM right-hand side of the viscous Burgers equation.

    Conservative form:  u_t + f(u)_x = nu * u_xx,  f(u) = u^2/2, discretised
    as (cell interfaces at x_{j+1/2}):

        du_j/dt = -(F_{j+1/2} - F_{j-1/2})/dx + nu (u_{j+1}-2u_j+u_{j-1})/dx^2

    with the Lax-Friedrichs (local) flux

        F(a, b) = 0.5*(f(a) + f(b)) - 0.5*c*(b - a),   c = max(|a|, |b|),

    and MUSCL-type second-order reconstruction with the van Leer TVD limiter:
        uL_{j+1/2} = u_j + 0.5*phi_L_j*(u_{j+1} - u_j),
        uR_{j+1/2} = u_{j+1} - 0.5*phi_R_j*(u_{j+2} - u_{j+1}).

    Periodic boundary conditions are imposed with ``np.roll``.
    """
    du_f = np.roll(u, -1) - u                    # du_f[j] = u[j+1] - u[j]
    du_b = u - np.roll(u, 1)                     # du_b[j] = u[j] - u[j-1]

    # left state at interface j+1/2
    phi_l = van_leer_limiter(du_b, du_f)
    u_l = u + 0.5 * phi_l * du_f
    # right state at interface j+1/2 (from cell j+1)
    phi_r = van_leer_limiter(du_f, np.roll(du_f, -1))
    u_r = np.roll(u, -1) - 0.5 * phi_r * np.roll(du_f, -1)

    # Lax-Friedrichs numerical flux F_{j+1/2}
    wave_speed = np.maximum(np.abs(u_l), np.abs(u_r))
    flux = 0.5 * (0.5 * u_l**2 + 0.5 * u_r**2) - 0.5 * wave_speed * (u_r - u_l)

    # flux divergence + central viscous term
    dudt = -(flux - np.roll(flux, 1)) / dx + nu * (np.roll(u, -1) - 2.0 * u
                                                   + np.roll(u, 1)) / dx2
    return dudt


def fdm_cfl_dt(dx, nu, umax):
    """Time step from the CFL conditions (both advective and diffusive).

    The diffusion term is the most restrictive constraint for the explicit
    RK4 scheme:  dt <= 2.785 / (4*nu/dx^2)  (RK4 real-axis stability limit),
    therefore a conservative factor CFL_DIFF is applied.  The advective CFL
    dt <= CFL_ADV*dx/max|u| is also enforced.
    """
    dt_diff = CFL_DIFF * dx**2 / nu          # diffusive CFL constraint
    dt_adv = CFL_ADV * dx / max(umax, 1e-12)  # advective CFL constraint
    return min(dt_diff, dt_adv)


def solve_fdm(u0, nu, dx, dt, n_steps, return_times=None):
    """Solve the viscous Burgers equation with the FDM scheme (RK4).

    Parameters
    ----------
    u0 : np.ndarray
        Initial condition on the N-point periodic grid.
    nu : float
        Viscosity.
    dx : float
        Grid spacing.
    dt : float
        Time step (CFL constrained).
    n_steps : int
        Number of RK4 steps.
    return_times : np.ndarray, optional
        Output times in [0, n_steps*dt] for which the trajectory is returned
        (linearly interpolated in time).

    Returns
    -------
    u_final : np.ndarray
        Solution at the final time.
    traj : np.ndarray, shape (len(return_times), N) or None
        The interpolated trajectory, only when ``return_times`` is given.
    """
    dx2 = dx * dx
    u = np.array(u0, dtype=float)

    if return_times is not None:
        times_fdm = np.arange(n_steps + 1) * dt
        stored = np.empty((n_steps + 1, u.size))
        stored[0] = u
    else:
        times_fdm = None
        stored = None

    for step in range(n_steps):
        k1 = fdm_rhs(u, nu, dx, dx2)
        k2 = fdm_rhs(u + 0.5 * dt * k1, nu, dx, dx2)
        k3 = fdm_rhs(u + 0.5 * dt * k2, nu, dx, dx2)
        k4 = fdm_rhs(u + dt * k3, nu, dx, dx2)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if stored is not None:
            stored[step + 1] = u

    if return_times is None:
        return u, None
    traj = np.column_stack([np.interp(return_times, times_fdm, stored[:, j])
                            for j in range(u.size)])
    return u, traj


# ============================================================================
# 4. Error metrics
# ============================================================================
def rel_l2(a, b):
    """Relative L2 error: ||a - b||_2 / ||b||_2."""
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def rel_linf(a, b):
    """Relative max (L-infinity) error: ||a - b||_inf / ||b||_inf."""
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


# ============================================================================
# 5. Plotting helpers (all figures saved as PNG, English labels)
# ============================================================================
def plot_final_snapshots(x_ref, x_fdm, ref, fdm, sample_ids, path):
    """FDM (N=64) vs spectral reference (N=2048) solutions at the final time."""
    n = len(sample_ids)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for ax, s in zip(axes.ravel(), sample_ids):
        ax.plot(x_ref, ref[s], lw=1.1, color="#1f77b4",
                label="Spectral reference (N=2048)")
        ax.plot(x_fdm, fdm[s], "o", ms=3.0, color="#d62728",
                label="FDM (N=64)")
        ax.set_title(f"Sample {s}")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$u(x,T)$")
    for ax in axes[1]:
        ax.set_xlabel("$x$")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"FDM vs. Spectral Reference at the Final Time "
                 f"$T = {T_MAX:g}$ ($\\nu = {NU:g}$)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_vs_time(t_snap, err_time, sample_ids, path):
    """Relative L2 error of the FDM baseline as a function of time."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for s, e in zip(sample_ids, err_time):
        ax.semilogy(t_snap, e, lw=1.4, label=f"Sample {s}")
    ax.set_xlabel("$t$")
    ax.set_ylabel("Relative L2 error  $||u_{\\mathrm{FDM}} - u_{\\mathrm{ref}}||_2 / ||u_{\\mathrm{ref}}||_2$")
    ax.set_title(f"FDM Error Growth in Time ($\\nu = {NU:g}$, N = {N_FDM})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fdm_evolution(x_fdm, t_snap, traj, sample_id, path):
    """Space-time diagram of the FDM solution of one sample."""
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    pc = ax.pcolormesh(x_fdm, t_snap, traj, shading="auto", cmap="viridis")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_title(f"FDM Solution $u(x,t)$, Sample {sample_id} "
                 rf"($\nu$ = {NU:g}, Lax-Friedrichs + TVD, N = {N_FDM})")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label(r"$u(x,t)$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_histogram(err_tr, err_te, path):
    """Histogram of the per-sample final relative L2 errors."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bins = np.histogram_bin_edges(np.concatenate([err_tr, err_te]), bins=25)
    ax.hist(err_tr, bins=bins, alpha=0.6, color="#1f77b4",
            label=f"Train (N={len(err_tr)}, mean={err_tr.mean():.3e})")
    ax.hist(err_te, bins=bins, alpha=0.6, color="#d62728",
            label=f"Test (N={len(err_te)}, mean={err_te.mean():.3e})")
    ax.set_xlabel("Relative L2 error at $T = 1.0$")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"FDM Baseline Error Distribution "
                 f"($\\nu = {NU:g}$, N = {N_FDM}, reference: spectral N={N_REF})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_summary(summary, path):
    """Bar chart of the mean +/- std relative L2 error per subset."""
    groups = [r for r in summary if r["group"] in ("train", "test", "overall")]
    names = [g["group"] for g in groups]
    means = [g["mean_l2"] for g in groups]
    stds = [g["std_l2"] for g in groups]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = ax.bar(names, means, yerr=stds, capsize=6, alpha=0.85,
                  color=["#1f77b4", "#d62728", "#2ca02c"])
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{m:.2e}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean relative L2 error at $T = 1.0$")
    ax.set_title(f"FDM Baseline Accuracy Summary ($\\nu = {NU:g}$, N = {N_FDM})\n"
                 "Spectral method (N=2048) is the reference (error = 0)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(grids, conv_l2, path):
    """Grid-convergence of the FDM baseline (expect ~2nd order, O(dx^2))."""
    dxs = 1.0 / np.array(grids)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.loglog(dxs, conv_l2, "o-", lw=1.5, color="#1f77b4",
              label="FDM vs. spectral reference (mean over samples)")
    # reference line with slope 2 (second-order convergence)
    k = conv_l2[-1] / dxs[-1]**2
    ax.loglog(dxs, k * dxs**2, "--", color="#d62728",
              label="Reference slope $O(\\Delta x^2)$")
    ax.set_xlabel(r"Grid spacing $\Delta x$")
    ax.set_ylabel("Mean relative L2 error at $T = 1.0$")
    ax.set_title(f"Grid Convergence of the FDM Baseline "
                 f"($\\nu = {NU:g}$, Lax-Friedrichs + TVD)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
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
    print("Section 4.1.3 - baseline comparison (FDM vs. spectral)")
    print(f"nu = {NU:g}, T = {T_MAX:g}, FDM grid N = {N_FDM}, "
          f"spectral reference N = {N_REF}")
    print("=" * 78)

    # --- load the GRF initial conditions and spectral reference ----------------
    u0_all, ref_all, train_idx, test_idx = load_reference_data()
    print(f"Loaded {u0_all.shape[0]} initial conditions ({N_REF}-point grid) "
          f"and reference solutions from Section 4.1.1.")

    # --- FDM grid: dx = 1/N, downsampled initial conditions ---------------------
    dx = 1.0 / N_FDM
    x_fdm = (np.arange(N_FDM) + 0.0) * dx            # x_j = j/64, periodic grid
    x_ref = np.arange(N_REF) / N_REF
    step_down = N_REF // N_FDM                        # 32
    u0_fdm = u0_all[:, ::step_down]                   # exact for band-limited u0
    ref_fdm = ref_all[:, ::step_down]                 # reference on the 64-grid

    # --- CFL-constrained time step ----------------------------------------------
    umax = np.abs(u0_fdm).max()
    dt = fdm_cfl_dt(dx, NU, umax)
    n_steps = int(np.ceil(T_MAX / dt))
    dt = T_MAX / n_steps                              # land exactly on T
    print(f"FDM scheme: dx = {dx:.6f}, max|u0| = {umax:.4e}")
    print(f"  CFL: diffusive dt = {CFL_DIFF*dx**2/NU:.4e}, "
          f"advective dt = {CFL_ADV*dx/umax:.4e}")
    print(f"  -> dt = {dt:.4e}, {n_steps} RK4 steps")

    # --- solve the FDM baseline for all samples ---------------------------------
    print(f"Solving the FDM baseline for {N_SAMPLES} samples ...")
    t_solve0 = time.time()
    fdm_final = np.empty((N_SAMPLES, N_FDM))
    for s in range(N_SAMPLES):
        u_final, _ = solve_fdm(u0_fdm[s], NU, dx, dt, n_steps)
        fdm_final[s] = u_final
    print(f"  FDM solve finished in {time.time() - t_solve0:.1f} s")

    # --- FDM trajectories for the example samples --------------------------------
    t_snap = np.linspace(0.0, T_MAX, N_SNAP)          # same grid as Section 4.1.1
    fdm_traj = {}
    for s in TRAJ_EXAMPLES:
        _, traj = solve_fdm(u0_fdm[s], NU, dx, dt, n_steps,
                            return_times=t_snap)
        fdm_traj[s] = traj

    # --- per-sample error metrics at the final time -------------------------------
    print("Computing error metrics ...")
    err_l2 = np.empty(N_SAMPLES)
    err_linf = np.empty(N_SAMPLES)
    for s in range(N_SAMPLES):
        err_l2[s] = rel_l2(fdm_final[s], ref_fdm[s])
        err_linf[s] = rel_linf(fdm_final[s], ref_fdm[s])

    # --- temporal convergence check of the FDM scheme (sample 0) ------------------
    sol_dt, _ = solve_fdm(u0_fdm[0], NU, dx, dt, n_steps)
    sol_half, _ = solve_fdm(u0_fdm[0], NU, dx, dt / 2.0, 2 * n_steps)
    temp_diff = np.abs(sol_dt - sol_half).max()
    print(f"  temporal check (dt vs dt/2): max difference = {temp_diff:.3e}")

    # --- error growth in time for the example samples ------------------------------
    err_time = []
    for s in TRAJ_EXAMPLES:
        ref_traj_s = np.loadtxt(os.path.join(DATA_DIR, "trajectories",
                                             f"sample_{s:03d}_trajectory.csv"),
                                delimiter=",", skiprows=1)[:, ::step_down]
        e = np.array([rel_l2(fdm_traj[s][k], ref_traj_s[k])
                      for k in range(N_SNAP)])
        err_time.append(e)

    # --- grid-convergence study -----------------------------------------------------
    print("Grid-convergence study (FDM at dx = 1/32, 1/64, 1/128) ...")
    conv_l2, conv_linf = [], []
    for n_g in CONV_GRIDS:
        u0_g_all = u0_all[:, ::N_REF // n_g]
        ref_g_all = ref_all[:, ::N_REF // n_g]
        dx_g = 1.0 / n_g
        dt_g = fdm_cfl_dt(dx_g, NU, np.abs(u0_g_all).max())
        n_g_steps = int(np.ceil(T_MAX / dt_g))
        dt_g = T_MAX / n_g_steps
        l2_g, linf_g = [], []
        for s in range(CONV_SAMPLES):
            u_g, _ = solve_fdm(u0_g_all[s], NU, dx_g, dt_g, n_g_steps)
            l2_g.append(rel_l2(u_g, ref_g_all[s]))
            linf_g.append(rel_linf(u_g, ref_g_all[s]))
        conv_l2.append(np.mean(l2_g))
        conv_linf.append(np.mean(linf_g))
    print("  mean relative L2 error: " +
          ", ".join(f"dx=1/{n}: {e:.3e}" for n, e in zip(CONV_GRIDS, conv_l2)))

    # --- validation: mean conservation of the FDM scheme ---------------------------
    mean_err = np.abs(fdm_final.mean(1) - u0_fdm.mean(1)).max()
    print(f"  FDM mean conservation: max |mean(u_T) - mean(u_0)| = {mean_err:.3e}")

    # --- error summary tables --------------------------------------------------------
    summary = []
    for name, idxs in (("train", train_idx), ("test", test_idx),
                       ("overall", np.arange(N_SAMPLES))):
        summary.append({
            "group": name, "n": len(idxs),
            "mean_l2": err_l2[idxs].mean(), "std_l2": err_l2[idxs].std(),
            "median_l2": np.median(err_l2[idxs]), "max_l2": err_l2[idxs].max(),
            "mean_linf": err_linf[idxs].mean(),
        })
    for g in summary:
        print(f"  {g['group']:>7s}: mean rel-L2 = {g['mean_l2']:.3e} "
              f"+- {g['std_l2']:.3e}, median = {g['median_l2']:.3e}, "
              f"max = {g['max_l2']:.3e}, mean rel-Linf = {g['mean_linf']:.3e}")

    # --- save CSV data ----------------------------------------------------------------
    print("Saving CSV datasets ...")
    save_matrix_csv(os.path.join(DATA_DIR, "4.1_fdm_initial_conditions.csv"),
                    u0_fdm, x_fdm)
    save_matrix_csv(os.path.join(DATA_DIR, "4.1_fdm_solutions_final.csv"),
                    fdm_final, x_fdm)
    save_matrix_csv(os.path.join(DATA_DIR, "4.1_spectral_reference_final_64.csv"),
                    ref_fdm, x_fdm)
    for s in TRAJ_EXAMPLES:
        save_matrix_csv(os.path.join(DATA_DIR,
                                     f"4.1_fdm_trajectory_sample_{s:03d}.csv"),
                        fdm_traj[s], x_fdm)
    split_labels = np.empty(N_SAMPLES, dtype=object)
    split_labels[:] = "train"
    split_labels[idx_test] = "test"
    np.savetxt(os.path.join(DATA_DIR, "4.1_error_metrics.csv"),
               np.column_stack([np.arange(N_SAMPLES),
                                split_labels,
                                err_l2, err_linf]),
               delimiter=",", fmt="%s",
               header="sample_index,split,fdm_rel_l2,fdm_rel_linf",
               comments="")
    with open(os.path.join(DATA_DIR, "4.1_error_summary.csv"), "w") as fh:
        fh.write("group,n_samples,mean_rel_l2,std_rel_l2,median_rel_l2,"
                 "max_rel_l2,mean_rel_linf\n")
        for g in summary:
            fh.write(f"{g['group']},{g['n']},{g['mean_l2']:.8e},"
                     f"{g['std_l2']:.8e},{g['median_l2']:.8e},"
                     f"{g['max_l2']:.8e},{g['mean_linf']:.8e}\n")
    with open(os.path.join(DATA_DIR, "4.1_convergence.csv"), "w") as fh:
        fh.write("grid_N,dx,mean_rel_l2,mean_rel_linf\n")
        for n_g, e2, ei in zip(CONV_GRIDS, conv_l2, conv_linf):
            fh.write(f"{n_g},{1.0/n_g:.8e},{e2:.8e},{ei:.8e}\n")
    with open(os.path.join(DATA_DIR, "4.1_parameters.csv"), "w") as fh:
        fh.write("parameter,value\n")
        for key, val in [
            ("equation", "u_t + u u_x = nu*u_xx, x in [0,1), periodic BC"),
            ("scenario", "medium viscosity nu = 0.01, N = 64"),
            ("nu", NU), ("T_max", T_MAX), ("N_FDM", N_FDM),
            ("N_REF", N_REF), ("N_SAMPLES", N_SAMPLES),
            ("N_TRAIN", N_TRAIN), ("N_TEST", N_TEST),
            ("seed", SEED),
            ("fdm_flux", "Lax-Friedrichs (local)"),
            ("fdm_limiter", "van Leer TVD"),
            ("fdm_time_scheme", "explicit RK4, CFL-constrained"),
            ("cfl_diff", CFL_DIFF), ("cfl_adv", CFL_ADV),
            ("fdm_dt", f"{dt:.8e}"), ("fdm_n_steps", n_steps),
            ("reference_solver", "Fourier pseudo-spectral + IF-RK4, N=2048"),
            ("fno_note", "FNO (L=4, k_max=16, d_v=64, GELU, Adam) is the "
                         "Chapter-3 surrogate; add its test errors to "
                         "4.1_error_metrics.csv for the final comparison"),
        ]:
            fh.write(f"{key},{val}\n")

    # --- save PNG figures ------------------------------------------------------------
    print("Saving PNG figures ...")
    plot_final_snapshots(x_ref, x_fdm, ref_all, fdm_final, [0, 1, 2, 3],
                         os.path.join(FIG_DIR,
                                      "4.1_fdm_vs_spectral_snapshots.png"))
    plot_error_vs_time(t_snap, err_time, TRAJ_EXAMPLES,
                       os.path.join(FIG_DIR, "4.1_fdm_error_vs_time.png"))
    plot_fdm_evolution(x_fdm, t_snap, fdm_traj[0], 0,
                       os.path.join(FIG_DIR, "4.1_fdm_solution_evolution.png"))
    plot_error_histogram(err_l2[train_idx], err_l2[test_idx],
                         os.path.join(FIG_DIR, "4.1_error_histogram.png"))
    plot_error_summary(summary,
                       os.path.join(FIG_DIR, "4.1_error_summary.png"))
    plot_convergence(CONV_GRIDS, conv_l2,
                     os.path.join(FIG_DIR, "4.1_fdm_convergence.png"))

    print("=" * 78)
    print(f"Done. Total wall-clock time: {time.time() - t_start:.1f} s")
    print(f"CSV data : {DATA_DIR}  (4.1_* files)")
    print(f"Figures  : {FIG_DIR}  (4.1_* files)")
    print("=" * 78)


if __name__ == "__main__":
    main()





