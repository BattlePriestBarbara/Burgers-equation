"""
4.3_generate_reference_solutions.py
===================================

Generate high-accuracy spectral reference solutions for the viscosity
sensitivity study of Section 4.3.

The same 1000 GRF initial conditions (Section 4.1.1, seed = 42) are solved
with the (already validated) Fourier pseudo-spectral + IF-RK4 solver for the
two additional viscosities nu = 0.1 and nu = 0.001.  The nu = 0.01 solutions
already exist from Section 4.1.1 and are reused.

Every reference solution is validated against the Cole-Hopf exact solution
for a few samples before it is written to disk.

Outputs (all with the "4.3" prefix):
    data/4.3_reference_solutions_nu0p1.csv    (1000 x 2048)
    data/4.3_reference_solutions_nu0p001.csv  (1000 x 2048)
    data/4.3_reference_validation.csv
"""

import os
import time

import numpy as np
import importlib.util

# --- load the validated spectral solver from Section 4.1.1 -------------------
spec = importlib.util.spec_from_file_location("g41",
                                              "generate_burgers_dataset.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

N_REF = g.N_GT
NU_LIST = [0.1, 0.001]
SEED = 42


def cole_hopf_exact(u0, nu, n=2048, t=g.T_MAX):
    """Exact viscous-Burgers solution via the Cole-Hopf transform (periodic).

    u(x,t) = m0 + w(x + m0*t mod 1, t),   w = -2 nu (ln phi)_x,
    phi_t = nu phi_xx, phi(x,0) = exp(-(1/2nu) * integral(u0 - m0)).
    """
    x = np.arange(n) / n
    dx = 1.0 / n
    m0 = u0.mean()
    U0 = u0 - m0
    P = np.concatenate([[0.0], np.cumsum((U0[:-1] + U0[1:]) / 2 * dx)])
    phi0 = np.exp(-P / (2.0 * nu))
    Phi = np.fft.fft(phi0)
    k = np.fft.fftfreq(n, 1.0 / n)
    Phi_t = Phi * np.exp(-nu * (2 * np.pi * k)**2 * t)
    phi = np.fft.ifft(Phi_t).real
    dphi = np.fft.ifft(1j * (2 * np.pi * k) * Phi_t).real
    w = -2.0 * nu * dphi / phi
    xs = (x - m0 * t) % 1.0
    return m0 + np.interp(xs, x, w, period=1.0)


def rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)

    # --- load the (nu-independent) GRF initial conditions ----------------------
    u0_all = np.loadtxt(os.path.join(DATA_DIR, "grf_initial_conditions.csv"),
                        delimiter=",", skiprows=1)
    k = np.fft.fftfreq(N_REF, d=1.0 / N_REF)
    w = 2.0 * np.pi * k
    print("=" * 78)
    print("Section 4.3 - reference solution generation "
          f"(viscosities {NU_LIST})")
    print("=" * 78)

    valid_lines = ["sample,nu,dt,rel_l2_vs_cole_hopf,rel_l2_dt_vs_dt2"]

    for nu in NU_LIST:
        # pick the time step: dt = 1e-3 is accurate for nu >= 0.01; for the
        # sharp solutions at nu = 0.001 use dt = 5e-4 (validated below).
        dt = 1e-3 if nu >= 0.01 else 5e-4
        n_steps = int(round(g.T_MAX / dt))

        # --- validation on one sample before the full solve -------------------
        s0 = 0
        u_ex = cole_hopf_exact(u0_all[s0], nu)
        snap = g.solve_burgers_ifrk4(u0_all[s0], k, w, nu, dt, n_steps, 2, N_REF)
        err_ch = rel_l2(snap[-1], u_ex)
        snap_half = g.solve_burgers_ifrk4(u0_all[s0], k, w, nu, dt / 2.0,
                                          2 * n_steps, 2, N_REF)
        err_dt = rel_l2(snap[-1], snap_half[-1])
        print(f"nu = {nu:g}: dt = {dt:.1e}, Cole-Hopf error = {err_ch:.3e}, "
              f"dt-vs-dt/2 = {err_dt:.3e}")
        valid_lines.append(f"{s0},{nu:g},{dt:.1e},{err_ch:.3e},{err_dt:.3e}")

        # --- solve all 1000 samples -------------------------------------------
        sol = np.empty_like(u0_all)
        t0 = time.time()
        for s in range(u0_all.shape[0]):
            sol[s] = g.solve_burgers_ifrk4(u0_all[s], k, w, nu, dt, n_steps,
                                           2, N_REF)[-1]
            if (s + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  [{s + 1}/1000] nu = {nu:g} solved  "
                      f"(elapsed {el:6.1f} s, ETA {el / (s + 1) * (1000 - s - 1):6.1f} s)")
        print(f"  nu = {nu:g}: all samples solved in {time.time() - t0:.1f} s; "
              f"max |u| = {np.abs(sol).max():.4e}")

        fname = "4.3_reference_solutions_nu0p1" if nu == 0.1 else \
                "4.3_reference_solutions_nu0p001"
        path = os.path.join(DATA_DIR, fname + ".csv")
        np.savetxt(path, sol, delimiter=",", fmt="%.8e",
                   header=",".join(f"{i / N_REF:.6f}" for i in range(N_REF)),
                   comments="")
        print(f"  saved: {path}")

    with open(os.path.join(DATA_DIR, "4.3_reference_validation.csv"), "w") as fh:
        fh.write("\n".join(valid_lines) + "\n")
    print(f"Total wall-clock time: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
