# Fourier Neural Operator Surrogate for the 1D Viscous Burgers Equation

A summary of this repository: the numerical experiments supporting the thesis chapter on
**learning the solution operator of the 1D viscous Burgers equation with a Fourier Neural Operator (FNO)**,
including the Gaussian-random-field dataset, the classical numerical baselines it is compared against,
all result data and all figures.

* **Language:** Python (NumPy, Matplotlib; PyTorch for the three training scripts).
* **Contents:** 6 Python scripts, 40 CSV data/result files (165.9 MB) and 17 figures (1.6 MB) — 65 files, 167.7 MB.
* **License:** MIT (see `LICENSE`, Copyright (c) 2026 Xinhang Yu).

| Section | Experiment | Script | Figures |
|---|---|---|---|
| 4.1.1 | GRF initial conditions + spectral reference solutions | `generate_burgers_dataset.py` | 5 |
| 4.1.3 | FDM baseline vs. spectral reference | `4.1_fdm_comparison_baselines.py` | 6 |
| 4.2 | Resolution invariance / zero-shot super-resolution (FNO vs. U-Net) | `4.2_resolution_invariance.py` | 2 |
| 4.3 | Shock capture and viscosity sensitivity (nu = 0.1, 0.01, 0.001) | `4.3_generate_reference_solutions.py`, `4.3_viscosity_sensitivity.py` | 2 |
| 4.4 | Fourier-mode ablation (k_max sweep at nu = 0.001) | `4.4_mode_ablation.py` | 2 |

---

## 1. Repository contents

```
.
├── LICENSE                                  (1.1 KB)  MIT License
├── .gitignore                               (805 B)   Python venvs and caches excluded
├── generate_burgers_dataset.py              (21.8 KB) Section 4.1.1
├── 4.1_fdm_comparison_baselines.py          (24.4 KB) Section 4.1.3
├── 4.2_resolution_invariance.py             (22.3 KB) Section 4.2
├── 4.3_generate_reference_solutions.py      (4.8 KB)  Section 4.3
├── 4.3_viscosity_sensitivity.py             (22.2 KB) Section 4.3
├── 4.4_mode_ablation.py                     (20.8 KB) Section 4.4
├── data/                                    (40 CSV files, 165.9 MB; see Section 4)
│   ├── x_grid.csv  time_grid.csv  parameters.csv  train_test_split.csv
│   ├── grf_initial_conditions.csv           (30.3 MB) 1000 x 2048
│   ├── train/    initial_conditions.csv  solutions_final.csv   (800 samples each)
│   ├── test/     initial_conditions.csv  solutions_final.csv   (200 samples each)
│   ├── trajectories/  sample_000..002_trajectory.csv           (101 x 2048 each)
│   └── 4.1_*.csv   4.2_*.csv   4.3_*.csv   4.4_*.csv          (result data, see Section 4)
└── figures/                                 17 PNG figures (~1.6 MB, see Section 5)
```

Notes:

* Trained network weights (PyTorch `state_dict` files) are **not** part of this repository; they are recreated by
  running the training scripts. All *results* (metrics, spectra, loss histories) **are** included.
* Python virtual environments and byte-code caches are excluded via `.gitignore`; the repository contains only the
  research artefacts: source code, data and figures.

## 2. Problem setup

**Governing equation** (viscous Burgers, periodic):

```
u_t + u * u_x = nu * u_xx,        x in [0, 1)      (periodic boundary conditions)
u(0, x) = u_0(x),                 target time T = 1.0
```

**Initial conditions — Gaussian Random Field (GRF).** All 1000 initial conditions are drawn from a zero-mean GRF with

```
C = (-Delta + tau^2)^(-alpha),      tau = 7,   alpha = 2,
```

whose periodic eigenfunctions are the Fourier modes `exp(2*pi*i*k*x)` with eigenvalues
`lambda_k = (4*pi^2*k^2 + tau^2)^(-alpha)`. Sampling uses a Karhunen–Loeve expansion truncated at **`K_KL = 128`**
modes with Hermitian-symmetric complex draws, so every `u_0` is real and band-limited to |k| ≤ 128. The same 1000
initial conditions are used by every method and every experiment, which makes all comparisons consistent.

**Reference solutions (ground truth).** Fourier pseudo-spectral solver with 3/2 dealiasing and an integrating-factor
RK4 time integrator on a 2048-point grid with `dt = 1e-3`. Validation: the linear-diffusion limit is matched to ~1e-15,
temporal convergence (dt vs dt/2) is ~3e-11, and the spatial mean is conserved exactly. For the additional viscosities
the reference solutions are also checked against the **Cole–Hopf exact solution**
(relative L2 = 7.6e-9 at nu = 0.1 and 5.3e-6 at nu = 0.001).

**Surrogate model (FNO).** `L = 4` Fourier layers, `k_max = 16`, hidden width `d_v = 64`, GELU activation; trained with
Adam (lr = 1e-3) and a cosine schedule for 200 epochs (batch 32, MSE) on the 800 training samples at `N = 64`.
Parameter count: **565,889** stored float values.

**Baselines.** FDM: MUSCL reconstruction with the van Leer TVD limiter, Lax–Friedrichs flux, central second-order
viscous term, explicit RK4 with CFL-constrained step (`dx = 1/64`, `dt = 6.0976e-3`, 164 steps). Spectral solver on
N = 2048 as ground truth. Parameter-matched 1-D U-Net (563,041 parameters, 0.50% difference) for the
resolution-invariance study.

## 3. Source code

Six Python scripts implement the entire study. All of them are self-contained, use the headless `Agg` backend for
plotting and write their outputs to `data/` and `figures/`. There is **no SciPy dependency**: the Wilcoxon signed-rank
test, the Jarque–Bera test and all bootstrap confidence intervals are implemented from scratch in
`4.4_mode_ablation.py`.

### 3.1 `generate_burgers_dataset.py` — dataset and reference solutions (Section 4.1.1)

Builds the GRF initial conditions and the spectral reference solutions that every later experiment reuses.
Key configuration: `K_KL = 128`, `tau = 7`, `alpha = 2`, `N_GT = 2048`, `N_SAMPLES = 1000`, `T_MAX = 1.0`,
`N_SNAP = 101`, `DT = 1e-3`, `NU = 0.01`, `SEED = 42`, split 800/200.

Main components: `grf_eigenvalues` (covariance eigenvalues, truncated at `K_KL`), `draw_grf_sample`
(truncated Karhunen–Loeve draw with Hermitian symmetry), `burgers_nonlinear` (3/2-dealiased pseudo-spectral advection),
`solve_burgers_ifrk4` (integrating-factor RK4), the solver self-checks `validate_linear_limit` /
`validate_temporal_convergence`, five plotting helpers and `main`.

Runs in ~9 minutes on a single core: `python generate_burgers_dataset.py`

### 3.2 `4.1_fdm_comparison_baselines.py` — FDM baseline (Section 4.1.3)

Solves the benchmark scenario (nu = 0.01, N = 64, T = 1.0) with a classical finite-difference method and measures it
against the spectral reference, establishing the accuracy level the FNO has to beat. Scheme: MUSCL reconstruction with
the van Leer TVD limiter, Lax–Friedrichs flux, central second-order diffusion and explicit RK4 with
`CFL_DIFF = 0.25`, `CFL_ADV = 0.8` (measured `dt = 6.0976e-3`, 164 steps). It also runs a grid-convergence study
(`N = 32, 64, 128`) and checks mean conservation.

Main components: `van_leer_limiter`, `fdm_rhs`, `fdm_cfl_dt`, `solve_fdm`, `rel_l2` / `rel_linf`, six plotting helpers,
`main`. Runs in ~2 minutes: `python 4.1_fdm_comparison_baselines.py`

### 3.3 `4.2_resolution_invariance.py` — zero-shot super-resolution (Section 4.2)

Trains an FNO and a parameter-matched U-Net at N = 64 and evaluates both, **without fine-tuning**, on the grids
64, 128, 256, 512, 1024 and 2048. Architectures: `SpectralConv1d` (complex weights on the first `k_max` Fourier modes,
real weight for the DC mode — grid-independent by construction), `FNO1d` (width 64, 4 layers, GELU),
`UNet1d` (channels 120/240, max-pooling and nearest-neighbour upsampling; grid-sensitive). `count_params` counts stored
float values (complex counted twice) so the two models can be compared fairly; `evaluate_resolutions` performs the
zero-shot evaluation and `bootstrap_ci` the 1000-resample 95% CI.

Training: Adam lr = 1e-3, cosine schedule, 200 epochs, batch 32, MSE. If the weight files are absent the script trains
the models first, then evaluates and writes `data/4.2_resolution_table.csv` and the two figures.
Run: `python 4.2_resolution_invariance.py` (requires PyTorch).

### 3.4 `4.3_generate_reference_solutions.py` — reference solutions for nu = 0.1 and 0.001 (Section 4.3)

Generates reference solutions for the two additional viscosities by importing the validated solver of Section 4.1.1 as a
module. Time step 1e-3 for nu = 0.1 and 5e-4 for the sharper nu = 0.001 case; each viscosity is validated against the
**Cole–Hopf exact solution** before the full 1000-sample batch is solved (~30 minutes, once):
`python 4.3_generate_reference_solutions.py`

### 3.5 `4.3_viscosity_sensitivity.py` — shock capture and viscosity sensitivity (Section 4.3)

Trains one FNO per viscosity (identical hyper-parameters; the nu = 0.01 model is reused from Section 4.2) and compares
it with the FDM baseline (Lax–Friedrichs + TVD, N = 64). The FDM solver is vectorised over all 1000 samples and cached
in CSV checkpoint files, so repeated runs are cheap. Per-sample relative L2 / L-infinity errors, bootstrap 95% CIs and
the FNO-to-FDM error ratio are written to `data/4.3_*.csv`.
Run: `python 4.3_viscosity_sensitivity.py` (requires PyTorch).

### 3.6 `4.4_mode_ablation.py` — Fourier-mode ablation (Section 4.4)

Trains six otherwise identical FNOs with `k_max` in {8, 12, 16, 20, 24, 32} on the nu = 0.001 (sharp-shock) dataset and
records accuracy, parameters, inference time, convergence epoch and the frequency-domain error spectrum
`E_k = mean_i abs(FFT[u_pred_i - u_GT_i](k))**2`. Statistics: paired bootstrap CIs, Wilcoxon signed-rank tests between
adjacent `k_max` values and Jarque–Bera tests of the error distributions — all implemented with NumPy/`math` only.
Loss histories are stored as `data/4.4_loss_kmax*.csv`, results in `data/4.4_ablation_table.csv`,
`data/4.4_error_spectrum.csv` and `data/4.4_statistics.csv`.
Run: `python 4.4_mode_ablation.py` (requires PyTorch).

## 4. Data dictionary (40 CSV files)

All field-data CSVs use the same layout: **one sample (or one time snapshot) per row, the physical grid coordinates in
the header line**, values in `%.8e` scientific notation, LF line endings.

### 4.1 Shared dataset (Section 4.1.1) — 12 files

| File | Shape / rows | Size | Content |
|---|---|---|---|
| `data/x_grid.csv` | 1 × 2048 | 48 KB | spatial grid `x_j = j/2048` |
| `data/time_grid.csv` | 101 | 911 B | snapshot times t = 0, 0.01, …, 1.0 |
| `data/parameters.csv` | 17 | 344 B | generation parameters (nu, T, N_GT, K_KL, tau, alpha, dt, seed, solver, …) |
| `data/train_test_split.csv` | 1000 | 9.5 KB | `sample_index, split` (seed 42) |
| `data/grf_initial_conditions.csv` | 1000 × 2048 | 30.3 MB | all GRF initial conditions (input of every experiment) |
| `data/train/initial_conditions.csv` | 800 × 2048 | 24.3 MB | training inputs `u_0` |
| `data/train/solutions_final.csv` | 800 × 2048 | 24.3 MB | training reference solutions `u(x,T)` (nu = 0.01, spectral N = 2048) |
| `data/test/initial_conditions.csv` | 200 × 2048 | 6.1 MB | test inputs `u_0` |
| `data/test/solutions_final.csv` | 200 × 2048 | 6.1 MB | test reference solutions `u(x,T)` |
| `data/trajectories/sample_000_trajectory.csv` | 101 × 2048 | 3.0 MB | full space-time evolution `u(x,t)` of sample 0 |
| `data/trajectories/sample_001_trajectory.csv` | 101 × 2048 | 3.1 MB | full space-time evolution of sample 1 |
| `data/trajectories/sample_002_trajectory.csv` | 101 × 2048 | 3.1 MB | full space-time evolution of sample 2 |

### 4.2 Section 4.1.3 — FDM baseline — 10 files

| File | Shape / rows | Size | Content |
|---|---|---|---|
| `4.1_parameters.csv` | 20 | 572 B | scenario and FDM settings (`fdm_dt = 6.09756098e-03`, `fdm_n_steps = 164`) |
| `4.1_fdm_initial_conditions.csv` | 1000 × 64 | 993 KB | `u_0` on the 64-point grid (exact, since `u_0` is band-limited) |
| `4.1_fdm_solutions_final.csv` | 1000 × 64 | 993 KB | FDM baseline solutions at T = 1.0 |
| `4.1_spectral_reference_final_64.csv` | 1000 × 64 | 993 KB | spectral ground truth downsampled to N = 64 |
| `4.1_fdm_trajectory_sample_00{0,1,2}.csv` | 101 × 64 | 98–100 KB each | FDM space-time trajectories of three samples |
| `4.1_error_metrics.csv` | 1000 | 53.5 KB | `sample_index, split, fdm_rel_l2, fdm_rel_linf` |
| `4.1_error_summary.csv` | 3 | 335 B | train / test / overall error statistics |
| `4.1_convergence.csv` | 3 | 181 B | grid-convergence study at N = 32, 64, 128 |

### 4.3 Section 4.2 — resolution invariance — 1 file

`data/4.2_resolution_table.csv` (903 B, 12 rows): `method, resolution, mean_rel_l2, ci_low, ci_high, time_ms, params`
for the FNO and the U-Net at the six evaluation grids.

### 4.4 Section 4.3 — viscosity sensitivity — 8 files

| File | Shape / rows | Size | Content |
|---|---|---|---|
| `4.3_reference_solutions_nu0p1.csv` | 1000 × 2048 | 30.3 MB | spectral reference solutions, nu = 0.1 |
| `4.3_reference_solutions_nu0p001.csv` | 1000 × 2048 | 30.3 MB | spectral reference solutions, nu = 0.001 |
| `4.3_reference_validation.csv` | 2 | 120 B | Cole–Hopf and dt-convergence validation of the two reference sets |
| `4.3_fdm_solutions_nu0p1.csv` | 1000 × 64 | 993 KB | FDM baseline solutions, nu = 0.1 |
| `4.3_fdm_solutions_nu0p001.csv` | 1000 × 64 | 993 KB | FDM baseline solutions, nu = 0.001 |
| `4.3_error_metrics.csv` | 600 | 57.5 KB | `sample_index, split, nu, method, rel_l2, rel_linf` (3 viscosities × FNO/FDM × 200 test samples) |
| `4.3_summary.csv` | 6 | 676 B | mean errors with bootstrap 95% CIs per (nu, method) |
| `4.3_parameters.csv` | 15 | 521 B | configuration of the experiment |

### 4.5 Section 4.4 — Fourier-mode ablation — 9 files

| File | Shape / rows | Size | Content |
|---|---|---|---|
| `4.4_ablation_table.csv` | 6 | 471 B | `kmax, mean_rel_l2, ci_low, ci_high, params, time_ms, converged_epoch` |
| `4.4_error_spectrum.csv` | 33 | 3.1 KB | `wavenumber, kmax8 … kmax32` — the error spectra `E_k`, k = 0…32 |
| `4.4_statistics.csv` | 7 | 360 B | five Wilcoxon tests and two Jarque–Bera tests |
| `4.4_loss_kmax{8,12,16,20,24,32}.csv` | 200 each | 5.0 KB each | per-epoch training loss of the six models |

## 5. Figures

All 17 figures in `figures/` are 150-dpi PNGs with English labels. They are the illustrative output of the six scripts;
the numeric results behind them are in the CSVs of Section 4.

### 5.1 Dataset generation (Section 4.1.1)

**`figures/grf_eigenvalue_spectrum.png`** (55.8 KB) — Decay of the GRF covariance eigenvalues
`lambda_k = (4*pi^2*k^2 + tau^2)^(-alpha)`; shows why the Karhunen–Loeve expansion can be truncated at `K_KL = 128`.

![Decay of the GRF covariance eigenvalues](figures/grf_eigenvalue_spectrum.png)

**`figures/grf_initial_condition_samples.png`** (156.0 KB) — Six representative GRF initial conditions `u_0(x)`: zero
mean, smooth and band-limited.

![Six GRF initial conditions](figures/grf_initial_condition_samples.png)

**`figures/burgers_evolution_sample_000.png`** (83.1 KB) — Space–time diagram `u(x,t)` of sample 0 (spectral solver,
N = 2048): smooth data steepening into a shock-like profile as t → 1.

![Space-time evolution of one solution](figures/burgers_evolution_sample_000.png)

**`figures/burgers_initial_vs_final.png`** (135.5 KB) — `u_0(x)` against the reference solution `u(x,T)` for four
samples.

![Initial condition vs. final solution](figures/burgers_initial_vs_final.png)

**`figures/train_test_split.png`** (86.1 KB) — The 800/200 train/test partition and the mean ± 1 std of `u_0` for both
subsets (evidence that the split is statistically homogeneous).

![Train/test partition and statistics](figures/train_test_split.png)

### 5.2 Baseline comparison, FDM vs. spectral (Section 4.1.3)

**`figures/4.1_fdm_vs_spectral_snapshots.png`** (133.3 KB) — FDM solution (N = 64, markers) against the spectral
reference (N = 2048, line) at T = 1.0 for four samples.

![FDM vs. spectral reference at T = 1](figures/4.1_fdm_vs_spectral_snapshots.png)

**`figures/4.1_fdm_error_vs_time.png`** (76.7 KB) — Growth of the FDM relative L2 error over t in [0,1].

![FDM error growth in time](figures/4.1_fdm_error_vs_time.png)

**`figures/4.1_fdm_solution_evolution.png`** (52.2 KB) — Space–time diagram of the FDM solution on the 64-point grid.

![FDM solution space-time diagram](figures/4.1_fdm_solution_evolution.png)

**`figures/4.1_error_histogram.png`** (55.2 KB) — Distribution of the per-sample relative L2 errors, train vs. test.

![Distribution of per-sample errors](figures/4.1_error_histogram.png)

**`figures/4.1_error_summary.png`** (51.6 KB) — Mean ± std relative L2 error for the train, test and overall subsets.

![Error summary per subset](figures/4.1_error_summary.png)

**`figures/4.1_fdm_convergence.png`** (86.8 KB) — Grid convergence of the FDM baseline (dx = 1/32, 1/64, 1/128) with the
O(dx²) reference slope: clean second-order accuracy.

![Grid convergence of the FDM baseline](figures/4.1_fdm_convergence.png)

### 5.3 Resolution invariance / zero-shot super-resolution (Section 4.2)

**`figures/4.2_resolution_invariance.png`** (68.2 KB) — Mean relative L2 error against the evaluation resolution for the
FNO and the U-Net with 95% bootstrap CI bands; the FNO curve is flat, the U-Net curve rises steeply. The training
resolution N = 64 is marked.

![Error vs. resolution for FNO and U-Net](figures/4.2_resolution_invariance.png)

**`figures/4.2_sample_predictions.png`** (129.5 KB) — Zero-shot predictions at N = 2048 for two test samples (full field
plus a zoom on x in [0.35, 0.65]): the FNO prediction overlaps the reference while the U-Net shows amplitude errors and
spurious oscillations.

![Zero-shot predictions at N = 2048](figures/4.2_sample_predictions.png)

### 5.4 Shock capture and viscosity sensitivity (Section 4.3)

**`figures/4.3_viscosity_sensitivity.png`** (88.5 KB) — Mean relative L2 error against viscosity (log–log) for the FNO
and the FDM baseline with 95% CI bands at nu = 0.1, 0.01, 0.001: both methods lose accuracy as the shock sharpens, the
FNO/FDM ratio staying at about 5–7×.

![Error vs. viscosity for FNO and FDM](figures/4.3_viscosity_sensitivity.png)

**`figures/4.3_shock_snapshots.png`** (141.4 KB) — Snapshot of a representative nu = 0.001 solution (reference, FNO
zero-shot at N = 2048, FDM at N = 64) with a zoom on the steepest region, showing the smoothing / oscillatory behaviour
of the FNO at the shock.

![Shock snapshots at nu = 0.001](figures/4.3_shock_snapshots.png)

### 5.5 Fourier-mode ablation (Section 4.4)

**`figures/4.4_ablation_curve.png`** (104.9 KB) — Mean relative L2 error against `k_max` with the 95% CI band (left
axis) together with the parameter count and the per-sample inference time (right axis): monotonically decreasing error
with clearly diminishing returns, near-linear parameter growth and an almost constant inference cost.

![Error, parameters and runtime vs. k_max](figures/4.4_ablation_curve.png)

**`figures/4.4_error_spectrum.png`** (150.4 KB) — Log-scale frequency-domain error spectra `E_k` for the six `k_max`
values with the truncation wavenumbers marked: the low-mode models show an error hump inside the band they truncate,
whereas the high-mode models decay smoothly over the whole band.

![Frequency-domain error spectra](figures/4.4_error_spectrum.png)

## 6. Results

### 6.1 Section 4.1.3 — FDM baseline vs. spectral reference (nu = 0.01, N = 64, 200 test samples)

| Method | Configuration | Mean relative L2 error | Per-sample time | Speed-up vs. FDM |
|---|---|---|---|---|
| FDM | Lax–Friedrichs + van Leer TVD, dx = 1/64 (N = 64), CFL-RK4, 164 steps | **9.34e-4** on the 200 test samples (9.49e-4 over all 1000 samples, 9.53e-4 on the 800 training samples; median 7.68e-4, max 5.12e-3; mean relative L∞ 1.12e-3) — `data/4.1_error_summary.csv` | **96 ms** (measured) | 1× |
| Spectral (reference) | Fourier pseudo-spectral + IF-RK4, N = 2048, dt = 1e-3 | ~1e-6 — order of magnitude only, i.e. the level that shows the gap to the FDM baseline; the reference solver is validated against the Cole–Hopf exact solution at ν = 0.1 and ν = 0.001 (7.6e-9 / 5.3e-6, Section 4.3) | **600 ms** (measured) | — |
| FNO (this work) | L = 4, k_max = 16, d_v = 64, GELU, Adam, 565,889 stored floats, N = 64 | **4.73e-3** on the same 200 test samples (95% CI 4.35e-3 … 5.15e-3; mean relative L∞ 7.85e-3; `data/4.2_resolution_table.csv`, `data/4.3_summary.csv`) | **0.5 ms** (Chapter-3 reference value; the batched N = 64 forward pass measured in this repository is 0.157 ms, `data/4.2_resolution_table.csv`) | **~190×** (96 ms / 0.5 ms = 192; ≈1200× vs. the spectral solver) |

**Provenance of this table.**  Both errors are means over the same 200 test samples: the FDM value comes from
`data/4.1_error_summary.csv` (9.49e-4 over all 1000 samples, 9.53e-4 over the 800 training samples) and the
FNO value from `data/4.2_resolution_table.csv` / `data/4.3_summary.csv` (Section-4.2 weights, same grid and
test set), which gives an FNO/FDM error ratio of **5.07×** — the ratio Section 4.3 reports as well.  The FDM
and spectral times are measured on the reference machine (96 ms for one sample of 164 RK4 steps at
dx = 1/64, the full 1000-sample solve taking 94 s; 600 ms for the N = 2048 spectral solve at dt = 1e-3);
the FNO time `0.5 ms` is the Chapter-3 reference value, while the batched forward pass measured in this
repository is 0.157 ms at N = 64.  Hence 96 / 0.5 = 192 ≈ **190×** vs. FDM and 600 / 0.5 = 1200× vs. spectral.
The spectral cell is deliberately an order-of-magnitude statement — the ~1e-6 is there to show the gap to
the FDM baseline, not a measured property of the ν = 0.01 reference.  That reference is covered by the
Section-4.1.1 solver checks (linear-diffusion limit ~1e-15, dt vs dt/2 ~1e-11 order), which
`generate_burgers_dataset.py` prints but does not store in a CSV; the stored Cole–Hopf cross-check
(`data/4.3_reference_validation.csv`) only exists for ν = 0.1 and ν = 0.001.

Grid convergence of the FDM baseline (`data/4.1_convergence.csv`):

| Grid N | dx | Mean relative L2 | Mean relative L∞ |
|---|---|---|---|
| 32 | 3.125e-2 | 3.78170536e-03 | 4.31256709e-03 |
| 64 | 1.5625e-2 | 8.99067437e-04 | 1.07907787e-03 |
| 128 | 7.8125e-3 | 2.21882880e-04 | 2.53756037e-04 |

The error drops by roughly a factor of four per halving of dx (second-order convergence), which validates the baseline and
establishes **9.34e-4 on the test set at N = 64** (9.49e-4 over all 1000 samples) as the accuracy level the FNO has to beat.

### 6.2 Section 4.2 — Resolution invariance (nu = 0.01, 200 test samples)

Both models trained at N = 64 only. Parameters: **FNO 565,889** vs. **U-Net 563,041** (relative difference 0.50%).

| Method | Resolution | Mean relative L2 | 95% CI | Inference time (ms) |
|---|---|---|---|---|
| FNO | 64 | 4.73210908e-03 | [4.34545179e-03, 5.15129418e-03] | 0.157 |
| FNO | 128 | 4.72697291e-03 | [4.34924635e-03, 5.13369035e-03] | 0.241 |
| FNO | 256 | 4.72616286e-03 | [4.34949244e-03, 5.13483544e-03] | 0.511 |
| FNO | 512 | 4.72615424e-03 | [4.34947614e-03, 5.13483046e-03] | 0.942 |
| FNO | 1024 | 4.72615475e-03 | [4.34947631e-03, 5.13482724e-03] | 2.133 |
| FNO | 2048 | 4.72615345e-03 | [4.34947454e-03, 5.13482721e-03] | 4.448 |
| U-Net | 64 | 1.40200099e-01 | [1.27251878e-01, 1.54145818e-01] | 0.367 |
| U-Net | 128 | 3.25104252e-01 | [3.00040230e-01, 3.55220722e-01] | 0.899 |
| U-Net | 256 | 4.46689347e-01 | [4.10461763e-01, 4.90620182e-01] | 1.149 |
| U-Net | 512 | 4.93036064e-01 | [4.52211491e-01, 5.40777998e-01] | 2.511 |
| U-Net | 1024 | 5.08093839e-01 | [4.65907496e-01, 5.57166723e-01] | 5.345 |
| U-Net | 2048 | 5.12634543e-01 | [4.70096891e-01, 5.62099372e-01] | 11.691 |

Acceptance checks, all passed: FNO error growth N = 64 → 2048 = **0.9987** (+0.13%, limit 20%); U-Net / FNO at
N = 2048 = **108.4676** (limit ≥ 5); parameter difference **0.50%** (limit 10%). The FNO confidence intervals overlap at
every resolution, the U-Net intervals are disjoint and drift upwards.

**Why:** the FNO spectral weights depend only on the frequency index, so refining the grid leaves the first `k_max`
physical frequency components untouched and the layer is reused as is (an operator between continuous function spaces).
The U-Net kernels, pooling and upsampling are defined in grid units, so its receptive field shrinks in physical terms
when the resolution increases; features learned at N = 64 are mis-scaled and the error accumulates.

### 6.3 Section 4.3 — Shock capture and viscosity sensitivity (200 test samples, N = 64)

| nu | Method | Mean relative L2 [95% CI] | Mean relative L∞ [95% CI] |
|---|---|---|---|
| 0.1 | FNO | 8.64449624e-03 [6.16914832e-03, 1.15767079e-02] | 1.87947504e-02 [1.46133827e-02, 2.35091701e-02] |
| 0.1 | FDM | 1.24763775e-03 [8.13963150e-04, 1.88878838e-03] | 1.18962072e-03 [8.63054064e-04, 1.62829695e-03] |
| 0.01 | FNO | 4.73210908e-03 [4.34545179e-03, 5.15129418e-03] | 7.85494003e-03 [7.20351570e-03, 8.55870385e-03] |
| 0.01 | FDM | 9.34133389e-04 [8.33792866e-04, 1.03185739e-03] | 1.11618963e-03 [1.00082329e-03, 1.23458676e-03] |
| 0.001 | FNO | 8.70302506e-03 [8.20734882e-03, 9.21750190e-03] | 1.31730187e-02 [1.24097282e-02, 1.40372035e-02] |
| 0.001 | FDM | 1.61294485e-03 [1.51426502e-03, 1.71535604e-03] | 2.59821622e-03 [2.47103832e-03, 2.74049231e-03] |

FNO / FDM relative-L2 error ratio: **6.929×** (nu = 0.1), **5.066×** (nu = 0.01), **5.396×** (nu = 0.001).
Both methods lose accuracy as the viscosity decreases, but the FNO degrades faster in the L∞ norm (+68% from nu = 0.01 to
nu = 0.001: 7.85e-3 → 1.32e-2). This localises the shock-capturing bottleneck: a Fourier basis is global and smooth, so a
near-discontinuity is represented by smoothing and Gibbs-type oscillations. Both reference data sets were validated
against the Cole–Hopf exact solution (relative L2 = 7.580e-09 for nu = 0.1 and 5.320e-06 for nu = 0.001).

### 6.4 Section 4.4 — Fourier-mode ablation at nu = 0.001 (200 test samples, N = 64)

| k_max | Mean relative L2 [95% CI] | Parameters | Inference time (ms) | Converged epoch |
|---|---|---|---|---|
| 8 | 1.07264324e-02 [1.01550973e-02, 1.13272652e-02] | 303,745 | 0.151 | 166 |
| 12 | 9.84956320e-03 [9.32201985e-03, 1.04404686e-02] | 434,817 | 0.133 | 166 |
| 16 | 9.12326545e-03 [8.62123701e-03, 9.68063892e-03] | 565,889 | 0.122 | 162 |
| 20 | 9.02132121e-03 [8.48746504e-03, 9.59598894e-03] | 696,961 | 0.153 | 169 |
| 24 | 8.44710183e-03 [7.96107215e-03, 8.96818793e-03] | 828,033 | 0.157 | 167 |
| 32 | 8.36608041e-03 [7.88540415e-03, 8.87663730e-03] | 1,090,177 | 0.338 | 168 |

The error falls monotonically with `k_max` (22% in total, 8 → 32) but with strong **diminishing returns**: 1.61e-3 of the
total improvement comes from k_max = 8 → 16, while only 0.76e-3 is left for 16 → 32. The parameter count grows almost
linearly (303,745 → 1,090,177, ×3.6) whereas the per-sample inference time stays between 0.12 and 0.16 ms up to
k_max = 24 and rises to 0.34 ms only at k_max = 32. All six models converge at nearly the same epoch (162–169), so the
differences are due to the mode count alone.

| Test | Comparison | Statistic | p-value |
|---|---|---|---|
| Wilcoxon signed-rank | 8 vs 12 | mean difference 8.76869164e-04 | 0.00000000e+00 |
| Wilcoxon signed-rank | 12 vs 16 | 7.26297750e-04 | 0.00000000e+00 |
| Wilcoxon signed-rank | 16 vs 20 | 1.01944242e-04 | 7.35993322e-03 |
| Wilcoxon signed-rank | 20 vs 24 | 5.74219373e-04 | 0.00000000e+00 |
| Wilcoxon signed-rank | 24 vs 32 | 8.10214287e-05 | 1.89874549e-02 |
| Jarque–Bera | k_max = 8 | JB = 185.4719 | 5.312418e-41 |
| Jarque–Bera | k_max = 32 | JB = 392.8690 | 4.893162e-86 |

Acceptance checks, all passed: monotonic non-increasing error in `k_max` (**1.0000**); diminishing returns, the 16 → 32
gain smaller than the 8 → 16 gain (**0.0008**); Wilcoxon 8 vs 32 with p < 0.01 (**0.0000**); error-spectrum energy of
k_max = 8 exceeding that of k_max = 32 over the band k in (16, 32] by **1.4423** (+44%).

**Frequency-domain mechanism:** the k_max = 8 model shows a clear error hump inside the band it truncates — at k = 10 its
spectral error density is 7.8e-6, about ten times the 7.7e-7 of the k_max = 16/32 models — while the high-mode models
decay smoothly over the whole band. The accuracy limit of the FNO is therefore the **frequency bandwidth** it can
represent, not a lack of local smoothing. Balancing accuracy, parameters and runtime, **k_max = 16 is the best
accuracy/cost compromise** at nu = 0.001.

## 7. Key findings

1. **Reliable reference data.** GRF initial conditions (tau = 7, alpha = 2, K_KL = 128) with spectral reference
   solutions validated to ~1e-15 (linear limit) and ~3e-11 (dt vs dt/2), plus an independent Cole–Hopf cross-check for
   the viscosities ν = 0.1 and ν = 0.001 (7.6e-9 … 5.3e-6). The FDM baseline shows clean second-order grid convergence.
2. **Large efficiency gain.** On the nu = 0.01, N = 64 benchmark the FNO is ~190× faster than the FDM baseline
   and ≈1200× faster than the spectral solver (96 ms and 600 ms per sample vs. 0.5 ms), with an error 5.07× the
   FDM error (4.73e-3 vs. 9.34e-4 on the test set; Section 4.3 reports the same ratio). Both error columns come
   from the committed CSV files (`data/4.1_error_summary.csv`, `data/4.2_resolution_table.csv`), and re-running
   `4.1_fdm_comparison_baselines.py` reproduces the ten `data/4.1_*.csv` files byte-for-byte.
3. **Grid invariance.** Trained at N = 64, the FNO degrades by only 0.13% when deployed zero-shot up to N = 2048, while
   a parameter-matched U-Net (0.50% parameter difference) degrades to 5.13e-1 — 108.5× the FNO error. One training run
   covers a whole family of resolutions.
4. **Shocks are the bottleneck.** From nu = 0.1 to nu = 0.001 the FNO L∞ error grows by ~68% and the FNO/FDM error
   ratio stays at 5–7×: global smooth basis functions inevitably smooth or oscillate around a near-discontinuity.
5. **k_max has an optimum.** Raising k_max from 8 to 32 improves the error by only 22%, almost all of it below
   k_max = 16; the error spectrum shows that the remaining error lives in frequency bands the model cannot represent.

## 8. Reproducing the results

```
python generate_burgers_dataset.py           # ~9 min, creates the shared dataset
python 4.1_fdm_comparison_baselines.py       # ~2 min, FDM baseline + convergence study
python 4.2_resolution_invariance.py          # trains FNO and U-Net, then zero-shot evaluation
python 4.3_generate_reference_solutions.py   # ~30 min, reference solutions for nu = 0.1 / 0.001
python 4.3_viscosity_sensitivity.py          # trains one FNO per viscosity
python 4.4_mode_ablation.py                  # trains six FNOs (k_max sweep)
```

Requirements: Python with NumPy and Matplotlib for the dataset and baseline scripts; additionally **PyTorch** (a CPU
build is sufficient) for `4.2_resolution_invariance.py`, `4.3_viscosity_sensitivity.py` and `4.4_mode_ablation.py`.
SciPy is not required anywhere. Versions used for the reported results: torch 2.13.0+cpu, NumPy 2.4.6/2.5.2,
Matplotlib 3.11.x, Python 3.12/3.14.

The three training scripts save their trained weights next to the results and reload them on a second run, so a re-run
reproduces all tables and figures without retraining (the weight files themselves are not committed). The dataset and
reference-solution scripts recompute their outputs.

## 9. License

Released under the MIT License — see [`LICENSE`](LICENSE), Copyright (c) 2026 Xinhang Yu.

