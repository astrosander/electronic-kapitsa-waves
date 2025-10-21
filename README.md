# 1D Drift–Diffusion Spectral Simulator

A lightweight, reproducible research codebase for simulating 1D hydrodynamic drift–diffusion dynamics with nonlinear damping and optional electrostatics, using spectral (FFT) spatial discretization and stiff time integration (SciPy `solve_ivp`, BDF). The repository also includes a rich set of utilities for parameter sweeps and publication‑quality post‑processing/plots.

> **Purpose.** This README documents the model, dependencies, how to run the code/sweeps, the data products it writes, and how to regenerate the figures with the included plotting tools.

## Contents

* [Key features](#key-features)
* [Repository layout](#repository-layout)
* [Installation](#installation)
* [Quick start](#quick-start)
* [Model summary](#model-summary)
* [Parameters (dataclass `P`)](#parameters-dataclass-p)
* [Numerics & performance notes](#numerics--performance-notes)
* [Output files & directory structure](#output-files--directory-structure)
* [Post‑processing & plotting](#post-processing--plotting)
* [Reproducing common experiments](#reproducing-common-experiments)
* [Citing this repository](#citing-this-repository)

## Key features

* **Spectral (FFT) spatial derivatives** with optional 2/3 de‑aliasing.
* **Stiff time stepping** via SciPy BDF with configurable tolerances.
* **Nonlinear damping** $\Gamma(n) = \Gamma_0 e^{-\max(n, n_\text{floor})/w}$ and **optional localized dissipation** perturbation.
* **Optional Poisson coupling** for electrostatics (solve $-\nabla^2\phi = (e/\varepsilon)(n-\bar n)$).
* **Momentum/field feedback** to maintain a target drift velocity.
* **Initial‑condition generators** (multi‑mode cosine seeds, uniform background options).
* **Parallel parameter sweeps** (over drift `u_d`, nonlinear width `w`, and diffusion coefficients `Dn`, `Dp`).
* **Thread control** for FFT/BLAS/LAPACK via `NTHREADS` and `threadpoolctl`.
* **Rich plotting utilities** to inspect spectra, spacetime diagrams, velocities, pulse counts, and combined multi‑run analyses.

## Repository layout

```
.
├── main.py                           # Core model, integrator, sweeps & save routines
├── plot_from_data.py                 # Post-processing & figure generation utilities
├── solve_PDE.py                      # PDE solver implementation
├── test_linear_instability.py        # Linear instability analysis
├── compare analytics vs. PDE.py      # Analytical vs numerical comparison
├── animation/                        # Animation generation scripts and outputs
├── linear instability increment/     # Linear instability analysis and phase diagrams
├── mass injection/                   # Mass injection experiments and analysis
├── dn vs u_d/                        # Density vs drift velocity analysis
├── diffusion in n and p/             # Diffusion analysis
├── Hydraulic jump/                   # Hydraulic jump analysis
├── multiple_u_d/                     # Parameter sweep results
├── out_drift/                        # Output data and figures
└── README.md                         # This document
```

## Installation

**Python:** 3.9–3.12 recommended

Create an isolated environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip wheel
pip install -r requirements.txt    # if present; otherwise install packages below
```

Minimal packages used by the repo:

```txt
numpy
scipy
matplotlib
threadpoolctl        # optional but recommended
```


**Threading**: set the environment variable `NTHREADS` (default 1). `main.py` propagates it to FFT (`scipy.fft.set_workers`) and BLAS/LAPACK via `threadpoolctl` when available.

```bash
export NTHREADS=4  # e.g., run with 4 threads
```

## Quick start

Run a single simulation with the defaults defined in the parameter dataclass `P`:

```bash
python main.py
```

By default the `__main__` block demonstrates one of the bundled experiments (see comments near the bottom of `main.py`). Toggle lines there to run:

* a single run: `run_once(tag=...)`
* a drift sweep: `run_multiple_ud()`
* a nonlinearity‑width sweep: `run_multiple_w()`
* a diffusion sweep: `run_diffusion_parameter_sweep()`

**Typical workflow**

1. Edit the `P()` defaults or override them programmatically.
2. Choose **one** experiment launcher in the `if __name__ == "__main__":` block.
3. Run `python main.py`. Data and figures are written under the configured `outdir`.
4. Use `plot_from_data.py` to regenerate publication figures from saved `.npz` files.

## Model summary

Let $x\in[0,L)$ with periodic boundary conditions and denote density $n(x,t)$ and momentum $p(x,t)$. The code evolves

$$\partial_t n = -\partial_x p + D_n \partial_{xx} n$$

$$\partial_t p = -\Gamma_\text{spatial}(n) p - \partial_x\Pi(n,p) + e n E_\text{eff} - n \partial_x\phi + D_p \partial_{xx}p$$

with $\Pi(n,p) = \tfrac{1}{2}U n^2 + \tfrac{p^2}{m n}$, optional electrostatic potential $-\partial_{xx}\phi = (e/\varepsilon)(n-\bar n)$, and nonlinear damping $\Gamma(n)=\Gamma_0\exp[-\max(n,n_\text{floor})/w]$.

A localized dissipation perturbation can be added (Gaussian of amplitude `lambda_diss` and width `sigma_diss` centered at `x0`):

$$\Gamma_\text{spatial}(x,n)=\Gamma(n)+ \lambda_\text{diss} \exp\left[-\frac{(x-x_0)^2}{2\sigma_\text{diss}^2}\right]$$

A uniform target drift `u_d` is maintained either by a constant field `E_base` (mode `maintain_drift="field"`) or with proportional feedback $E_\text{eff}=E_\text{base}+K_p(u_d-\langle v \rangle)$ (mode `"feedback"`).

Spatial derivatives use FFTs; an optional **2/3 de‑alias** filter is available. Time integration uses SciPy’s BDF with tolerances `rtol`, `atol`.

**Initial conditions.** Several seed modes are offered. For example `seed_mode=7` seeds cosine modes m = 3,5,7; `seed_mode=2` uses a multi‑mode sum (e.g., 2,3,5,8,13,21,34,55). Uniform backgrounds are used for modes 2 and 7. See `initial_fields()` for details.

> The source term `S_injection` is implemented but **disabled** in the default RHS for speed (see comments in `rhs`). Enable if needed.

## Parameters (dataclass `P`)

Parameters live in `main.py` as a dataclass `P`. Defaults are shown below with brief notes.

| Name                                          | Type / Default                               | Meaning                                                              |
| --------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------- |
| `m`                                           | `1.0`                                        | Effective mass.                                                      |
| `e`                                           | `1.0`                                        | Charge.                                                              |
| `U`                                           | `1.0`                                        | Equation‑of‑state parameter in $\Pi_0(n)=\tfrac12 U n^2$.            |
| `nbar0`                                       | `0.2`                                        | Uniform background density $\bar n$.                                 |
| `Gamma0`                                      | `2.50`                                       | Base damping scale.                                                  |
| `w`                                           | `0.04`                                       | Nonlinear damping width in $\Gamma(n)$.                              |
| `include_poisson`                             | `False`                                      | Solve Poisson for `phi_from_n` if `True`.                            |
| `eps`                                         | `20.0`                                       | Permittivity $\varepsilon$.                                          |
| `u_d`                                         | `5.245`                                      | Target drift velocity.                                               |
| `maintain_drift`                              | `"field"`                                    | `"field"` or `"feedback"` to maintain `u_d`.                         |
| `Kp`                                          | `0.15`                                       | Proportional gain for feedback mode.                                 |
| `Dn`                                          | `0.5`                                        | Density diffusion.                                                   |
| `Dp`                                          | `0.1`                                        | Momentum diffusion.                                                  |
| `J0`, `sigma_J`, `x0`, `source_model`         | `1.0`, `sqrt(2)`, `12.5`, `"as_given"`       | Injection profile controls (source term **off by default** in RHS).  |
| `lambda_diss`, `sigma_diss`                   | `0.0`, `2.0`                                 | Localized dissipation amplitude & width. Set amplitude 0 to disable. |
| `lambda_gauss`, `sigma_gauss`, `x0_gauss`     | `0.0`, `2.0`, `12.5`                         | Time‑independent Gaussian density perturbation (off by default).     |
| `use_nbar_gaussian`, `nbar_amp`, `nbar_sigma` | `False`, `0.0`, `120.0`                      | Alternate background shaping; off by default.                        |
| `L`                                           | `10.0`                                       | Domain length.                                                       |
| `Nx`                                          | `1212`                                       | Grid points (periodic).                                              |
| `t_final`, `n_save`                           | `50.0`, `100`                                | Final time and number of saved steps.                                |
| `rtol`, `atol`                                | `1e-4`, `1e-7`                               | ODE tolerances.                                                      |
| `n_floor`                                     | `1e-7`                                       | Lower bound for density in divisions.                                |
| `dealias_23`                                  | `True`                                       | 2/3 de‑aliasing for spectral products.                               |
| `seed_amp_n`, `seed_amp_p`                    | `0.03`, `0.03`                               | Amplitudes for seeded density/momentum modes.                        |
| `seed_mode`                                   | `7`                                          | Chooses mode set; see `initial_fields()`.                            |
| `outdir`                                      | `"out_drift/small_dissipation_perturbation"` | Output directory (many sweep functions override this).               |
| `cmap`                                        | `"inferno"`                                  | Colormap for figures.                                                |

You can modify parameters either by editing `P()` defaults, or by constructing and assigning a new `P` inside helper workers (as sweep utilities do).

## Numerics & performance notes

* **FFTs**: `scipy.fft.fft/ifft` with explicit `workers=NTHREADS`.
* **Threading**: set `export NTHREADS=...`. The code attempts to set BLAS/LAPACK threads via `threadpoolctl` and `scipy.linalg` APIs when present.
* **Progress prints**: during integration, a single‑line progress indicator shows simulated time, wall time, and a rough ETA.
* **Stability**: `dealias_23=True` applies a 2/3 spectral filter to help control aliasing.

## Output files & directory structure

Each run writes compressed NumPy archives and figures under `outdir`.

### Data archives (`*.npz`)

Saved via `save_final_spectra(...)` with fields:

* `m` (int): seed mode index
* `t` (1D array): saved time points
* `n_t` (2D array, shape `Nx × n_save`): density field over time
* `p_t` (2D array): momentum field over time
* `L` (float): domain length
* `Nx` (int): grid size
* `meta` (dict): a snapshot of all parameters (`asdict(par)`) and `outdir`

### Figures (examples)

* `spacetime_n_lab_*.png/pdf`: spacetime density in lab frame
* `spacetime_n_comoving_*.png/pdf`: in frame co‑moving at `u_d`
* `snapshots_n_*.png/pdf`: selected spatial slices
* `fft_compare_*.png/pdf`: initial vs final spectra
* `period_detection_*.png/pdf`: robust shift/velocity detection at late times
* Various panel/overlay/summary plots for sweeps (see below)

Sweep utilities programmatically set `outdir` to encode parameters in the path (e.g., `multiple_u_d/.../out_drift_ud4p6000/`).

## Post‑processing & plotting

All plotting helpers live in **`plot_from_data.py`**. Import them in a script or an interactive session to (re)generate publication figures from saved `.npz` files.

### Load data

```python
from plot_from_data import load_data
D = load_data("path/to/data_m03_ud4p6000_tag.npz")
```

### Core visualization utilities

* `plot_spacetime_lab(D)` and `plot_spacetime_comoving(D, u_d)`
* `plot_snapshots(D)`
* `plot_fft_compare(D)`
* `plot_velocity_detection(D, u_d)` (cross‑correlation/shift method)
* `plot_velocity_evolution(D, u_d)` (time‑trace of measured drift and moments)
* `plot_velocity_field(D, u_d)` (plots `n(x)`, `p(x)`, and `v(x)=p/(m n)`)

### Multi‑run summaries (parameter sweeps)

* `plot_velocity_vs_ud(data_files)` – measured `u_true` vs target `u_d`, pulse counts, frequency.
* `plot_multiple_ud_panel(...)` / `plot_multiple_ud_panel_p(...)` – panel figures of final profiles across `u_d` and across different base directories (e.g., different `w`).
* `plot_overlay_final_profiles_n(...)` / `plot_overlay_final_profiles_p(...)` – overlays of final fields.
* `plot_combined_velocity_analysis(base_dirs, labels)` – compare several datasets.
* `plot_delta_n_vs_ud(base_dirs, labels)` – publication‑ready joint plot of amplitude $\Delta n$ and mean current $\langle j\rangle$ vs `u_d`, with optional fits.

All plot functions save figures next to the input data (or under a common `multiple_u_d/` directory for combined plots) and use a STIX font setup suitable for manuscripts.

## Reproducing common experiments

> Edit the `__main__` block in `main.py` to enable **one** of the following at a time.

### 1) Single run (custom tag)

```python
# In main.py (__main__):
par.lambda_diss = 0.0
par.sigma_diss  = -1.0    # negative width effectively disables the localized term
par.seed_mode   = 7
par.seed_amp_n  = 0.03
par.seed_amp_p  = 0.03
par.L           = 10.0
par.t_final     = 50.0
par.include_poisson = False
run_once(tag="example_run")
```

`outdir` defaults to `out_drift/small_dissipation_perturbation/`. Adjust as needed.

### 2) Drift sweep `u_d`

```python
run_multiple_ud()   # sweeps u_d ∈ {0.2, 0.3, …, 1.9}
```

This spawns parallel workers (up to `cpu_count()-1`) and stores each run under `multiple_u_d/.../out_drift_ud{u_d}`.

### 3) Width sweep `w` (and `u_d` grid)

```python
run_multiple_w()    # w ∈ 0.01…0.15, u_d ∈ 0.1…1.9
```

### 4) Diffusion sweep `(Dn, Dp)` across `w × u_d`

```python
run_diffusion_parameter_sweep()
```

A large grid of runs across six diffusion settings (halved/doubled combinations), `w ∈ 0.01…0.25`, and `u_d ∈ 0.1…1.9`.

### 5) Detect and fill missing runs in a sweep

```python
check_and_run_missing_simulations()
```

Scans the diffusion‑sweep directory tree for missing folders/data and re‑runs only what’s missing.

## Citing this repository

If you use this code or figures in a publication, please cite the repository.

### BibTeX

```bibtex
@software{drift_spectral_simulator,
  title        = {1D Drift--Diffusion Spectral Simulator},
  author       = {[Authors to be specified]},
  year         = {[Year to be specified]},
  version      = {[Version to be specified]},
  url          = {[Repository URL to be specified]},
  note         = {Python, SciPy BDF integrator, spectral derivatives}
}
```

## License

[License to be specified]

## Acknowledgments

Built on NumPy/SciPy/Matplotlib. Please cite upstream packages as appropriate.
