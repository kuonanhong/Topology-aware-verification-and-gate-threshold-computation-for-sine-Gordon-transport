# Transformation-informed gating and topology-aware verification of sine-Gordon transport

Reproducible Python code, numerical data, and figure-generation tools for the sine-Gordon research manuscript and technical report prepared by Nan-Hong Kuo.

Repository: <https://github.com/kuonanhong/Topology-aware-verification-and-gate-threshold-computation-for-sine-Gordon-transport>

This repository separates mathematical representations, physically changed coefficients, and measured numerical observables. A modular or Landen transformation is not treated as an external physical actuator. The new control construction specifies the coefficient profile that must be implemented, solves the resulting field equation, and checks energy and topology independently.

## What is reproduced

1. **Elliptic reference calculations.** Jacobi identities, elliptic periods, winding quadrature, integer-matrix/parity checks, static branches, and Lamé stability spectra.
2. **Overdamped double-sine-Gordon ring.** A phase-driven dissipative system with fixed winding. Static continuation and geometric cycle transport are compared with time-dependent PDE calculations and an independent BDF implementation.
3. **Static inertial sine-Gordon gate.** An incoming kink scatters from a preset anisotropy barrier. A screened local parameter interval is refined, with transmission, reflection, unresolved, and rejected-numerics labels preserved.
4. **Transformation-informed coefficient design.** A specified mathematical profile or coordinate map is converted into explicit physical coefficients. The new validation tests the required coefficient implementation instead of applying a representation change to a state and calling it a dynamical transition.

All calculations use dimensionless classical fields. Winding `Q`, finite-gap phase number/genus `g`, and spatial dimension `d` are distinct. Detector passage is a topological field observable; it is not a calibrated current in amperes or an electrical transistor gain.

## Installation

Use Python 3.12, or a compatible recent Python version. A GPU, login, and external dataset are not needed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The baseline package uses NumPy, SciPy, Matplotlib, and mpmath. The tested version information is recorded in the run receipts; dependency ranges are declared in `requirements.txt`. For numerically comparable CPU timing, the driver restricts BLAS threads to one. Timings depend on the machine and concurrent work.

## Reproduce the new manuscript computations (Figures 1–4)

Run from the repository root:

```bash
mkdir -p results audit
python code/run_transformation_suite.py --out results
python code/make_transformation_figures.py --results results --out results
```

`code/transformation_gate.py` is the importable solver module; executing that file alone does not generate the manuscript results. `run_transformation_suite.py` performs the simulations and numerical checks. `make_transformation_figures.py` reads the CSV/JSON/NPZ outputs and renders four main paper figures and an optional trial-energy comparison. Read `results/transformation_gate_summary.json`, `audit/execution_receipt.json`, and `audit/figure_receipt.json`; the declared checks must pass before the results are used.

| New manuscript figure | Figure filename (PDF and PNG) | Numerical source | Script dependency |
|---|---|---|---|
| Figure 1: coordinate-designed coefficients and exact-pullback residual | `transformation_design` | `matched_residuals.csv`; analytic `mapping` and `coefficients` definitions | `transformation_gate.py` → `run_transformation_suite.py` → `make_transformation_figures.py` |
| Figure 2: matched-field convergence and delay | `matched_convergence_delay` | `matched_spatial_convergence.csv`, `matched_temporal_convergence.csv`, `matched_delay.csv`, `matched_eta0.25_trace.csv`, `matched_eta0.5_trace.csv` | Same simulation-to-plot chain |
| Figure 3: mapped versus virtual barrier calculations | `mapped_virtual_comparison` | `mapped_gate_comparison.csv`, `mapped_G0.1_h0.1_comparison_trace.csv`, `mapped_G0.25_h0.1_comparison_trace.csv`, corresponding mapped/virtual traces and field NPZ files | Same simulation-to-plot chain; equal virtual initial state and pulled-back detector |
| Figure 4: fixed-anisotropy compensation gate | `coordinated_fixedK_gate` | `fixed_anisotropy_screen.csv`, `fixed_anisotropy_brackets.csv`, `fixed_anisotropy_bracket_runs.csv`, `fixed_anisotropy_coefficients.csv`, `fixedK_lambda0_trace.csv`, `fixedK_lambda1_trace.csv`, corresponding field NPZ files, predictor value in `transformation_gate_summary.json` | Same simulation-to-plot chain |
| Optional supplementary figure: static trial-energy predictor | `coordinated_trial_energy` | `fixed_anisotropy_trial_energy.csv`, `fixed_anisotropy_brackets.csv`, predictor value in `transformation_gate_summary.json` | Same simulation-to-plot chain; compares a trial-profile approximation with the dynamic local bracket |

Additional new verification data are `matched_independent_time_check.csv`, `all_run_diagnostics.csv`, per-case summary JSON files, and `transformation_gate_summary.json`. The independent DOP853 calculation tests time integration on the same conservative spatial model; it is not an independent spatial discretization.

The figure filenames identify the source unambiguously even if a later journal changes figure numbering. The final article captions take precedence over this release's numbering.

## Reproduce the baseline computations

Run from the repository root:

```bash
python run_all.py
```

The driver executes each calculation in dependency order, writes logs to `audit/`, and stops if a declared numerical check fails. It records the source hashes, package versions, measured elapsed times, and pass/fail status in `audit/reproduction_receipt.json`. Do not report a result from a failed or incomplete run.

For an individual calculation, use:

```bash
mkdir -p results
python code/elliptic_verification.py --out results
python code/dsg_continuation.py --output results
python code/dsg_independent_check.py
python code/ring_multisector_check.py
python code/gate_threshold.py --out results
python code/gate_refinement_audit.py
python code/make_gate_audit_figures.py
```

`dsg_independent_check.py` reads the continuation outputs and must run after `dsg_continuation.py`. `gate_refinement_audit.py` reads/imports the baseline gate implementation; run it after `gate_threshold.py`. `make_gate_audit_figures.py` needs both gate runs. The scripts without output arguments use the repository's `results/` directory.

A quick pilot is useful for exploration, but does not regenerate the manuscript's full verification:

```bash
python code/gate_threshold.py --pilot --out results
```

The baseline main gate program expects exactly one sampled transmission-to-reflection window. If there are multiple windows, no window, or an unresolved/vetoed midpoint, it stops or preserves the corresponding stop reason. It does not silently impose global monotonicity.

## Models, acceptance rules, and observables

The baseline dissipative ring is

```text
u_t = u_xx - sin(u) - 2 eta sin(2u + chi(t)),
u(x + L,t) = u(x,t) + 2 pi Q.
```

An autonomous overdamped model of this type cannot sustain an inertial sine-Gordon breather. Its results must not be conflated with those of the inertial gate.

The baseline scattering model is

```text
u_tt - u_xx + [1 + G sech^2(x/3)] sin(u) = 0,
u(-160,t) = 0,  u(160,t) = 2 pi.
```

The incoming continuum kink has speed `v=0.35` and initial center `x0=-24`. The gate is static and preset; its electrical charging cost and switching bandwidth are not included.

For either smooth model,

```text
rho_top = u_x/(2 pi),
j_top = -u_t/(2 pi),
P_detector(t) = -[u(x_detector,t) - u(x_detector,0)]/(2 pi).
```

The baseline detector is at `x_detector=10`. Reflection and transmission retain the same total winding `Q=1`; their different detector passage supports routing. A breather has `Q=0`, but a finite detector interval can still contain a nonzero local flux. No fixed-boundary breather-to-single-kink conversion is inferred from that flux.

Before velocity-Verlet integration, the conservative stability bound

```text
dt sqrt(4/dx^2 + max(abs(a))) < 2
```

is checked. The trajectory's physical label is vetoed as `rejected_numerics` if relative discrete energy defect exceeds `1e-3`, detector flux discrepancy exceeds `1e-3`, winding drift exceeds `1e-14`, or a required field/center diagnostic is nonfinite. The raw outcome and reason are exported. The extended audit exercises both rejection paths and an exhausted iteration budget.

The Hamiltonian defect checks the implemented discrete model. It is not a rigorous bound on continuum solution error. Spatial and temporal convergence and independent temporal integration are therefore reported separately.

## Mathematical map to a physical coefficient gate

Choose a smooth increasing coordinate map

```text
y = f_eta(x) = x + eta w tanh(x/w),
J_eta(x) = 1 + eta sech^2(x/w),  eta > -1.
```

Pulling back the homogeneous action produces

```text
rho = J_eta,  kappa = 1/J_eta,  a = J_eta,
rho u_tt - d_x(kappa u_x) + a sin(u) = 0.
```

All three coefficients must be implemented. At exact matching, the incoming kink is `u(x,t)=U(f_eta(x),t)` and its center has speed `v/J_eta`. The layer changes spatial width and delay, while transmitting the kink in the ideal lossless continuum model. Increasing anisotropy alone does not implement this cancellation.

Two control families are included:

- **Pulled-back virtual barrier:** `a=J_eta[1+G sech^2(f_eta(x)/3)]`, with `rho=J_eta` and `kappa=1/J_eta`. This keeps the continuous dynamics conjugate to the original virtual barrier problem. A comparison must hold virtual input, detector position, and observation horizon consistent; the exact threshold is then independent of `eta`.
- **Fixed-anisotropy compensation:** hold `a=J_eta` fixed and set `R_lambda=1+lambda eta sech^2(x/w)`, `rho=R_lambda`, `kappa=1/R_lambda`, with `0<=lambda<=1`. `lambda=0` gives an anisotropy-only barrier; `lambda=1` gives the matched, transmitting layer. Intermediate values reduce the mismatch in the transformed equation. The trial-profile energy decreases with `lambda`, but this does not prove globally monotone scattering outcomes.

A preset compensation setting is held static throughout each run. A time-dependent setting requires inertia/stiffness/anisotropy work terms and cannot be represented by simply substituting a time-dependent `J` in the static solver. A physical antiferromagnet requires calibrated control of effective inertia, exchange stiffness, and anisotropy. A voltage-controlled anisotropy electrode alone does not establish the required coupled profile.

A conditional microscopic route is available in the bare continuum limit of a nearest-neighbor bipartite antiferromagnetic chain. For fixed spin length `S` and lattice spacing `a_lat`, the sublattice angular-momentum density, exchange stiffness, susceptibility and inertia satisfy

```text
I = hbar S/(2 a_lat),
A = j_ex S^2 a_lat,
chi = a_lat/(j_ex S^2),
rho = I^2 chi = hbar^2/(4 j_ex a_lat).
```

Thus an exchange target `j_ex(x,lambda)=j_ex,0/R_lambda(x)` co-tunes `rho/rho_0=R_lambda` and `A/A_0=1/R_lambda`. The dictionary is given by Kim, Tserkovnyak and Tchernyshyov, *Physical Review B* **90**, 104406 (2014), [Appendices A and C](https://arxiv.org/pdf/1406.6051). Its use requires a slowly varying exchange profile relative to the lattice, fixed spin density, a valid planar semiclassical reduction, and calibration of the anisotropy maintained along the path. Quantum renormalization, further exchange interactions, and distinct interlayer/intralayer coupling in synthetic antiferromagnets can change these ratios. In the `u=2 theta` convention, a dimensionless kink energy of `8` corresponds to physical wall energy `2 sqrt(A_0 K_0)`.

[Kossak et al., *Science Advances* **9**, eadd0548 (2023)](https://doi.org/10.1126/sciadv.add0548) provide an experimental precedent for voltage-controlled RKKY exchange in multilayers. Applying it to this gate remains a material-control proposal. Measure a response map `(j_ex,K)=F(V_1,V_2,...)`, then solve a bounded inverse problem for the target exchange and fixed anisotropy. Report response-matrix rank, voltage/coefficient bounds and residual errors; insufficient rank or inaccessible targets invalidate the realization. These simulations do not establish a fabricated transistor, its electrical gain, or its voltage and energy performance.

The new physical-grid discretization uses midpoint stiffness on edges and a conservative divergence of flux, followed by mass-weighted velocity Verlet. Its weighted discrete Hamiltonian is checked, alongside detector-current continuity and fixed winding. This remains a standard second-order numerical method; the contribution under study is the constructive coefficient design and validation.

## Baseline script-to-data-to-figure map

| Script | Principal numerical outputs in `results/` | Figures or manuscript use |
|---|---|---|
| `code/elliptic_verification.py` | `elliptic_identities.csv`, `elliptic_periods.csv`, `elliptic_residuals.csv`, `elliptic_winding_quadrature.csv`, `elliptic_theta2_parity.csv`, `elliptic_integer_matrix.json`, `elliptic_branch_convergence.csv`, `elliptic_ode_convergence.csv`, `elliptic_branch_profiles.npz`, `elliptic_lame_convergence.csv`, `elliptic_lame_bands.csv`, `elliptic_summary.json` | Generates `elliptic_verification.pdf/png` and `elliptic_lame_bands.pdf/png`; elliptic identity/static-profile/stability checks. |
| `code/dsg_continuation.py` | `dsg_static_branches.csv`, `dsg_geometric_prediction.csv`, `dsg_sg_limit.csv`, `dsg_hessian.csv`, `dsg_geometric_refinement.csv`, `dsg_pde_cycles.csv`, `dsg_pde_refinement.csv`, `dsg_pde_trace.csv`, `dsg_pde_fields.npz`, weak-amplitude/length CSV files, verification JSON files | Generates `dsg_geometric_transport.pdf/png`, `dsg_diagnostics.pdf/png`, and `dsg_length_dependence.pdf/png`; static continuation, ring transport, convergence, energy and length checks. |
| `code/dsg_independent_check.py` | `dsg_independent_connections.csv`, `elliptic_compressibility.csv`, `dsg_independent_verification.json` | Independent quadrature/parameter-derivative/BDF values supporting the ring numerical comparison. |
| `code/ring_multisector_check.py` | `ring_multisector_check.json` | Repeated-cell `Q=2,L=8` versus `Q=1,L=4` verification. This compares fixed sectors; it does not simulate a change of winding. |
| `code/gate_threshold.py` | `gate_screen.csv`, `gate_bisection.csv`, `gate_refinement.csv`, `gate_refinement_runs.csv`, `gate_controls.csv`, `gate_on_trace.csv`, `gate_off_trace.csv`, `gate_breather_trace.csv`, corresponding field NPZ files, `gate_uniform_baseline.csv`, `gate_summary.json` | Generates `gate_threshold.pdf/png`, `gate_diagnostics.pdf/png`, and `gate_spacetime.pdf/png`; gate outcomes, detector passage, conservation, and parameter-sampling cost. |
| `code/gate_refinement_audit.py` | `gate_saddle_energy.csv`, `gate_fine_refinement.csv`, `gate_fine_refinement_runs.csv`, `gate_time_refinement.csv`, `gate_independent_time_check.csv`, `gate_extended_audit.json` | Fine threshold intervals, relaxed saddle-energy predictor, independent DOP853 comparison, temporal convergence, and acceptance-contract checks. |
| `code/make_gate_audit_figures.py` | Reads the preceding gate CSV/JSON files; performs no additional PDE integration | Generates `gate_predictors_convergence.pdf/png`; predictor, spatial/time refinement, and independent-integrator comparisons. |
| `article_figures/prepare_article.py` | Reads the numerical `results/` directory; writes bilingual LaTeX numbers/tables | Generates `reference_checks.pdf/png`, `gate_overview.pdf/png`, and `gate_audit.pdf/png` under the selected output's `figures/`; the composite baseline figures used in the previous revised article. |

To rebuild those composite article figures and numeric fragments without rerunning the PDEs:

```bash
python article_figures/prepare_article.py --results results --out article_generated
```

Figure numbers can change between manuscript versions. The filenames and data dependencies above identify the computation unambiguously; the final manuscript captions identify which panels are reproduced. The figure generator reads data rather than inserting illustrative or guessed values.

## What to upload to GitHub

Upload the complete release tree, preserving relative paths:

- `README.md`, `LICENSE`, `requirements.txt`, and the execution driver.
- All seven baseline scripts under `code/`, plus `transformation_gate.py`, `run_transformation_suite.py`, and `make_transformation_figures.py`.
- `article_figures/prepare_article.py`, because the individual numerical scripts alone do not recreate the previous article's composite layout.
- The full `results/` CSV, JSON, NPZ, and figure outputs associated with the release.
- `audit/` run receipts, source hashes, and relevant numerical logs.
- Manuscript sources and figure assets if desired; preserve any bundled fonts' own license files.

Do not substitute old placeholder illustrations, a smooth animation between unrelated formulae, or an unvalidated legacy script for the numerical source of a paper figure. The supplied archive inventory and review identify historical illustration/fragment files separately from executed scientific outputs. Third-party publications and downloaded journal-policy PDFs are reference material, not code to redistribute under this license.

As checked on 2 October 2026, the public repository contained a one-line README and the seven baseline scripts under `code/`; it did not yet contain this release's documentation, data, dependency file, or license. The accompanying upload-ready package is the complete proposed update. No upload is performed by this README.

## Numerical claims and reproducibility

The baseline finite-time refined interval is approximately

```text
G in [0.14974365234375, 0.149755859375]
```

for the stated speed, barrier width, domain, classifier, and refined discretization. This is a finite-time local outcome interval, not a global mathematical bifurcation certificate. The symmetric static-saddle energy gives a close independent predictor but does not account exactly for radiation in the scattering orbit.

For the new fixed-anisotropy compensation path (`eta=0.25`, `w=3`, `v=0.35`, `T=240`), the three tested space/time settings give the local reflected-to-transmitted interval

```text
lambda in [0.35546875, 0.3564453125].
```

The independent static trial-profile prediction is approximately `lambda=0.371602`. It differs from the dynamic interval and is not used to initialize the five-point screen. Agreement of the three finite-grid intervals is an audit at the exported tolerance; it does not prove an exact continuum threshold or exclude narrow resonance windows.

A fair search-cost comparison uses the same coarse-screened interval for local bisection and a uniform sweep, counts each strategy's endpoint evaluations, and excludes the same shared screen from both. PDE-call and force-evaluation counts are more portable than wall time. This comparison is between parameter-sampling strategies, not between new and old time integrators.

Every release should include the outputs generated from the same source hashes used in its manuscript. Keep failed-run outputs separately, and retain unresolved outcomes. Tag the submission version; an archival DOI may be added after an actual deposit. No DOI is invented here.

## Citation and license

Cite the manuscript and the repository release corresponding to the calculations you use. Until a paper or archival release has a persistent identifier, cite the repository URL with the access date and actual commit or tag. The repository records the research version; it does not imply journal acceptance or a permanent archival DOI.

The project code is supplied under the MIT License; see `LICENSE`. Copyright (c) 2026 Nan-Hong Kuo. Dependencies, third-party assets, and bundled fonts retain their own licenses. 

## Acknowledgements

Nan-Hong Kuo thanks C. D. Hu of National Taiwan University.

## Optional combined driver and release integrity

`python run_complete_study.py` runs the baseline and new study in order; `python run_complete_study.py --new-only` executes only the new coefficient suite and figure builder. Baseline receipts are retained from the preceding verified revision; the new suite was executed on 2 October 2026. `RELEASE_MANIFEST.json` records every distributed file's SHA-256. The archive review covers all 41 historical Python programs, but the defective historical programs themselves are not included as MIT release code.

The new manuscript uses the symbol `epsilon` for coordinate deformation; the code's `eta` parameter represents that symbol. The baseline double-SG harmonic `eta=0.2` is a different parameter.
