# Nonlinear Filtering Benchmark on Lorenz-96

Companion code for:
> "A Survey of Nonlinear Filtering Methods for Data Assimilation and Recent Advances in Generative AI-Enabled Score Filters"
> Cath Pham, Florida State University, 2026

Implements and benchmarks four data assimilation filters on the Lorenz-96 system with nonlinear arctan observations:
- **PF** — Bootstrap Particle Filter (Gordon et al. 1993)
- **EnKF** — Ensemble Kalman Filter (Evensen 1994)
- **EnSF** — Ensemble Score Filter (Bao et al. 2024)
- **LETKF** — Local Ensemble Transform Kalman Filter (Hunt et al. 2007)

---

## Files Organization

1. `filters` contains the implementation of all four data assimilation filters.
   - `pf_final.py` runs the Bootstrap Particle Filter (Gordon et al. 1993).
   - `enkf_final.py` runs the Ensemble Kalman Filter (Evensen 1994).
   - `ensf_final.py` runs the Ensemble Score Filter (Bao et al. 2024).
   - `letkf_final.py` runs the Local Ensemble Transform Kalman Filter (Hunt et al. 2007).
   - Set `n_dim` at the top of each script, then run for each dimension in `[10, 50, 100, 500, 1000]`.
   - Each run saves `rmse_{filter}_d{n_dim}.npy` and `time_{filter}_d{n_dim}.npy` locally. These files are not uploaded to this repository but are required to run the figure scripts.

2. `figures` contains the plotting scripts for all paper figures.
   - `fig1_rmse_vs_dim.py` generates Figure 1: RMSE vs state dimension with confidence bands.
   - `fig2_rmse_vs_step.py` generates Figure 2: RMSE over filtering steps at `d = 10, 50, 100, 500`.
   - `fig3_ensf_vs_letkf.py` generates Figure 3: EnSF vs LETKF accuracy and wall-clock cost.
   - `fig4_trajectory_spread.py` generates Figure 4: trajectory tracking and ensemble spread at `d = 10`.
   - Run all filter scripts first to generate the `.npy` files, then place them in the same directory as the figure scripts before running.

3. `results` contains the final paper figures as `.png` files.
   - Figures can be viewed directly without running any code.
   - To regenerate, run the filter scripts followed by the figure scripts.

---

## Key Parameters

All parameters are defined at the top of each filter script:

| Parameter | Default | Description |
|---|---|---|
| `n_dim` | 10 | State dimension — **change this per run** |
| `ensemble_size` | 100 | Ensemble / particle count |
| `n_seeds` | 10 | Number of independent random seeds |
| `filtering_steps` | 500 | Number of filtering steps |
| `dt` | 0.005 | Integration timestep |
| `F` | 8 | Lorenz-96 forcing constant |
| `obs_sigma` | 0.1 | Observation noise standard deviation |
| `dyn_sigma` | 0.1 | Process noise standard deviation |

### EnSF only

| Parameter | Default | Description |
|---|---|---|
| `euler_steps` | 50 | Number of reverse SDE steps K |
| `eps_alpha` | 0.05 | Alpha schedule regularization |
| `eps_beta` | 0.001 | Beta schedule regularization |

### LETKF only

Localization and inflation are set automatically per dimension via a lookup table tuned:

| d | inflation | neighbor_size |
|---|---|---|
| 10 | 1.00 | 4 |
| 50 | 1.02 | 5 |
| 100 | 1.04 | 10 |
| 500 | 1.05 | 15 |
| 1000 | 1.05 | 20 |

For dimensions not in the table, the script defaults to `inflation=1.05` and `neighbor_size=max(1, n_dim//10)`.

---

## Experimental Setup

| Item | Choice |
|---|---|
| True state dynamics | Deterministic Lorenz-96 (Euler, dt=0.005) |
| Spin-up | 500 deterministic steps before filtering |
| Observation operator | g(x) = arctan(x) |
| Observation noise | N(0, 0.01 I) |
| Ensemble forecast | Stochastic — process noise sigma_dyn=0.1 |
| Ensemble initialization | Uninformed N(0, I) for all methods |
| Seeds | 10 independent seeds (0 through 9) |
| Reported RMSE | Mean over last 50 steps (steady state) |
| Confidence bands | ±1 std across seeds |

---

## Citation

If you use this code, please cite:

```bibtex
@article{pham2026survey,
  title={A Survey of Nonlinear Filtering Methods for Data Assimilation and Recent Advances in Generative AI-Enabled Score Filters},
  author={Pham, Cath,
  journal={arXiv preprint},
  year={2026}
}
```

---

## References

- Gordon, N.J., Salmond, D.J., Smith, A.F.M. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F*, 140(2), 107–113.
- Evensen, G. (1994). Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics. *Journal of Geophysical Research*, 99(C5), 10143–10162.
- Hunt, B.R., Kostelich, E.J., Szunyogh, I. (2007). Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter. *Physica D*, 230(1-2), 112–126.
- Bao, F., et al. (2024). An Ensemble Score Filter for Tracking High-Dimensional Nonlinear Dynamical Systems. *arXiv preprint*.
