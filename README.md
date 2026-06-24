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

## Requirements

```bash
pip install torch numpy matplotlib
```

Tested with Python 3.10, PyTorch 2.1, NumPy 1.24.

---

## Quickstart

### Step 1 — Run filters

Open each filter script, set `n_dim` at the top, then run:

```bash
python filters/pf_final.py
python filters/enkf_final.py
python filters/ensf_final.py
python filters/letkf_final.py
```

Run each script once per dimension by changing `n_dim` at the top of the file:

```python
n_dim = 10    # change to 50, 100, 500, 1000
```

Each run saves `rmse_{filter}_d{n_dim}.npy` to the working directory.
LETKF and EnSF also save `time_{filter}_d{n_dim}.npy` for the cost comparison.
Move all `.npy` files into `results/` when done.

### Step 2 — Generate figures

Run each figure script from inside the `results/` folder:

```bash
cd results/
python ../figures/fig1_rmse_vs_dim.py
python ../figures/fig2_rmse_vs_step.py
python ../figures/fig3_ensf_vs_letkf.py
python ../figures/fig4_trajectory_spread.py
```

Figures are saved as `.png` files in the current directory.

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

## Expected Runtime (single seed, CPU)

| Filter | d=10 | d=100 | d=500 | d=1000 |
|---|---|---|---|---|
| PF | <1s | ~5s | ~5s | ~5s |
| EnKF | <1s | ~1s | ~6s | ~33s |
| EnSF | ~5s | ~8s | ~24s | ~42s |
| LETKF | <1s | ~2min | ~8min | ~15min |

Full benchmark (4 filters × 5 dimensions × 10 seeds) takes approximately **6–8 hours** on a standard CPU.

---

## Reproducing Paper Figures

All results are fully reproducible from fixed seeds. Each seed loop starts with:

```python
torch.manual_seed(seed)
np.random.seed(seed)
```

ensuring identical results across machines given the same PyTorch version.

---

## Citation

If you use this code, please cite:

```bibtex
@article{pham2026survey,
  title={A Survey of Nonlinear Filtering Methods for Data Assimilation and Recent Advances in Generative AI-Enabled Score Filters},
  author={Pham, Cath,
  journal={arXiv preprint},
  year={2025}
}
```

---

## References

- Gordon, N.J., Salmond, D.J., Smith, A.F.M. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F*, 140(2), 107–113.
- Evensen, G. (1994). Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics. *Journal of Geophysical Research*, 99(C5), 10143–10162.
- Hunt, B.R., Kostelich, E.J., Szunyogh, I. (2007). Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter. *Physica D*, 230(1-2), 112–126.
- Bao, F., et al. (2024). An Ensemble Score Filter for Tracking High-Dimensional Nonlinear Dynamical Systems. *arXiv preprint*.
