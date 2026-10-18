# Nonlinear Filtering Benchmark on Lorenz-96

Companion code for:
> "A Survey of Nonlinear Filtering Methods for Data Assimilation and Recent Advances in Generative AI-Enabled Score Filters"
> Cath Pham, Kadesha Reynolds, Florida State University, 2026

Implements and benchmarks four data assimilation filters on the Lorenz-96 system with nonlinear arctan observations:
- **PF** — Bootstrap Particle Filter (Gordon et al. 1993), with kernel jittering adapted from [DAPPER](https://github.com/nansencenter/DAPPER).
- **EnKF** — Ensemble Kalman Filter (Evensen 1994)
- **EnSF** — Ensemble Score Filter (Bao et al. 2024)
- **LETKF** — Local Ensemble Transform Kalman Filter (Hunt et al. 2007)

---

## Files Organization

1. `filters` contains the implementation of all four data assimilation filters.
   - `pf.py` runs the Bootstrap Particle Filter (Gordon et al. 1993), with kernel jittering adapted from [DAPPER](https://github.com/nansencenter/DAPPER).
   - `enkf.py` runs the Ensemble Kalman Filter (Evensen 1994).
   - `ensf_1.py` and `ensf_2.py` runs the Ensemble Score Filter (Bao et al. 2024).
   - `letkf.py` runs the Local Ensemble Transform Kalman Filter (Hunt et al. 2007).
   - Set `n_dim` at the top of each script, then run PF, EnKF and EnSF for each dimension in `[5, 10, 50]` (Experiment 1), and EnSF and LETKF at `d = 1000` (Experiment 2).

2. `results` contains the final paper figures as `.png` files.
   - `fig1_rmse_vs_dim.png`: RMSE vs state dimension (`d = 5, 10, 50`) with confidence bands.
   - `fig2_rmse_vs_step.png`: RMSE over assimilation cycles at `d = 5, 10, 50`.
   - `fig3_trajectory_spread_d50_seed7.png`: trajectory tracking and ensemble spread at `d = 50`.
   - `fig4_l96_comparison_rmse.py`: EnSF vs LETKF RMSE on observed and unobserved components at `d = 1000`.
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
- Bao, F., Zhang, Z., & Zhang, G. (2024). An ensemble score filter for tracking high-dimensional nonlinear dynamical systems. *Computer Methods in Applied Mechanics and Engineering*, 432, 117447. https://doi.org/10.1016/j.cma.2024.117447
