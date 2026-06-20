import torch
import numpy as np
import time

# ==============================================================================
# SETTINGS
# ==============================================================================
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
device = torch.device("cpu")

F             = 8
n_dim         = 1000        # change to 50, 100, 500, 1000
dyn_sigma     = 0.1
dt            = 0.005
filtering_steps = 500
obs_sigma     = 0.1
obs_gap       = 1
ensemble_size = 100
n_seeds       = 10

print(f"EnKF | d={n_dim}, N={ensemble_size}, seeds={n_seeds}")

# ==============================================================================
# LORENZ-96
# ==============================================================================
def lorenz96_drift(x):
    return (torch.roll(x, -1, dims=-1) - torch.roll(x, 2, dims=-1)) \
           * torch.roll(x, 1, dims=-1) - x + F

def drift_step(x, dt):
    return x + dt * lorenz96_drift(x)

def forecast_step(x, dt):
    return drift_step(x, dt) + np.sqrt(dt) * dyn_sigma * torch.randn_like(x)

def obs_fn(x):
    return torch.atan(x)

# ==============================================================================
# ENSEMBLE KALMAN FILTER
# Ref: Evensen (1994) — stochastic EnKF with perturbed observations
# Kalman gain: K = PHT (HPHT + R)^{-1}
# PHT and HPHT estimated from ensemble anomalies (no Jacobian needed)
# linalg.solve used instead of inv for numerical stability
# ==============================================================================
def enkf_update(ensemble, obs):
    N, d = ensemble.shape

    mean   = ensemble.mean(dim=0)
    A      = ensemble - mean           # state anomalies  (N, d)

    H_ens  = obs_fn(ensemble)          # nonlinear H applied to ensemble
    H_mean = H_ens.mean(dim=0)
    HA     = H_ens - H_mean            # obs-space anomalies (N, d)

    # sample cross-covariance PHT and obs-space covariance HPHT
    PHT  = (A.T @ HA)  / (N - 1)      # (d, d)
    HPHT = (HA.T @ HA) / (N - 1)      # (d, d)

    # innovation covariance S = HPHT + R
    S = HPHT + obs_sigma**2 * torch.eye(d, device=device)

    # Kalman gain via solve (numerically stable vs direct inv)
    # K = PHT @ S^{-1}  =>  K = solve(S^T, PHT^T)^T
    K = torch.linalg.solve(S.T, PHT.T).T   # (d, d)

    # perturbed observations — each member gets independent obs noise
    obs_perturbed = obs + obs_sigma * torch.randn(N, d, device=device)
    innovation    = obs_perturbed - H_ens   # (N, d)

    return ensemble + (K @ innovation.T).T

# ==============================================================================
# MULTI-SEED LOOP
# ==============================================================================
all_rmse = []
t_total  = time.time()

for seed in range(n_seeds):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # spin-up: 500 deterministic steps onto attractor
    state_true = torch.randn(n_dim)
    for _ in range(500):
        state_true = drift_step(state_true, dt)

    # uninformed initial ensemble
    x_ensemble = torch.randn(ensemble_size, n_dim, device=device)

    rmse_list = []
    for step in range(filtering_steps):
        # deterministic truth
        state_true = drift_step(state_true, dt)
        # stochastic ensemble
        x_ensemble = forecast_step(x_ensemble, dt)

        if step % obs_gap == 0:
            obs        = obs_fn(state_true) + obs_sigma * torch.randn_like(state_true)
            x_ensemble = enkf_update(x_ensemble, obs)

        x_mean = x_ensemble.mean(dim=0)
        rmse   = torch.sqrt(torch.mean((x_mean - state_true)**2)).item()

        if not np.isfinite(rmse) or rmse > 1000:
            print(f"  Seed {seed} diverged at step {step}")
            rmse_list += [np.nan] * (filtering_steps - len(rmse_list))
            break
        rmse_list.append(rmse)

    all_rmse.append(rmse_list)
    print(f"Seed {seed+1}/{n_seeds} | RMSE(last50)={np.nanmean(rmse_list[-50:]):.4f} "
          f"| {time.time()-t_total:.1f}s")

# ==============================================================================
# SAVE
# ==============================================================================
all_rmse = np.array(all_rmse)
np.save(f"rmse_enkf_d{n_dim}.npy", all_rmse)
print(f"\nSaved rmse_enkf_d{n_dim}.npy  shape={all_rmse.shape}")
print(f"Mean RMSE (last 50): {np.nanmean(all_rmse[:,-50:]):.4f} "
      f"+/- {np.nanstd(np.nanmean(all_rmse[:,-50:],axis=1)):.4f}")
