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
n_dim         = 500        # change to 50, 100, 500, 1000
dyn_sigma     = 0.1
dt            = 0.005
filtering_steps = 500
obs_sigma     = 0.1
obs_gap       = 1
ensemble_size = 100
n_seeds       = 10

# LETKF hyperparameters — tuned per dimension following Hunt et al. (2007)
# Tuned via grid search on 3 seeds, 200 steps:
#   d=10:   inflation=1.0,  neighbor_size=4
#   d=50:   inflation=1.02, neighbor_size=5
#   d=100:  inflation=1.04, neighbor_size=10
#   d=500:  inflation=1.05, neighbor_size=15  (standard DA practice)
#   d=1000: inflation=1.05, neighbor_size=20  (standard DA practice)
_hyperparam_table = {
    10:   (1.00,  4),
    50:   (1.02,  5),
    100:  (1.04, 10),
    500:  (1.05, 15),
    1000: (1.05, 20),
}
inflation, neighbor_size = _hyperparam_table.get(n_dim, (1.05, max(1, n_dim // 10)))

print(f"LETKF | d={n_dim}, N={ensemble_size}, seeds={n_seeds}")
print(f"inflation={inflation}, neighbor_size={neighbor_size}")

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
# LOCAL ENSEMBLE TRANSFORM KALMAN FILTER
# Ref: Hunt, Kostelich & Szunyogh (2007), Physica D 230, 112-126
#
# For each grid point i:
#   1. Select local observations within neighbor_size radius (periodic)
#   2. Apply Gaussian localization to inflate R_loc
#   3. Solve for analysis weights in ensemble space (N x N, not d x d)
#   4. Apply weight matrix to update local state
#
# linalg.solve used throughout instead of inv for numerical stability
# ==============================================================================
def matrix_sqrt(M):
    # symmetric matrix square root via eigendecomposition
    L, Q = torch.linalg.eigh(M)
    L    = torch.clamp(L, min=0.0)
    return Q * torch.sqrt(L)[None, :]

def letkf_update(ensemble, obs):
    N, d = ensemble.shape

    # obs-space ensemble anomalies
    Y      = obs_fn(ensemble)
    Y_mean = Y.mean(dim=0)
    Y      = Y - Y_mean              # (N, d)

    # state-space anomalies
    X_mean = ensemble.mean(dim=0)
    X_anom = ensemble - X_mean       # (N, d)

    offsets = torch.arange(2 * neighbor_size + 1, device=device) \
              - neighbor_size        # local window offsets

    ensemble_post = []
    for m in range(d):
        # local observation indices (periodic wrap)
        id_y = ((offsets + m) % d).long()

        # Gaussian localization weights on R (Hunt et al. Eq. 10)
        rho_loc = torch.exp(offsets.float()**2 / neighbor_size**2)

        # local quantities
        Y_loc   = Y[:, id_y]                           # (N, p)
        inn_loc = (obs[id_y] - Y_mean[id_y])[None, :]  # (1, p)

        # C = R_loc^{-1} Y_loc^T  (Hunt et al. Eq. 5)
        C = Y_loc / (obs_sigma**2 * rho_loc[None, :])  # (N, p)

        # local analysis covariance in ensemble space (Hunt et al. Eq. 5)
        # P_tilde = [(N-1)/rho * I + Y_loc C^T]^{-1}
        A_mat   = (N - 1) / inflation * torch.eye(N, device=device) \
                  + Y_loc @ C.T
        P_tilde = torch.linalg.solve(A_mat.T, torch.eye(N, device=device)).T

        # weight perturbation matrix W_pert = sqrt[(N-1) P_tilde]
        W = matrix_sqrt((N - 1) * P_tilde)             # (N, N)

        # mean weight vector w_bar = P_tilde C inn_loc^T
        w = inn_loc @ C.T @ P_tilde                     # (1, N)

        # full weight matrix = perturbations + mean shift
        W = W + w[0]                                    # (N, N)

        # update local state
        X_loc = W @ X_anom[:, [m]] + X_mean[m]         # (N, 1)
        ensemble_post.append(X_loc)

    return torch.cat(ensemble_post, dim=1)              # (N, d)

# ==============================================================================
# MULTI-SEED LOOP
# ==============================================================================
all_rmse = []
all_time = []
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
    time_list = []

    for step in range(filtering_steps):
        t_step = time.time()

        # deterministic truth
        state_true = drift_step(state_true, dt)
        # stochastic ensemble
        x_ensemble = forecast_step(x_ensemble, dt)

        if step % obs_gap == 0:
            obs        = obs_fn(state_true) + obs_sigma * torch.randn_like(state_true)
            x_ensemble = letkf_update(x_ensemble, obs)

        x_mean = x_ensemble.mean(dim=0)
        rmse   = torch.sqrt(torch.mean((x_mean - state_true)**2)).item()
        time_list.append(time.time() - t_step)

        if not np.isfinite(rmse) or rmse > 1000:
            print(f"  Seed {seed} diverged at step {step}")
            rmse_list += [np.nan] * (filtering_steps - len(rmse_list))
            break
        rmse_list.append(rmse)

    all_rmse.append(rmse_list)
    all_time.append(time_list)
    print(f"Seed {seed+1}/{n_seeds} | RMSE(last50)={np.nanmean(rmse_list[-50:]):.4f} "
          f"| {time.time()-t_total:.1f}s")

# ==============================================================================
# SAVE
# ==============================================================================
all_rmse = np.array(all_rmse)
all_time = np.array(all_time)
np.save(f"rmse_letkf_d{n_dim}.npy", all_rmse)
np.save(f"time_letkf_d{n_dim}.npy", all_time)
print(f"\nSaved rmse_letkf_d{n_dim}.npy  shape={all_rmse.shape}")
print(f"Mean RMSE (last 50): {np.nanmean(all_rmse[:,-50:]):.4f} "
      f"+/- {np.nanstd(np.nanmean(all_rmse[:,-50:],axis=1)):.4f}")
