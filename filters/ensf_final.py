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
n_dim         = 10      # change to 10, 50, 100, 500, 1000
dyn_sigma     = 0.1
dt            = 0.005
filtering_steps = 500
obs_sigma     = 0.1
obs_gap       = 1
ensemble_size = 100
n_seeds       = 10

# EnSF hyperparameters (Bao et al. 2024, Eq. 37)
euler_steps = 50      # K = number of reverse SDE steps
eps_alpha   = 0.05    # regularization for alpha schedule

print(f"EnSF | d={n_dim}, N={ensemble_size}, seeds={n_seeds}, K={euler_steps}")

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
# ENSEMBLE SCORE FILTER
# Ref: Bao et al. (2024), "An Ensemble Score Filter for Tracking
#      High-Dimensional Nonlinear Dynamical Systems"
#
# Regularized diffusion schedule (Eq. 37):
#   alpha_bar(tau) = 1 - tau*(1 - eps_alpha)
#   beta_bar^2(tau) = tau
#
# Prior score: batch-size-1 MC estimator (Eq. 38, paper's practical implementation)
#   S_prior(z, x0, tau) = -(z - alpha_bar*x0) / beta_bar^2
#
# Likelihood score with damping h(tau) = 1 - tau (Eq. 33-34):
#   S_like(z, tau) = h(tau) * J_arctan(z)^T * Sigma^{-1} * (y - arctan(z))
#   where J_arctan(z) = diag(1/(1+z^2))
#
# Reverse SDE — Euler-Maruyama (Eq. 31)
# ==============================================================================
def alpha_bar(t):    return 1.0 - (1.0 - eps_alpha) * t
def beta2_bar(t):    return t
def drift_coeff(t):  return -(1.0 - eps_alpha) / alpha_bar(t)
def diff_coeff(t):   return np.sqrt(1.0 - 2.0 * drift_coeff(t) * beta2_bar(t))

def score_prior(xt, x0, t):
    return -(xt - alpha_bar(t) * x0) / beta2_bar(t)

def score_likelihood(xt, obs, t):
    return (1.0 - t) * (-(torch.atan(xt) - obs) / obs_sigma**2
                         * (1.0 / (1.0 + xt**2)))

def reverse_SDE(x0, obs):
    dt_s = 1.0 / euler_steps
    xt   = torch.randn_like(x0)
    t    = 1.0

    for _ in range(euler_steps):
        b_t  = drift_coeff(t)
        g_t  = diff_coeff(t)
        g2_t = g_t**2

        score = score_prior(xt, x0, t) + score_likelihood(xt, obs, t)

        xt = xt - dt_s * (b_t * xt - g2_t * score) \
             + np.sqrt(dt_s) * g_t * torch.randn_like(xt)
        t -= dt_s

    return xt

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
            x_ensemble = reverse_SDE(x_ensemble, obs)

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
np.save(f"rmse_ensf_d{n_dim}.npy", all_rmse)
np.save(f"time_ensf_d{n_dim}.npy", all_time)
print(f"\nSaved rmse_ensf_d{n_dim}.npy  shape={all_rmse.shape}")
print(f"Saved time_ensf_d{n_dim}.npy  shape={all_time.shape}")
print(f"Mean RMSE (last 50): {np.nanmean(all_rmse[:,-50:]):.4f} "
      f"+/- {np.nanstd(np.nanmean(all_rmse[:,-50:],axis=1)):.4f}")
