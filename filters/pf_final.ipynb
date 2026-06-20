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
num_particles = 100
n_seeds       = 10

print(f"PF | d={n_dim}, N={num_particles}, seeds={n_seeds}")

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
# PARTICLE FILTER
# Ref: Gordon, Salmond & Smith (1993) — bootstrap PF
# Likelihood: p(y|x) = N(y; arctan(x), sigma_obs^2 I)
# ==============================================================================
def pf_update(particles, obs, weights):
    pred_obs = obs_fn(particles)
    diff     = obs.unsqueeze(0) - pred_obs
    log_w    = torch.log(weights + 1e-300) \
               - 0.5 * (diff**2).sum(dim=-1) / obs_sigma**2
    log_w   -= log_w.max()
    weights  = torch.exp(log_w)
    return weights / weights.sum()

def pf_resample(particles, weights):
    M   = len(weights)
    ESS = 1.0 / (weights**2).sum().item()
    if ESS < M / 2:
        idx      = torch.multinomial(weights, M, replacement=True)
        particles = particles[idx]
        weights  = torch.full((M,), 1.0 / M)
    return particles, weights

def pf_estimate(particles, weights):
    return (weights.unsqueeze(1) * particles).sum(dim=0)

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

    # uninformed initial ensemble — same as EnKF and EnSF
    particles = torch.randn(num_particles, n_dim)
    weights   = torch.full((num_particles,), 1.0 / num_particles)

    rmse_list = []
    for step in range(filtering_steps):
        # deterministic truth
        state_true = drift_step(state_true, dt)
        # stochastic ensemble
        particles  = forecast_step(particles, dt)

        if step % obs_gap == 0:
            obs     = obs_fn(state_true) + obs_sigma * torch.randn(n_dim)
            weights = pf_update(particles, obs, weights)
            particles, weights = pf_resample(particles, weights)

        estimate = pf_estimate(particles, weights)
        rmse     = ((estimate - state_true)**2).mean().sqrt().item()

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
all_rmse = np.array(all_rmse)   # (n_seeds, filtering_steps)
np.save(f"rmse_pf_d{n_dim}.npy", all_rmse)
print(f"\nSaved rmse_pf_d{n_dim}.npy  shape={all_rmse.shape}")
print(f"Mean RMSE (last 50): {np.nanmean(all_rmse[:,-50:]):.4f} "
      f"+/- {np.nanstd(np.nanmean(all_rmse[:,-50:],axis=1)):.4f}")
