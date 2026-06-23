import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# ==============================================================================
# SETTINGS
# ==============================================================================
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
device = torch.device("cpu")

F             = 8
n_dim         = 100        # change to any dimension
dyn_sigma     = 0.1
dt            = 0.005
filtering_steps = 500
obs_sigma     = 0.1
obs_gap       = 1
ensemble_size = 100
euler_steps   = 50
eps_alpha     = 0.05
eps_beta      = 0.001
SEED          = 0

# 4 evenly spaced components to plot
components = [0, n_dim//4, n_dim//2, 3*n_dim//4]

print(f"Trajectory + Spread | d={n_dim}, seed={SEED}, components={components}")

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
# ENKF
# ==============================================================================
def enkf_update(ensemble, obs):
    N, d   = ensemble.shape
    mean   = ensemble.mean(dim=0);  A = ensemble - mean
    H_ens  = obs_fn(ensemble);      H_mean = H_ens.mean(dim=0);  HA = H_ens - H_mean
    PHT    = (A.T @ HA)  / (N - 1)
    HPHT   = (HA.T @ HA) / (N - 1)
    S      = HPHT + obs_sigma**2 * torch.eye(d, device=device)
    K      = torch.linalg.solve(S.T, PHT.T).T
    obs_perturbed = obs + obs_sigma * torch.randn(N, d, device=device)
    return ensemble + (K @ (obs_perturbed - H_ens).T).T

# ==============================================================================
# ENSF
# ==============================================================================
def alpha_bar(t):   return 1.0 - (1.0 - eps_alpha) * t
def beta2_bar(t):   return eps_beta + t * (1.0 - eps_beta)
def drift_coeff(t): return -(1.0 - eps_alpha) / alpha_bar(t)
def diff_coeff(t):  return np.sqrt(1.0 - 2.0 * drift_coeff(t) * beta2_bar(t))

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
        b_t  = drift_coeff(t);  g_t = diff_coeff(t);  g2_t = g_t**2
        score = score_prior(xt, x0, t) + score_likelihood(xt, obs, t)
        xt = xt - dt_s * (b_t * xt - g2_t * score) \
             + np.sqrt(dt_s) * g_t * torch.randn_like(xt)
        t -= dt_s
    return xt

# ==============================================================================
# PARTICLE FILTER
# ==============================================================================
def pf_update(particles, obs, weights):
    diff  = obs.unsqueeze(0) - obs_fn(particles)
    log_w = torch.log(weights + 1e-300) \
            - 0.5 * (diff**2).sum(dim=-1) / obs_sigma**2
    log_w -= log_w.max()
    w     = torch.exp(log_w)
    return w / w.sum()

def pf_resample(particles, weights):
    M = len(weights)
    if 1.0 / (weights**2).sum().item() < M / 2:
        idx      = torch.multinomial(weights, M, replacement=True)
        particles = particles[idx]
        weights  = torch.full((M,), 1.0 / M)
    return particles, weights

def pf_estimate(particles, weights):
    return (weights.unsqueeze(1) * particles).sum(dim=0)

# ==============================================================================
# GENERATE SHARED TRUTH + OBS
# ==============================================================================
torch.manual_seed(SEED)
state_true = torch.randn(n_dim)
for _ in range(500):
    state_true = drift_step(state_true, dt)

true_traj = [state_true.numpy().copy()]
obs_traj  = []
state_tmp = state_true.clone()
for step in range(filtering_steps):
    state_tmp = drift_step(state_tmp, dt)
    true_traj.append(state_tmp.numpy().copy())
    obs_traj.append(
        (obs_fn(state_tmp) + obs_sigma * torch.randn(n_dim)).numpy()
    )

true_traj = np.array(true_traj)   # (501, n_dim)
obs_traj  = np.array(obs_traj)    # (500, n_dim)
np.save(f"true_traj_d{n_dim}.npy", true_traj)
print(f"Saved true_traj_d{n_dim}.npy")

# ==============================================================================
# RUN ALL THREE FILTERS ON SHARED TRUTH
# ==============================================================================
def run_filter(name, init_seed):
    torch.manual_seed(init_seed)

    # all methods: uninformed init
    x_ens   = torch.randn(ensemble_size, n_dim)
    weights = torch.full((ensemble_size,), 1.0 / ensemble_size)

    ens_hist  = []
    rmse_list = []

    for step in range(filtering_steps):
        x_ens = forecast_step(x_ens, dt)

        obs = torch.from_numpy(obs_traj[step])
        if name == "EnKF":
            x_ens = enkf_update(x_ens, obs)
        elif name == "EnSF":
            x_ens = reverse_SDE(x_ens, obs)
        elif name == "PF":
            weights = pf_update(x_ens, obs, weights)
            x_ens, weights = pf_resample(x_ens, weights)

        ens_hist.append(x_ens.detach().numpy().copy())

        x_mean = pf_estimate(x_ens, weights) if name == "PF" \
                 else x_ens.mean(dim=0)
        truth  = torch.from_numpy(true_traj[step + 1])
        rmse   = torch.sqrt(torch.mean((x_mean - truth)**2)).item()
        rmse_list.append(rmse)

    ens_hist = np.array(ens_hist)   # (500, ensemble_size, n_dim)
    np.save(f"ensemble_hist_{name.lower()}_d{n_dim}.npy", ens_hist)
    print(f"{name}: mean RMSE (last 50) = {np.mean(rmse_list[-50:]):.4f}")
    return ens_hist

t0 = time.time()
hist = {}
hist["EnKF"] = run_filter("EnKF", init_seed=1)
hist["EnSF"] = run_filter("EnSF", init_seed=2)
hist["PF"]   = run_filter("PF",   init_seed=3)
print(f"All filters done in {time.time()-t0:.1f}s")

# ==============================================================================
# FIGURE 4 — 4 rows x 3 cols
# Each row: one state component (x_0, x_{d/4}, x_{d/2}, x_{3d/4})
# Each col: one method (EnSF, EnKF, PF)
# Each panel: ensemble spread (5-95th pct) + ensemble mean + true state
# ==============================================================================
methods = ["EnSF", "EnKF", "PF"]
colors  = {"EnSF": "#2ca02c", "EnKF": "#ff7f0e", "PF": "#1f77b4"}
labels  = {"EnSF": "EnSF mean", "EnKF": "EnKF mean", "PF": "PF mean"}
steps   = np.arange(filtering_steps)

fig, axes = plt.subplots(4, 3, figsize=(14, 14), sharey='row', sharex=True)

for row, comp in enumerate(components):
    true_comp = true_traj[1:, comp]   # (500,) — steps 1..500

    for col, m in enumerate(methods):
        ens      = hist[m][:, :, comp]   # (500, ensemble_size)
        ens_mean = ens.mean(axis=1)
        lo       = np.percentile(ens, 5,  axis=1)
        hi       = np.percentile(ens, 95, axis=1)

        ax = axes[row, col]
        ax.fill_between(steps, lo, hi,
                        color=colors[m], alpha=0.25,
                        label="5–95th pct spread")
        ax.plot(steps, ens_mean,
                color=colors[m], lw=1.5, ls='--', label=labels[m])
        ax.plot(steps, true_comp,
                color='black', lw=1.8, label='True state', zorder=3)

        if row == 0:
            ax.set_title(m, fontsize=13)
            ax.legend(fontsize=8, loc='upper right')
        if col == 0:
            ax.set_ylabel(f"$x_{{{comp}}}$", fontsize=11)
        if row == 3:
            ax.set_xlabel("Filtering Step", fontsize=11)
        ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Trajectory Tracking and Ensemble Spread — EnSF, EnKF, PF\n"
    f"($d={n_dim}$, $N={ensemble_size}$, $F=8$, $T={filtering_steps}$)",
    fontsize=13
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"fig4_trajectory_spread_d{n_dim}.png", dpi=150)
plt.show()
print(f"Saved fig4_trajectory_spread_d{n_dim}.png")
