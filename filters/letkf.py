import time

import numpy as np
import torch

# ==============================================================================
# Settings
# ==============================================================================
dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

forcing = 8.0
n_dim = 1000
dt = 0.01
filtering_steps = 1500

obs_sigma = 0.05
dyn_sigma = 0.0
obs_gap = 10

ensemble_size = 100
n_seeds = 10

# observation network
obs_stride = 2
obs_idx_fixed = torch.arange(0, n_dim, obs_stride, dtype=torch.long)
n_obs = obs_idx_fixed.numel()

# initialization and clipping
burn_in_steps = 1000
truth_init_std = 3.0
state_clip = 50.0

# localization and inflation
tol = 1000.0
letkf_table = {
    5: (1.000, 3.15),
    10: (0.915, 6.75),
    50: (0.915, 2.55),
    100: (0.915, 0.75),
    500: (0.915, 2.00),
    1000: (0.915, 1.00),
}
inflation, r_loc = letkf_table.get(n_dim, (1.075, 3.15))
neighbor_size = int(np.floor(np.sqrt(np.log(tol)) * r_loc))

num_da_steps = (filtering_steps - 1) // obs_gap + 1

print(
    f"letkf | d={n_dim}, n={ensemble_size}, seeds={n_seeds}, "
    f"observed={n_obs}/{n_dim}, device={device}, "
    f"inflation={inflation}, r_loc={r_loc}, neighbor_size={neighbor_size}"
)

# ==============================================================================
# Lorenz-96
# ==============================================================================
def lorenz96_drift(x):
    return (torch.roll(x, -1, dims=-1) - torch.roll(x, 2, dims=-1)) * torch.roll(x, 1, dims=-1) - x + forcing


def lorenz96_drift_numpy(x):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing


def rk4_step(x, h):
    k1 = lorenz96_drift(x)
    k2 = lorenz96_drift(x + 0.5 * h * k1)
    k3 = lorenz96_drift(x + 0.5 * h * k2)
    k4 = lorenz96_drift(x + h * k3)
    return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def rk4_step_numpy(x, h):
    k1 = lorenz96_drift_numpy(x)
    k2 = lorenz96_drift_numpy(x + 0.5 * h * k1)
    k3 = lorenz96_drift_numpy(x + 0.5 * h * k2)
    k4 = lorenz96_drift_numpy(x + h * k3)
    return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def truth_step(x):
    return torch.clamp(rk4_step(x, dt), min=-state_clip, max=state_clip)


def forecast_step(x, generator):
    x_next = rk4_step(x, dt)
    if dyn_sigma > 0:
        noise = torch.randn(x_next.shape, generator=generator, device="cpu").to(x_next.device)
        x_next = x_next + noise * (dyn_sigma * dt ** 0.5)
    return torch.clamp(x_next, min=-state_clip, max=state_clip)

# ==============================================================================
# Utilities
# ==============================================================================
def make_generator(seed, generator_device="cpu"):
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


obs_mask = torch.zeros(n_dim, dtype=torch.bool, device=device)
obs_mask[obs_idx_fixed.to(device)] = True


def rmse(ensemble, truth):
    err2 = (ensemble.mean(dim=0) - truth).square()
    rmse_all = torch.sqrt(err2.mean()).item()
    rmse_obs = torch.sqrt(err2[obs_mask].mean()).item()
    rmse_unobs = torch.sqrt(err2[~obs_mask].mean()).item() if (~obs_mask).any() else float("nan")
    return rmse_all, rmse_obs, rmse_unobs


def generate_initial_truths():
    rng = np.random.RandomState(12345)
    truths = []
    for _ in range(n_seeds):
        state = truth_init_std * rng.randn(n_dim)
        for _ in range(burn_in_steps):
            state = rk4_step_numpy(state, dt)
        truths.append(state.copy())
    return np.stack(truths, axis=0)

# ==============================================================================
# Observations
# ==============================================================================
def obs_fn(x):
    return torch.atan(x)


def make_observation_index_schedule():
    return obs_idx_fixed.unsqueeze(0).repeat(num_da_steps, 1)

# ==============================================================================
# LETKF
# ==============================================================================
def letkf_update(ensemble, obs, obs_idx):
    n, d = ensemble.shape

    x_mean = ensemble.mean(dim=0)
    x_anom = ensemble - x_mean

    y_ens = obs_fn(ensemble[:, obs_idx])
    y_mean = y_ens.mean(dim=0)
    y_anom = y_ens - y_mean

    innovation = obs - y_mean
    identity = torch.eye(n, dtype=ensemble.dtype, device=ensemble.device)
    obs_pos = obs_idx.to(ensemble.dtype)

    analysis_cols = []

    for j in range(d):
        direct_distance = torch.abs(j - obs_pos)
        periodic_distance = torch.minimum(direct_distance, d - direct_distance)

        in_range = (periodic_distance <= neighbor_size).to(ensemble.dtype)
        taper = torch.exp(-periodic_distance.square() / r_loc**2) * in_range

        c = y_anom * taper.unsqueeze(0) / obs_sigma**2

        a_mat = (n - 1) / inflation * identity + c @ y_anom.T
        a_mat = 0.5 * (a_mat + a_mat.T)

        p_tilde = torch.linalg.inv(a_mat)
        p_tilde = 0.5 * (p_tilde + p_tilde.T)

        w_mean = p_tilde @ (c @ innovation)

        eigvals, eigvecs = torch.linalg.eigh((n - 1) * p_tilde)
        eigvals = eigvals.clamp_min(0.0)
        w_anom = eigvecs @ torch.diag(torch.sqrt(eigvals))

        w_total = w_anom + w_mean.unsqueeze(0)
        analysis_cols.append(w_total @ x_anom[:, j] + x_mean[j])

    return torch.stack(analysis_cols, dim=1)

# ==============================================================================
# experiment
# ==============================================================================
initial_truths = generate_initial_truths()

all_rmse_steps = np.full((n_seeds, filtering_steps), np.nan)
all_rmse_obs_steps = np.full((n_seeds, filtering_steps), np.nan)
all_rmse_unobs_steps = np.full((n_seeds, filtering_steps), np.nan)
all_rmse_da = np.full((n_seeds, num_da_steps), np.nan)
all_rmse_obs_da = np.full((n_seeds, num_da_steps), np.nan)
all_rmse_unobs_da = np.full((n_seeds, num_da_steps), np.nan)
all_time_da = np.full((n_seeds, num_da_steps), np.nan)
all_obs_indices = np.zeros((n_seeds, num_da_steps, n_obs), dtype=np.int64)

obs_schedule = make_observation_index_schedule()

total_start = time.time()

for seed in range(n_seeds):
    ensemble_generator = make_generator(seed)
    obs_noise_generator = make_generator(10000 + seed)
    ens_noise_generator = make_generator(60000 + seed)

    state_true = torch.as_tensor(initial_truths[seed], dtype=dtype, device=device).clone()

    ensemble = torch.randn(ensemble_size, n_dim, dtype=dtype, generator=ensemble_generator).to(device)
    ensemble = torch.clamp(ensemble, min=-state_clip, max=state_clip)

    obs_noise = torch.randn(num_da_steps, n_dim, dtype=dtype, generator=obs_noise_generator)
    all_obs_indices[seed] = obs_schedule.numpy()

    da_index = 0

    for step in range(filtering_steps):
        state_true = truth_step(state_true)
        ensemble = forecast_step(ensemble, ens_noise_generator)

        if step % obs_gap == 0:
            obs_idx_cpu = obs_schedule[da_index]
            obs_idx = obs_idx_cpu.to(device)
            obs = obs_fn(state_true[obs_idx]) + obs_sigma * obs_noise[da_index, obs_idx_cpu].to(device)

            da_start = time.time()
            ensemble = letkf_update(ensemble, obs, obs_idx)
            all_time_da[seed, da_index] = time.time() - da_start

        error, error_obs, error_unobs = rmse(ensemble, state_true)
        all_rmse_steps[seed, step] = error
        all_rmse_obs_steps[seed, step] = error_obs
        all_rmse_unobs_steps[seed, step] = error_unobs

        if step % obs_gap == 0:
            all_rmse_da[seed, da_index] = error
            all_rmse_obs_da[seed, da_index] = error_obs
            all_rmse_unobs_da[seed, da_index] = error_unobs
            da_index += 1

        if not np.isfinite(error) or error > 1000:
            print(f"seed {seed}: diverged at model step {step}")
            break

    print(
        f"seed {seed + 1}/{n_seeds} | rmse(last50): "
        f"all={np.nanmean(all_rmse_da[seed, -50:]):.4f}, "
        f"obs={np.nanmean(all_rmse_obs_da[seed, -50:]):.4f}, "
        f"unobs={np.nanmean(all_rmse_unobs_da[seed, -50:]):.4f} | "
        f"elapsed={time.time() - total_start:.1f}s"
    )

# ==============================================================================
# Save
# ==============================================================================
prefix = f"letkf_d{n_dim}"

np.save(f"rmse_steps_{prefix}.npy", all_rmse_steps)
np.save(f"rmse_obs_steps_{prefix}.npy", all_rmse_obs_steps)
np.save(f"rmse_unobs_steps_{prefix}.npy", all_rmse_unobs_steps)
np.save(f"rmse_da_{prefix}.npy", all_rmse_da)
np.save(f"rmse_obs_da_{prefix}.npy", all_rmse_obs_da)
np.save(f"rmse_unobs_da_{prefix}.npy", all_rmse_unobs_da)
np.save(f"time_da_{prefix}.npy", all_time_da)
np.save(f"obs_indices_{prefix}.npy", all_obs_indices)


def report(label, values):
    seed_all = np.nanmean(values, axis=1)
    seed_last50 = np.nanmean(values[:, -50:], axis=1)
    print(
        f"{label} | all da: {np.nanmean(seed_all):.4f} +/- {np.nanstd(seed_all):.4f} | "
        f"last 50 da: {np.nanmean(seed_last50):.4f} +/- {np.nanstd(seed_last50):.4f}"
    )


print()
report("rmse", all_rmse_da)
report("rmse_obs", all_rmse_obs_da)
report("rmse_unobs", all_rmse_unobs_da)
print(f"total elapsed: {time.time() - total_start:.1f}s")
