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
n_dim = 10
dt = 0.01
filtering_steps = 1500

obs_sigma = 0.1
dyn_sigma = 0.2
obs_gap = 10

num_particles = 10000
n_seeds = 10

# observation network
obs_coverage = 1.0
n_obs = max(1, min(n_dim, round(obs_coverage * n_dim)))
random_obs_each_da = False

# initialization and clipping
burn_in_steps = 1000
truth_init_std = 3.0
ensemble_init_std = 0.1
state_clip = 50.0

# regularization
pf_reg = 1.5
pf_nuj = True

num_da_steps = (filtering_steps - 1) // obs_gap + 1

print(
    f"pf | d={n_dim}, n={num_particles}, seeds={n_seeds}, "
    f"observed={n_obs}/{n_dim}, device={device}, reg={pf_reg}, nuj={pf_nuj}"
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


def euler_step(x, h):
    return x + h * lorenz96_drift(x)


def truth_step(x):
    return torch.clamp(rk4_step(x, dt), min=-state_clip, max=state_clip)


def forecast_step(x, generator):
    x_next = euler_step(x, dt)
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


def rmse(particles, weights, truth):
    estimate = (weights.unsqueeze(1) * particles).sum(dim=0)
    return torch.sqrt((estimate - truth).square().mean()).item()


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


def make_observation_index_schedule(generator):
    if n_obs == n_dim:
        return torch.arange(n_dim, dtype=torch.long).unsqueeze(0).repeat(num_da_steps, 1)
    if random_obs_each_da:
        schedule = [torch.sort(torch.randperm(n_dim, generator=generator)[:n_obs]).values for _ in range(num_da_steps)]
        return torch.stack(schedule, dim=0)
    fixed = torch.sort(torch.randperm(n_dim, generator=generator)[:n_obs]).values
    return fixed.unsqueeze(0).repeat(num_da_steps, 1)

# ==============================================================================
# PF
# ==============================================================================
def pf_update(particles, obs, weights, obs_idx):
    diff = obs.unsqueeze(0) - obs_fn(particles[:, obs_idx])
    log_w = torch.log(weights + 1e-300) - 0.5 * (diff ** 2).sum(dim=-1) / obs_sigma ** 2
    log_w -= log_w.max()
    weights = torch.exp(log_w)
    return weights / weights.sum()


def compute_ess(weights):
    return 1.0 / (weights ** 2).sum().item()


def weighted_cov_factor(particles, weights):
    m = particles.shape[0]
    x = particles.double()
    w = weights.double()

    if (1.0 - w.max()) < 1e-6:
        w = torch.full_like(w, 1.0 / m)

    mu = (w[:, None] * x).sum(dim=0)
    anom = x - mu
    unbias = 1.0 / (1.0 - (w ** 2).sum())
    cov = unbias * (anom * w[:, None]).T @ anom

    evals, evecs = torch.linalg.eigh(cov)
    factor = evecs * evals.clamp_min(0.0).sqrt().unsqueeze(0)
    return factor.to(particles.dtype)


def regularize(particles_pre, weights_pre, idx, generator):
    m, d = particles_pre.shape
    new = particles_pre[idx]

    if pf_reg <= 0:
        return new

    h = pf_reg * m ** (-1.0 / (d + 4))
    factor = weighted_cov_factor(particles_pre, weights_pre)

    if pf_nuj:
        mask = (idx == torch.roll(idx, 1)) | (idx == torch.roll(idx, -1))
    else:
        mask = torch.ones(m, dtype=torch.bool, device=new.device)

    n_jit = int(mask.sum().item())
    if n_jit > 0:
        z = torch.randn(n_jit, d, dtype=new.dtype, generator=generator)
        new[mask] += h * (z.to(new.device) @ factor.T)

    return torch.clamp(new, min=-state_clip, max=state_clip)


def pf_resample(particles, weights, ess, resample_generator, jitter_generator):
    m = len(weights)
    if ess < m / 2:
        u0 = torch.rand(1, generator=resample_generator).item()
        positions = (u0 + torch.arange(m, dtype=weights.dtype, device=weights.device)) / m
        cumsum = torch.cumsum(weights, dim=0)
        cumsum[-1] = 1.0
        idx = torch.searchsorted(cumsum, positions)

        particles = regularize(particles, weights, idx, jitter_generator)
        weights = torch.full((m,), 1.0 / m, device=particles.device, dtype=particles.dtype)
    return particles, weights

# ==============================================================================
# Experiment
# ==============================================================================
initial_truths = generate_initial_truths()

all_rmse_steps = np.full((n_seeds, filtering_steps), np.nan)
all_rmse_da = np.full((n_seeds, num_da_steps), np.nan)
all_time_da = np.full((n_seeds, num_da_steps), np.nan)
all_ess_da = np.full((n_seeds, num_da_steps), np.nan)
all_obs_indices = np.zeros((n_seeds, num_da_steps, n_obs), dtype=np.int64)

total_start = time.time()

for seed in range(n_seeds):
    ensemble_generator = make_generator(seed)
    obs_noise_generator = make_generator(10000 + seed)
    obs_index_generator = make_generator(30000 + seed)
    resample_generator = make_generator(40000 + seed)
    ens_noise_generator = make_generator(60000 + seed)
    jitter_generator = make_generator(70000 + seed)

    state_true = torch.as_tensor(initial_truths[seed], dtype=dtype, device=device).clone()

    init_noise = torch.randn(num_particles, n_dim, dtype=dtype, generator=ensemble_generator)
    particles = (state_true.cpu().unsqueeze(0) + init_noise * ensemble_init_std).to(device)
    particles = torch.clamp(particles, min=-state_clip, max=state_clip)
    weights = torch.full((num_particles,), 1.0 / num_particles, device=device, dtype=dtype)

    obs_noise = torch.randn(num_da_steps, n_dim, dtype=dtype, generator=obs_noise_generator)
    obs_schedule = make_observation_index_schedule(obs_index_generator)
    all_obs_indices[seed] = obs_schedule.numpy()

    da_index = 0

    for step in range(filtering_steps):
        state_true = truth_step(state_true)
        particles = forecast_step(particles, ens_noise_generator)

        if step % obs_gap == 0:
            obs_idx_cpu = obs_schedule[da_index]
            obs_idx = obs_idx_cpu.to(device)
            obs = obs_fn(state_true[obs_idx]) + obs_sigma * obs_noise[da_index, obs_idx_cpu].to(device)

            da_start = time.time()
            weights = pf_update(particles, obs, weights, obs_idx)
            ess = compute_ess(weights)
            particles, weights = pf_resample(particles, weights, ess, resample_generator, jitter_generator)
            all_time_da[seed, da_index] = time.time() - da_start
            all_ess_da[seed, da_index] = ess

        error = rmse(particles, weights, state_true)
        all_rmse_steps[seed, step] = error

        if step % obs_gap == 0:
            all_rmse_da[seed, da_index] = error
            da_index += 1

        if not np.isfinite(error) or error > 1000:
            print(f"seed {seed}: diverged at model step {step}")
            break

    print(
        f"seed {seed + 1}/{n_seeds} | rmse(last50)={np.nanmean(all_rmse_da[seed, -50:]):.4f} | "
        f"ess(mean)={np.nanmean(all_ess_da[seed]):.1f} | ess(min)={np.nanmin(all_ess_da[seed]):.1f} | "
        f"elapsed={time.time() - total_start:.1f}s"
    )

# ==============================================================================
# Save
# ==============================================================================
prefix = f"pf_d{n_dim}"

np.save(f"rmse_steps_{prefix}.npy", all_rmse_steps)
np.save(f"rmse_da_{prefix}.npy", all_rmse_da)
np.save(f"time_da_{prefix}.npy", all_time_da)
np.save(f"ess_da_{prefix}.npy", all_ess_da)
np.save(f"obs_indices_{prefix}.npy", all_obs_indices)


def report(label, values, digits=4):
    seed_all = np.nanmean(values, axis=1)
    seed_last50 = np.nanmean(values[:, -50:], axis=1)
    print(
        f"{label} | all da: {np.nanmean(seed_all):.{digits}f} +/- {np.nanstd(seed_all):.{digits}f} | "
        f"last 50 da: {np.nanmean(seed_last50):.{digits}f} +/- {np.nanstd(seed_last50):.{digits}f}"
    )


print()
report("rmse", all_rmse_da)
report("ess", all_ess_da, digits=1)
print(f"total elapsed: {time.time() - total_start:.1f}s")
