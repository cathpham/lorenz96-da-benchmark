import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# SETTINGS
# ==============================================================================
dims    = [10, 50, 100, 500, 1000]
methods = ["PF", "EnKF", "EnSF"]
colors  = {"EnSF": "#2ca02c", "EnKF": "#ff7f0e", "PF": "#1f77b4"}
markers = {"EnSF": "o",       "EnKF": "s",        "PF": "^"}
labels  = {"EnSF": "EnSF (N=100)", "EnKF": "EnKF (N=100)", "PF": "PF (N=100)"}

# ==============================================================================
# LOAD — shape (n_seeds, 500) or (500,) both handled
# ==============================================================================
rmse_mean = {m: [] for m in methods}
rmse_std  = {m: [] for m in methods}

for m in methods:
    for d in dims:
        data = np.load(f"rmse_{m.lower()}_d{d}.npy")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        per_seed = np.nanmean(data[:, -50:], axis=1)
        rmse_mean[m].append(np.nanmean(per_seed))
        rmse_std[m].append(np.nanstd(per_seed))

rmse_mean = {m: np.array(v) for m, v in rmse_mean.items()}
rmse_std  = {m: np.array(v) for m, v in rmse_std.items()}

# ==============================================================================
# FIGURE 1 — RMSE vs Dimension
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

for m in methods:
    ax.plot(dims, rmse_mean[m],
            color=colors[m], marker=markers[m],
            linewidth=2, markersize=7, label=labels[m])
    ax.fill_between(dims,
                    rmse_mean[m] - rmse_std[m],
                    rmse_mean[m] + rmse_std[m],
                    color=colors[m], alpha=0.2)

ax.set_xscale('log')
ax.set_xticks(dims)
ax.set_xticklabels([str(d) for d in dims])
ax.set_xlabel("State Dimension $d$", fontsize=13)
ax.set_ylabel("RMSE (mean over last 50 steps)", fontsize=13)
ax.set_title("RMSE vs Dimension\n"
             "(mean $\\pm$ 1 std across 10 seeds, $F=8$, $T=500$)",
             fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("fig1_rmse_vs_dim.png", dpi=150)
plt.show()
print("Saved fig1_rmse_vs_dim.png")
