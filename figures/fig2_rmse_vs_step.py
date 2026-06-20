import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# SETTINGS
# ==============================================================================
dims    = [10, 50, 100, 500]
methods = ["PF", "EnKF", "EnSF"]
colors  = {"EnSF": "#2ca02c", "EnKF": "#ff7f0e", "PF": "#1f77b4"}
labels  = {"EnSF": "EnSF (N=100)", "EnKF": "EnKF (N=100)", "PF": "PF (N=100)"}
steps   = np.arange(1, 501)

# ==============================================================================
# FIGURE 2 — RMSE vs Filtering Step, 2x2 subplots
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
axes = axes.flatten()

for di, (d, ax) in enumerate(zip(dims, axes)):
    for m in methods:
        data = np.load(f"rmse_{m.lower()}_d{d}.npy")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        mean = np.nanmean(data, axis=0)
        std  = np.nanstd(data,  axis=0)

        ax.plot(steps, mean, color=colors[m], lw=1.5, label=labels[m])
        ax.fill_between(steps, mean - std, mean + std,
                        color=colors[m], alpha=0.2)

    ax.set_title(f"$d = {d}$", fontsize=12)
    ax.set_xlabel("Filtering Step", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_ylim(bottom=0, top=7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("RMSE vs Filtering Step — EnSF, EnKF, PF\n"
             "(mean $\\pm$ 1 std across 10 seeds, $N=100$, $F=8$, $T=500$)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig2_rmse_vs_step.png", dpi=150)
plt.show()
print("Saved fig2_rmse_vs_step.png")
