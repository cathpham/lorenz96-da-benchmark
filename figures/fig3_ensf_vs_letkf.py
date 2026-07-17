import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# SETTINGS
# ==============================================================================
dims    = [10, 50, 100, 500, 1000]
methods = ["EnSF", "LETKF"]
colors  = {"EnSF": "#2ca02c", "LETKF": "#9467bd"}
markers = {"EnSF": "o",       "LETKF": "s"}
labels  = {"EnSF": "EnSF (N=100)", "LETKF": "LETKF (N=100)"}

# ==============================================================================
# LOAD — shape (n_seeds, 500) or (500,) both handled
# ==============================================================================
rmse_summary  = {m: [] for m in methods}
time_per_step = {m: [] for m in methods}

for d in dims:
    for m in methods:
        rmse_data = np.load(f"rmse_{m.lower()}_d{d}.npy")
        if rmse_data.ndim == 2:
            rmse_data = rmse_data[0]
        rmse_summary[m].append(np.nanmean(rmse_data[-50:]))

        time_data = np.load(f"time_{m.lower()}_d{d}.npy")
        if time_data.ndim == 2:
            time_data = time_data[0]
        time_per_step[m].append(np.nanmean(time_data))

rmse_summary  = {m: np.array(v) for m, v in rmse_summary.items()}
time_per_step = {m: np.array(v) for m, v in time_per_step.items()}

# ==============================================================================
# FIGURE 3 — Accuracy and Cost side by side
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- LEFT: RMSE vs Dimension ---
for m in methods:
    ax1.plot(dims, rmse_summary[m],
             color=colors[m], marker=markers[m],
             lw=2, markersize=7, label=labels[m])

ax1.set_xscale('log')
ax1.set_xticks(dims)
ax1.set_xticklabels([str(d) for d in dims])
ax1.set_xlabel("State Dimension $d$", fontsize=13)
ax1.set_ylabel("RMSE (mean over last 50 steps)", fontsize=13)
ax1.set_title("Accuracy vs Dimension", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# --- RIGHT: Wall-clock time per step vs Dimension ---
for m in methods:
    ax2.plot(dims, time_per_step[m],
             color=colors[m], marker=markers[m],
             lw=2, markersize=7, label=labels[m])

ax2.set_xscale('log')
ax2.set_xticks(dims)
ax2.set_xticklabels([str(d) for d in dims])
ax2.set_xlabel("State Dimension $d$", fontsize=13)
ax2.set_ylabel("Wall-clock time per step (s)", fontsize=13)
ax2.set_title("Computational Cost vs Dimension", fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

fig.suptitle("EnSF vs LETKF — Accuracy and Cost\n"
             "($N=100$, $F=8$, $T=500$)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig3_ensf_vs_letkf.png", dpi=150)
plt.show()
print("Saved fig3_ensf_vs_letkf.png")
