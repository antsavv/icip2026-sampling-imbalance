import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === DALES per-class IoU ===
results_KPConv = {
    "equal":  [96.510, 93.779, 85.062, 42.676, 93.960, 61.052, 72.155, 94.920],
    "invf":   [95.883, 91.102, 71.594, 32.495, 86.191, 31.007, 40.264, 93.622],
    "cb":     [96.456, 93.471, 85.084, 43.479, 94.779, 57.073, 63.340, 94.849],
    "invl":   [96.480, 93.841, 84.812, 43.398, 94.503, 62.600, 75.034, 94.788],
    "invp":   [96.529, 93.799, 85.131, 43.889, 94.611, 63.175, 74.402, 94.972],
    "comf":   [96.488, 93.758, 85.105, 42.484, 94.615, 63.281, 74.237, 94.972],
    "FL":     [96.480, 93.768, 85.078, 43.840, 93.801, 62.423, 70.678, 94.843],
    "LDAM":   [96.520, 93.821, 85.223, 44.560, 94.475, 62.502, 73.150, 94.986],
    "LADJ":   [96.571, 93.665, 84.047, 42.462, 94.336, 59.874, 70.975, 95.084],
    "BS":     [96.126, 91.777, 74.630, 22.739, 93.775, 33.615, 38.903, 93.710],
    "SS":     [96.546, 93.830, 85.016, 43.662, 93.979, 62.957, 71.996, 94.944]
}

results_RandLA = {
    "equal":  [97.148, 93.463, 83.352, 38.220, 91.493, 53.511, 60.266, 96.641],
    "invf":   [96.313, 89.435, 64.182, 32.725, 91.212, 23.269, 46.648, 95.193],
    "cb":     [97.007, 92.955, 80.660, 37.879, 93.530, 53.537, 64.579, 96.401],
    "invl":   [97.034, 93.175, 83.681, 39.589, 89.842, 54.049, 60.650, 96.726],
    "invp":   [97.109, 93.499, 83.569, 39.585, 93.032, 56.368, 68.105, 96.577],
    "comf":   [97.035, 93.232, 83.347, 39.306, 91.040, 54.002, 60.994, 96.728],
    "FL":     [97.155, 93.356, 83.481, 32.884, 90.917, 52.261, 61.089, 96.633],
    "LDAM":   [97.182, 93.603, 84.144, 38.690, 93.199, 57.326, 72.420, 96.809],
    "LADJ":   [97.038, 91.130, 73.658, 18.631, 90.722, 25.906, 40.825, 96.186],
    "BS":     [96.852, 90.782, 74.525, 24.390, 88.973, 24.881, 34.702, 96.333],
    "SS":     [97.192, 92.805, 79.831, 30.681, 91.024, 45.357, 60.493, 96.694]
}

# Select which results to visualize
data = results_RandLA

classes = ["ground", "vegetation", "cars", "trucks", "power lines", "fences", "poles", "buildings"]

df = pd.DataFrame(data, index=classes)

# Compute delta vs equal
delta_df = df.subtract(df["equal"], axis=0)

fig, ax = plt.subplots(figsize=(20,12))
width = 0.1
x = np.arange(len(classes)) * 1.3  # Increase spacing between class groups

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf", "#bcbd22"]
methods = delta_df.columns.tolist()
methods.remove("equal")

for i, method in enumerate(methods):
    ax.bar(x + i*width, delta_df[method], width=width, label=method)

ax.set_xticks(x + width*len(methods)/2 - width/2)
ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=34)
ax.set_ylabel("Δ IoU vs uniform (%)", fontsize=38)
# Determine network name from selected data
network_name = "KPConv" if data is results_KPConv else "RandLA-Net"
ax.set_title(f"DALES ({network_name}): Class-wise IoU delta relative to uniform", fontsize=38)
ax.axhline(0, color='black', linewidth=0.8)

for i in range(len(classes) - 1):
    ax.axvline(x=x[i] + 1.15, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

ax.legend(loc="lower left", fontsize=34, ncol=2)
# ax.set_ylim(-3, 3)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.5)
ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3)
ax.minorticks_on()
ax.tick_params(axis='y', labelsize=34)

plt.tight_layout()

plt.savefig(f"DALES_{network_name}_deltas.png", dpi=300, bbox_inches='tight', format='png')

# Maximise the window when running interactively; only supported on some
# backends (e.g. TkAgg), so ignore failures on headless/other backends
try:
    plt.get_current_fig_manager().window.state('zoomed')
except Exception:
    pass

plt.show()
