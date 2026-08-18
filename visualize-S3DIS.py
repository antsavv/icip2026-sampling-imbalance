import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === S3DIS per-class IoU ===
results_KPConv = {
    "equal":  [93.636, 98.515, 80.843, 0.000, 22.259, 44.236, 61.458, 87.107, 79.278, 70.984, 64.165, 60.649, 57.433],
    "invf":   [92.212, 98.305, 80.333, 0.000, 25.148, 46.485, 61.312, 87.132, 79.381, 71.166, 70.813, 60.570, 55.048],
    "cb":     [92.728, 98.405, 80.571, 0.000, 24.341, 44.645, 62.272, 87.440, 78.390, 71.073, 67.183, 62.709, 56.127],
    "invl":   [93.443, 98.486, 80.786, 0.000, 24.058, 45.156, 59.885, 87.778, 79.509, 71.855, 67.775, 60.852, 58.569],
    "invp":   [93.010, 98.415, 80.546, 0.000, 23.339, 46.768, 60.452, 87.024, 78.933, 70.266, 60.969, 61.074, 57.592],
    "comf":   [92.944, 98.410, 81.089, 0.000, 21.425, 46.484, 66.631, 87.458, 79.543, 71.006, 63.212, 61.433, 56.755],
    "FL":     [92.248, 98.405, 79.705, 0.000, 22.489, 45.381, 60.079, 87.715, 79.163, 69.916, 67.915, 61.801, 56.430],
    "LDAM":   [92.706, 98.473, 80.670, 0.000, 24.218, 45.179, 61.765, 87.993, 78.938, 71.449, 63.680, 61.173, 56.468],
    "LADJ":   [92.779, 98.392, 81.497, 0.000, 24.403, 49.590, 62.383, 87.270, 78.890, 71.815, 64.734, 63.405, 56.239],
    "BS":     [93.165, 98.387, 82.492, 0.000, 28.040, 54.397, 63.871, 86.873, 78.086, 71.117, 66.885, 62.996, 55.472],
    "SS":     [92.947, 98.459, 80.608, 0.000, 21.814, 48.496, 62.857, 87.806, 78.637, 70.714, 67.919, 62.051, 55.672]
}

results_RandLA = {
    "equal":  [93.116, 97.028, 80.367, 0.000, 17.390, 57.660, 36.838, 78.216, 84.665, 55.844, 70.799, 71.421, 54.524],
    "invf":   [91.363, 97.390, 78.549, 0.000, 15.893, 60.357, 30.560, 76.231, 85.703, 61.124, 71.363, 64.896, 51.420],
    "cb":     [91.821, 97.074, 79.904, 0.000, 26.317, 61.150, 33.409, 78.371, 81.896, 75.646, 70.521, 64.509, 50.663],
    "invl":   [92.351, 97.355, 80.518, 0.000, 16.287, 59.860, 39.551, 77.654, 86.433, 60.295, 70.575, 68.157, 52.555],
    "invp":   [92.722, 97.734, 80.037, 0.000, 22.035, 59.819, 41.352, 78.766, 86.838, 72.210, 71.187, 72.921, 52.304],
    "comf":   [92.319, 97.902, 80.974, 0.000, 24.176, 59.874, 33.803, 78.060, 87.450, 61.501, 71.115, 73.440, 53.712],
    "FL":     [91.577, 97.346, 81.361, 0.000, 21.368, 58.047, 52.706, 76.267, 86.805, 55.145, 71.403, 68.610, 52.198],
    "LDAM":   [92.158, 96.902, 81.553, 0.000, 28.788, 59.396, 50.455, 77.356, 88.658, 66.383, 72.024, 73.208, 54.194],
    "LADJ":   [92.080, 96.661, 82.223, 0.000, 32.908, 62.755, 47.018, 75.976, 88.004, 69.933, 72.854, 67.595, 52.687],
    "BS":     [91.241, 97.327, 81.789, 0.000, 23.710, 61.296, 43.184, 78.355, 87.483, 67.940, 71.363, 63.855, 52.973],
    "SS":     [91.811, 97.823, 80.882, 0.000, 20.586, 59.820, 45.444, 78.221, 86.209, 56.853, 71.604, 69.960, 53.166]
}

# Select which results to visualize
data = results_KPConv


classes = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "chair", "table", "bookcase", "sofa", "board", "clutter"]

df = pd.DataFrame(data, index=classes)

# Compute delta vs equal
delta_df = df.subtract(df["equal"], axis=0)

fig, ax = plt.subplots(figsize=(24,12))
width = 0.08
x = np.arange(len(classes)) * 1.2  # Increase spacing between class groups

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf", "#bcbd22"]
methods = delta_df.columns.tolist()
methods.remove("equal")

for i, method in enumerate(methods):
    ax.bar(x + i*width, delta_df[method], width=width, label=method)

ax.set_xticks(x + width*len(methods)/2 - width/2)
ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=34)
ax.set_ylabel("Δ IoU vs uniform (%)", fontsize=38)
# Determine network name from selected data
network_name = "KPConv" if data is results_KPConv else "RandLA-Net"
ax.set_title(f"S3DIS  ({network_name}): Class-wise IoU delta relative to uniform", fontsize=38)
ax.axhline(0, color='black', linewidth=0.8)

for i in range(len(classes) - 1):
    ax.axvline(x=x[i] + 0.95, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

ax.legend(loc="upper left", fontsize=30, ncol=2)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.5)
ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3)
ax.minorticks_on()
ax.tick_params(axis='y', labelsize=34)

plt.tight_layout()

plt.savefig(f"S3DIS_{network_name}_deltas.png", dpi=300, bbox_inches='tight', format='png')

# Maximise the window when running interactively; only supported on some
# backends (e.g. TkAgg), so ignore failures on headless/other backends
try:
    plt.get_current_fig_manager().window.state('zoomed')
except Exception:
    pass

plt.show()
