# partition_function_compare.py
import numpy as np

# Constants
h = 6.626e-34   # J·s
kB = 1.381e-23  # J/K
c = 3e10        # cm/s

TS_frequencies = [
    210.52, 235.78, 265.29, 292.93, 334.53,
    350.67, 381.81, 455.09, 495.01, 531.51,
    545.49, 613.42, 672.22, 815.90, 857.08,
    911.14, 1129.07, 1557.48, 1633.46, 1805.53
]  # imaginary mode excluded

reactant_frequencies = [
    171.85, 213.25, 237.97, 269.10, 277.50,
    280.36, 296.49, 354.13, 391.14, 501.64,
    728.75, 774.90, 830.35, 875.24, 889.17,
    946.47, 998.66, 1391.33, 1451.27, 1535.16,
    1674.74
]

def partition_function(frequencies, T):
    Q = 1.0
    for nu in frequencies:
        freq_hz = nu * c
        x = (h * freq_hz) / (kB * T)
        qi = np.exp(-x / 2) / (1 - np.exp(-x))
        Q *= qi
    return Q

temperatures = [255, 260, 265, 270, 275, 280, 340]

print(f"{'T (K)':<10} {'Q_TS':<20} {'Q_reactant':<20} {'Q_TS/Q_reactant':<20}")
print("=" * 70)

for T in temperatures:
    Q_TS = partition_function(TS_frequencies, T)
    Q_R = partition_function(reactant_frequencies, T)
    ratio = Q_TS / Q_R
    print(f"{T:<10} {Q_TS:<20.6e} {Q_R:<20.6e} {ratio:<20.6e}")