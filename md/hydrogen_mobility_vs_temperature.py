import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read
from ase.geometry import find_mic

temperatures = [100, 200, 300, 340, 400]

# Match these to your jump detection script settings
TIMESTEP_FS = 0.5
SAVE_INTERVAL = 1
LAG_PS = 1.0

BASE_DIR = Path.home() / "Library/CloudStorage/OneDrive-CardiffUniversity"

trajectory_files = {
    100: BASE_DIR / "XTB-100K" / "md_100K_RECOVERED.extxyz",
    200: BASE_DIR / "XTB-200K" / "md_200K.traj",
    300: BASE_DIR / "XTB-300K" / "md_300K.traj",
    340: BASE_DIR / "XTB-340K" / "md_340K.traj",
    400: BASE_DIR / "XTB-400K" / "md_400K.traj",
}

def unwrap_positions(frames, h_indices):
    nframes = len(frames)
    nat = len(h_indices)
    unwrapped = np.zeros((nframes, nat, 3))

    p0 = frames[0].get_positions()
    for k, idx in enumerate(h_indices):
        unwrapped[0, k] = p0[idx]

    cell = frames[0].get_cell()
    pbc = frames[0].get_pbc()

    for t in range(1, nframes):
        pa = frames[t - 1].get_positions()
        pb = frames[t].get_positions()
        for k, idx in enumerate(h_indices):
            dr = pb[idx] - pa[idx]
            dr_mic, _ = find_mic(dr, cell=cell, pbc=pbc)
            unwrapped[t, k] = unwrapped[t - 1, k] + dr_mic

    return unwrapped

results = {}

for T in temperatures:
    filename = trajectory_files[T]
    print(f"Processing {T} K...")

    try:
        frames = read(str(filename), index=':')
    except Exception as e:
        print(f"  Could not load {filename}: {e}")
        continue

    h_indices = [i for i, a in enumerate(frames[0]) if a.symbol == 'H']

    dt_ps = (TIMESTEP_FS * SAVE_INTERVAL) / 1000.0
    lag_frames = int(round(LAG_PS / dt_ps))

    if lag_frames >= len(frames):
        print(f"  Not enough frames for lag time at {T} K")
        continue

    unwrapped = unwrap_positions(frames, h_indices)
    disp = unwrapped[lag_frames:] - unwrapped[:-lag_frames]
    dist = np.linalg.norm(disp, axis=2)

    results[T] = {
        'mean': np.mean(dist),
        'max': np.max(dist),
        'std': np.std(dist),
        'percent_above_2A': np.sum(dist > 2.0) / dist.size * 100,
        'percent_above_15A': np.sum(dist > 1.5) / dist.size * 100
    }

    print(f"  Frames: {len(frames)}")
    print(f"  Mean displacement: {results[T]['mean']:.4f} Å")
    print(f"  Max displacement:  {results[T]['max']:.4f} Å")
    print(f"  % above 2.0 Å:     {results[T]['percent_above_2A']:.3f}%")

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 75)
print("SUMMARY TABLE")
print("=" * 75)
print(f"{'T (K)':<10} {'Mean |Δr| (Å)':<18} {'Max |Δr| (Å)':<18} "
      f"{'% > 2.0 Å':<15} {'% > 1.5 Å':<15}")
print("-" * 75)

for T in temperatures:
    if T in results:
        r = results[T]
        print(f"{T:<10} {r['mean']:<18.4f} {r['max']:<18.4f} "
              f"{r['percent_above_2A']:<15.3f} {r['percent_above_15A']:<15.3f}")

# ============================================================
# PLOTTING
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

T_vals = [T for T in temperatures if T in results]
mean_vals = [results[T]['mean'] for T in T_vals]
max_vals = [results[T]['max'] for T in T_vals]
pct_vals = [results[T]['percent_above_2A'] for T in T_vals]

axes[0].plot(T_vals, mean_vals, 'o-',
             color='steelblue', linewidth=2,
             markersize=8, markerfacecolor='white',
             markeredgewidth=2, label='Mean |Δr|')
axes[0].plot(T_vals, max_vals, 's--',
             color='darkorange', linewidth=2,
             markersize=8, markerfacecolor='white',
             markeredgewidth=2, label='Max |Δr|')
axes[0].axhline(y=2.0, color='red', linestyle=':',
                alpha=0.7, label='Jump threshold (2.0 Å)')
axes[0].axvline(x=265, color='green', linestyle='--',
                alpha=0.7, label='Experimental onset (265 K)')
axes[0].axvline(x=340, color='purple', linestyle='--',
                alpha=0.7, label='MD analysis T (340 K)')
axes[0].set_xlabel('Temperature (K)', fontsize=12)
axes[0].set_ylabel('Hydrogen Displacement (Å)', fontsize=12)
axes[0].set_title('Hydrogen Displacement vs Temperature', fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(T_vals, pct_vals, 'o-',
             color='crimson', linewidth=2,
             markersize=8, markerfacecolor='white',
             markeredgewidth=2)
axes[1].axvline(x=265, color='green', linestyle='--',
                alpha=0.7, label='Experimental onset (265 K)')
axes[1].axvline(x=340, color='purple', linestyle='--',
                alpha=0.7, label='MD analysis T (340 K)')
axes[1].set_xlabel('Temperature (K)', fontsize=12)
axes[1].set_ylabel('% Displacements > 2.0 Å', fontsize=12)
axes[1].set_title('Fraction of Large Displacement Events\nvs Temperature',
                  fontsize=13)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hydrogen_mobility_vs_temperature.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure saved as hydrogen_mobility_vs_temperature.png")
