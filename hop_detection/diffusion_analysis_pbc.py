import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read

def calculate_msd_pbc(trajectory, temperature):
    """
    Calculate MSD with proper PBC unwrapping
    Tracks cumulative displacement accounting for
    periodic boundary crossings
    """
    h_indices = [i for i, atom in enumerate(trajectory[0])
                 if atom.symbol == 'H']

    print(f"  Processing {len(trajectory)} frames, "
          f"{len(h_indices)} H atoms")

    cell = trajectory[0].get_cell()

    prev_positions = trajectory[0].get_positions()[h_indices]
    cumulative_displacement = np.zeros_like(prev_positions)

    msd_values = [0.0]

    for frame in trajectory[1:]:
        curr_positions = frame.get_positions()[h_indices]

        delta = curr_positions - prev_positions

        scaled_delta = np.linalg.solve(cell.T, delta.T).T
        scaled_delta -= np.round(scaled_delta)
        corrected_delta = scaled_delta @ cell

        cumulative_displacement += corrected_delta

        msd = np.mean(np.sum(cumulative_displacement**2, axis=1))
        msd_values.append(msd)

        prev_positions = curr_positions

    return np.array(msd_values)

def calculate_diffusion_coefficient(msd_values, timestep_ps,
                                     fit_fraction=0.5):
    """
    Extract D from linear regime of MSD
    Skips early ballistic regime
    """
    n_frames = len(msd_values)
    start_fit = int(n_frames * (1 - fit_fraction))

    time_array = np.arange(n_frames) * timestep_ps
    fit_time = time_array[start_fit:]
    fit_msd = msd_values[start_fit:]

    coeffs = np.polyfit(fit_time, fit_msd, 1)
    slope = coeffs[0]
    D = slope / 6.0

    fit_line = np.polyval(coeffs, fit_time)
    ss_res = np.sum((fit_msd - fit_line)**2)
    ss_tot = np.sum((fit_msd - np.mean(fit_msd))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return D, time_array, r_squared

# ============================================================
# MAIN ANALYSIS
# ============================================================

TIMESTEP_PS = 0.001  # 1 fs
temperatures = [100, 200, 300, 340, 400]

BASE_DIR = Path.home() / "Library/CloudStorage/OneDrive-CardiffUniversity"

trajectory_files = {
    100: BASE_DIR / "XTB-100K" / "md_100K_RECOVERED.extxyz",
    200: BASE_DIR / "XTB-200K" / "md_200K.traj",
    300: BASE_DIR / "XTB-300K" / "md_300K.traj",
    340: BASE_DIR / "XTB-340K" / "md_340K.traj",
    400: BASE_DIR / "XTB-400K" / "md_400K.traj",
}

results = {}
all_msd = {}
all_time = {}

print("=" * 60)
print("DIFFUSION COEFFICIENT ANALYSIS WITH PBC CORRECTION")
print("=" * 60)

for T in temperatures:
    filename = trajectory_files[T]
    print(f"\nTemperature: {T} K")

    try:
        traj = read(str(filename), index=':')
    except Exception as e:
        print(f"  Could not load {filename}: {e}")
        continue

    msd_values = calculate_msd_pbc(traj, T)

    D, time_array, r_sq = calculate_diffusion_coefficient(
        msd_values, TIMESTEP_PS, fit_fraction=0.5
    )

    results[T] = {'D': D, 'R2': r_sq,
                  'final_msd': msd_values[-1]}
    all_msd[T] = msd_values
    all_time[T] = time_array

    print(f"  Final MSD:             {msd_values[-1]:.4f} Å²")
    print(f"  Diffusion coefficient: {D:.6f} Å²/ps")
    print(f"  D in cm²/s:            {D * 1e-8:.4e} cm²/s")
    print(f"  R² of linear fit:      {r_sq:.4f}")

    if r_sq < 0.9:
        print(f"  WARNING: Poor linear fit (R²={r_sq:.3f})")
        print(f"  Motion may not be truly diffusive at {T} K")

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 65)
print("SUMMARY TABLE")
print("=" * 65)
print(f"{'T (K)':<10} {'D (Å²/ps)':<18} {'D (cm²/s)':<18} "
      f"{'R²':<10} {'Regime':<15}")
print("-" * 65)

for T in temperatures:
    if T in results:
        D = results[T]['D']
        r_sq = results[T]['R2']
        D_cms = D * 1e-8

        if D < 0.0005 or r_sq < 0.7:
            regime = "Localised"
        elif D < 0.005 or r_sq < 0.85:
            regime = "Onset"
        else:
            regime = "Diffusive"

        print(f"{T:<10} {D:<18.6f} {D_cms:<18.4e} "
              f"{r_sq:<10.4f} {regime:<15}")

# ============================================================
# PLOTTING
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

for i, T in enumerate(temperatures):
    if T in all_msd:
        axes[0].plot(
            all_time[T],
            all_msd[T],
            label=f'{T} K',
            color=colors[i],
            linewidth=1.5
        )

axes[0].set_xlabel('Time (ps)', fontsize=12)
axes[0].set_ylabel('MSD (Å²)', fontsize=12)
axes[0].set_title('MSD vs Time (PBC corrected)', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

T_vals = [T for T in temperatures if T in results]
D_vals = [results[T]['D'] for T in T_vals]

axes[1].plot(
    T_vals,
    D_vals,
    'o-',
    color='steelblue',
    linewidth=2,
    markersize=8,
    markerfacecolor='white',
    markeredgewidth=2
)

axes[1].axvline(x=265, color='green', linestyle='--',
                alpha=0.7, label='Experimental onset (265 K)')
axes[1].axvline(x=340, color='red', linestyle='--',
                alpha=0.7, label='MD analysis T (340 K)')
axes[1].set_xlabel('Temperature (K)', fontsize=12)
axes[1].set_ylabel('D (Å²/ps)', fontsize=12)
axes[1].set_title('Diffusion Coefficient vs Temperature', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

T_arr = np.array(T_vals)
D_arr = np.array(D_vals)

mask = D_arr > 0
if np.sum(mask) > 1:
    inv_T = 1000 / T_arr[mask]
    ln_D = np.log(D_arr[mask])

    axes[2].plot(
        inv_T,
        ln_D,
        'o-',
        color='darkorange',
        linewidth=2,
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2
    )

    coeffs = np.polyfit(inv_T, ln_D, 1)
    fit_line = np.polyval(coeffs, inv_T)
    axes[2].plot(
        inv_T,
        fit_line,
        '--',
        color='gray',
        alpha=0.7,
        label='Linear fit'
    )

    kB = 8.617e-5  # eV/K
    Ea = -coeffs[0] * kB * 1000
    print(f"\nArrhenius activation energy: {Ea:.4f} eV")
    print("(Compare to your NEB barrier)")

    axes[2].set_xlabel('1000/T (K⁻¹)', fontsize=12)
    axes[2].set_ylabel('ln(D)', fontsize=12)
    axes[2].set_title(f'Arrhenius Plot\nEa = {Ea:.3f} eV',
                      fontsize=13)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diffusion_analysis_pbc.png', dpi=300,
            bbox_inches='tight')
plt.show()

print("\nFigure saved as diffusion_analysis_pbc.png")
