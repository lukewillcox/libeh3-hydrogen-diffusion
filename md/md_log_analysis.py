import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

temperatures = [100, 200, 300, 340, 400]

BASE_DIR = Path.home() / "Library/CloudStorage/OneDrive-CardiffUniversity"

log_files = {
    100: BASE_DIR / "XTB-100K" / "md_100K.log",
    200: BASE_DIR / "XTB-200K" / "md_200K.log",
    300: BASE_DIR / "XTB-300K" / "md_300K.log",
    340: BASE_DIR / "XTB-340K" / "md_340K.log",
    400: BASE_DIR / "XTB-400K" / "md_400K.log",
}

def read_log_file(filename):
    """
    Read ASE MD log file and extract time,
    total energy, potential energy, kinetic energy, temperature
    Handles multiple restart segments by taking
    only the longest continuous segment
    """
    segments = []
    current_segment = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if 'Time[ps]' in line:
                if current_segment:
                    segments.append(current_segment)
                current_segment = []
                continue

            try:
                parts = line.split()
                if len(parts) == 5:
                    t, etot, epot, ekin, temp = map(float, parts)
                    current_segment.append([t, etot, epot, ekin, temp])
            except (ValueError, IndexError):
                continue

    if current_segment:
        segments.append(current_segment)

    if not segments:
        return None

    longest = max(segments, key=len)
    return np.array(longest)

results = {}

print("=" * 60)
print("LOG FILE ANALYSIS")
print("=" * 60)

for T in temperatures:
    filename = log_files[T]

    try:
        data = read_log_file(filename)
        if data is None:
            print(f"T={T}K: Could not parse log file")
            continue

        time = data[:, 0]
        etot = data[:, 1]
        epot = data[:, 2]
        ekin = data[:, 3]
        temp = data[:, 4]

        results[T] = {
            'time': time,
            'etot': etot,
            'epot': epot,
            'ekin': ekin,
            'temp': temp,
            'mean_T': np.mean(temp),
            'std_T': np.std(temp),
            'mean_Ekin': np.mean(ekin),
            'total_time': time[-1]
        }

        print(f"\nT = {T} K")
        print(f"  Data points:      {len(time)}")
        print(f"  Total time:       {time[-1]:.3f} ps")
        print(f"  Mean temperature: {np.mean(temp):.1f} K")
        print(f"  Std temperature:  {np.std(temp):.1f} K")
        print(f"  Mean Ekin:        {np.mean(ekin):.4f} eV")

    except FileNotFoundError:
        print(f"T={T}K: File {filename} not found")

# ============================================================
# PLOTTING
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

for i, T in enumerate(temperatures):
    if T in results:
        r = results[T]
        window = min(50, max(1, len(r['temp']) // 10))
        running_avg = np.convolve(
            r['temp'],
            np.ones(window) / window,
            mode='valid'
        )
        time_avg = r['time'][:len(running_avg)]
        axes[0, 0].plot(
            time_avg, running_avg,
            label=f'{T} K',
            color=colors[i],
            linewidth=1.5
        )

axes[0, 0].set_xlabel('Time (ps)', fontsize=12)
axes[0, 0].set_ylabel('Temperature (K)', fontsize=12)
axes[0, 0].set_title('Temperature Stability\n(running average)', fontsize=12)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

T_vals = [T for T in temperatures if T in results]
Ekin_vals = [results[T]['mean_Ekin'] for T in T_vals]
mean_T_vals = [results[T]['mean_T'] for T in T_vals]

axes[0, 1].plot(
    T_vals, Ekin_vals, 'o-',
    color='steelblue', linewidth=2,
    markersize=8, markerfacecolor='white',
    markeredgewidth=2
)
axes[0, 1].set_xlabel('Target Temperature (K)', fontsize=12)
axes[0, 1].set_ylabel('Mean Kinetic Energy (eV)', fontsize=12)
axes[0, 1].set_title('Mean Kinetic Energy vs Temperature\n(confirms thermostat working)', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

if len(T_vals) > 1:
    kB = 8.617e-5  # eV/K
    T_line = np.array([min(T_vals), max(T_vals)])
    axes[0, 1].plot(
        T_line, T_line * kB * 0.5 * 3,
        '--', color='gray', alpha=0.5,
        label='Equipartition (per atom)'
    )
    axes[0, 1].legend(fontsize=9)

for i, T in enumerate(temperatures):
    if T in results:
        r = results[T]
        window = min(50, max(1, len(r['epot']) // 10))
        running_avg = np.convolve(
            r['epot'],
            np.ones(window) / window,
            mode='valid'
        )
        time_avg = r['time'][:len(running_avg)]
        axes[1, 0].plot(
            time_avg, running_avg,
            label=f'{T} K',
            color=colors[i],
            linewidth=1.5
        )

axes[1, 0].set_xlabel('Time (ps)', fontsize=12)
axes[1, 0].set_ylabel('Potential Energy (eV)', fontsize=12)
axes[1, 0].set_title('Potential Energy vs Time\n(running average)', fontsize=12)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(
    T_vals, mean_T_vals,
    color='steelblue', s=100, zorder=5
)
axes[1, 1].plot(
    [min(T_vals), max(T_vals)],
    [min(T_vals), max(T_vals)],
    '--', color='gray', alpha=0.7,
    label='Perfect thermostat'
)
for T, mT in zip(T_vals, mean_T_vals):
    axes[1, 1].annotate(
        f'{mT:.0f} K',
        (T, mT),
        textcoords="offset points",
        xytext=(5, 5), fontsize=9
    )
axes[1, 1].set_xlabel('Target Temperature (K)', fontsize=12)
axes[1, 1].set_ylabel('Mean Achieved Temperature (K)', fontsize=12)
axes[1, 1].set_title('Thermostat Performance', fontsize=12)
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Molecular Dynamics Simulation Analysis — LiBeH₃',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('md_log_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure saved as md_log_analysis.png")

print("\n" + "=" * 65)
print("THERMOSTAT SUMMARY")
print("=" * 65)
print(f"{'Target T (K)':<15} {'Mean T (K)':<15} {'Std T (K)':<15} "
      f"{'Total time (ps)':<15}")
print("-" * 65)
for T in temperatures:
    if T in results:
        r = results[T]
        print(f"{T:<15} {r['mean_T']:<15.1f} "
              f"{r['std_T']:<15.1f} {r['total_time']:<15.1f}")
