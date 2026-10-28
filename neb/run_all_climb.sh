#!/usr/bin/env bash
set -euo pipefail

# --- paths (edit only if yours differ) ---
ROOT="$HOME/Library/CloudStorage/OneDrive-CardiffUniversity"
TRAJ="$ROOT/XTB-340K/md_340K.traj"

EXTRACT="$ROOT/extract_neb_endpoints.py"     # <-- use the extract script you actually have
RELAX="$ROOT/relax_pbc.py"
NEBCLIMB="$ROOT/run_neb_xtb_climb.py"

export XTBPATH="$HOME/miniforge3/envs/xtb-md/share/xtb"
export OMP_NUM_THREADS=1

# format: "frame0 frame1 Hindex"
EVENTS=(
  "1726 2126 32"
  "1725 2125 32"
  "1717 2117 32"
  "1718 2118 32"
  "1772 2172 32"
  "1771 2171 32"
  "1767 2167 24"
  "1721 2121 28"
  "1768 2168 24"
)

for e in "${EVENTS[@]}"; do
  read -r F0 F1 H <<< "$e"
  DIR="neb_${F0}_${F1}_H${H}_CLIMB"

  echo "=============================="
  echo "CLIMB NEB: frames $F0 -> $F1   H$H"
  echo "Folder: $DIR"
  echo "=============================="

  mkdir -p "$DIR"
  cd "$DIR"

  # clean any old partial run in THIS folder only
  rm -f initial.xyz final.xyz initial_relaxed.xyz final_relaxed.xyz
  rm -f neb_stage*.log neb_stage*.traj neb_image_*.xyz

  # 1) extract endpoints from md_340K.traj
  python "$EXTRACT" \
    --traj "$TRAJ" \
    --initial-frame "$F0" \
    --final-frame "$F1"

  # 2) relax endpoints
  python "$RELAX"

  # 3) run CLIMB NEB
  python "$NEBCLIMB"

  cd ..
done

echo "All CLIMB NEBs finished."
