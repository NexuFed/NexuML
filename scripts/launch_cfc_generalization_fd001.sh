#!/usr/bin/env bash
set -euo pipefail

# Default: full 2x2 width/dropout study on matched CfC Flow and encoder heads.
# Optional independent head restarts (same frozen FD001 seed-42 encoder and Flow):
# HEAD_SEEDS="42 43 44" bash scripts/launch_cfc_generalization_fd001.sh

git pull
mkdir -p .experiments/rul/generalization

read -r -a seeds <<< "${HEAD_SEEDS:-42}"
for seed in "${seeds[@]}"; do
  case "$seed" in
    42|43|44) ;;
    *) echo "Unsupported readout seed: $seed" >&2; exit 1 ;;
  esac
  for variant in w128_d010 w128_d030 w064_d010 w064_d030; do
    for arm in encoder flow; do
      name="maintenance-rul-cfc-generalization-${arm}-${variant}-head-s${seed}"
      config=".experiments/rul/generalization/${name}.yaml"
      echo "=== Training ${name} ==="
      nexuml resolve "$name" --output "$config"
      nexuml train --config "$config"
    done
  done
done

python -m nexuml_library.evaluation.maintenance.rul_generalization_summary \
  --head-seeds "${seeds[@]}" \
  --output .experiments/rul/generalization/summary.json

echo "=== CfC generalization study complete ==="
