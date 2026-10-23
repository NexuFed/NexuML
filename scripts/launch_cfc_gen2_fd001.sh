#!/usr/bin/env bash
set -euo pipefail

# Reference settings: FD001 seed-42 frozen DPAN / CfC, width 64.
# The default experiment grid contains 24 readout trainings, seed 42 only.
# Include extra independent head seeds with HEAD_SEEDS="42 43 44".
# Optionally restrict dimensions via DROPOUTS, OPTIMIZERS, WEIGHTINGS, ARMS.
# All selected cells must be complete for validation-only paired summaries.

git pull
mkdir -p .experiments/rul/gen2

read -r -a seeds <<< "${HEAD_SEEDS:-42}"
read -r -a arms <<< "${ARMS:-encoder flow}"
read -r -a dropouts <<< "${DROPOUTS:-d010 d030}"
read -r -a opts <<< "${OPTIMIZERS:-adam_1e3 adamw_3e4 adamw_1e4}"
read -r -a weights <<< "${WEIGHTINGS:-window engine}"

for seed in "${seeds[@]}"; do
  case "$seed" in
    42|43|44) ;;
    *) echo "Unsupported head seed: $seed" >&2; exit 1 ;;
  esac
  for arm in "${arms[@]}"; do
    for dropout in "${dropouts[@]}"; do
      for opt in "${opts[@]}"; do
        for weight in "${weights[@]}"; do
          name="maintenance-rul-cfc-gen2-${arm}-${dropout}-${opt}-${weight}-head-s${seed}"
          config=".experiments/rul/gen2/${name}.yaml"
          echo "=== Training ${name} ==="
          nexuml resolve "$name" --output "$config"
          nexuml train --config "$config"
        done
      done
    done
  done
done

if [[ "${ARMS:-encoder flow}" == "encoder flow" && \
      "${DROPOUTS:-d010 d030}" == "d010 d030" && \
      "${OPTIMIZERS:-adam_1e3 adamw_3e4 adamw_1e4}" == "adam_1e3 adamw_3e4 adamw_1e4" && \
      "${WEIGHTINGS:-window engine}" == "window engine" ]]; then
  python -m nexuml_library.evaluation.maintenance.rul_gen2_summary \
    --head-seeds "${seeds[@]}" \
    --output .experiments/rul/gen2/summary.json
else
  echo "Partial run completed; full paired validation summary requires all grid cells."
fi

echo "=== CfC optimization and engine-weighted RUL study completed ==="
