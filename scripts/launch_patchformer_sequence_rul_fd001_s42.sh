#!/usr/bin/env bash
set -euo pipefail

git pull
mkdir -p .experiments/rul/sequential

for architecture in transformer conformer cfc; do
  for arm in encoder flow; do
    name="maintenance-rul-patchformer-sequence-${architecture}-${arm}-fd001-s42"
    config=".experiments/rul/sequential/${name}.yaml"

    echo "=== Training ${name} ==="
    nexuml resolve "$name" --output "$config"
    nexuml train --config "$config"
  done
done

echo "=== Summarizing paired full-sequence RUL results ==="
python -m nexuml_library.evaluation.maintenance.rul_sequence_flow_summary \
  --subset FD001 --seed 42 \
  --output .experiments/rul/sequential/rul-comparison-fd001-s42.json

echo "=== All six sequence RUL experiments completed ==="
