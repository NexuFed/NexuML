#!/usr/bin/env bash
set -euo pipefail

git pull
mkdir -p .experiments/rul/sequential

for architecture in transformer conformer cfc; do
  name="maintenance-rul-patchformer-sequence-${architecture}-fd001-s42"
  config=".experiments/rul/sequential/${name}.yaml"

  echo "=== Running ${name} ==="
  nexuml resolve "$name" --output "$config"
  nexuml train --config "$config"
done

echo "=== All three sequence flow experiments completed ==="
