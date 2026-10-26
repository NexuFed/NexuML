#!/usr/bin/env bash
set -euo pipefail

# Only existing NexuML scenarios are resolved and trained.
# Upstream PatchFormer/CfC checkpoints remain frozen at FD001 seed 42.
mkdir -p .experiments/rul/head-architectures

read -r -a seeds <<< "${HEAD_SEEDS:-42 43 44}"
read -r -a heads <<< "${HEADS:-transformer mamba cfc}"
read -r -a arms <<< "${ARMS:-encoder flow}"

for seed in "${seeds[@]}"; do
  case "$seed" in
    42|43|44) ;;
    *) echo "Unsupported RUL readout seed: $seed" >&2; exit 1 ;;
  esac
  for head in "${heads[@]}"; do
    case "$head" in
      transformer|mamba|cfc) ;;
      *) echo "Unsupported RUL readout architecture: $head" >&2; exit 1 ;;
    esac
    for arm in "${arms[@]}"; do
      case "$arm" in
        encoder|flow) ;;
        *) echo "Unsupported RUL input arm: $arm" >&2; exit 1 ;;
      esac
      scenario="maintenance-rul-cfc-head-${head}-${arm}-s${seed}"
      config=".experiments/rul/head-architectures/${scenario}.yaml"
      nexuml resolve "$scenario" --output "$config"
      nexuml train --config "$config"
    done
  done
done

# Comparison requires all three heads and both input arms for each seed.
if [[ " ${heads[*]} " == *" transformer "* &&
      " ${heads[*]} " == *" mamba "* &&
      " ${heads[*]} " == *" cfc "* &&
      " ${arms[*]} " == *" encoder "* &&
      " ${arms[*]} " == *" flow "* ]]; then
  python -m nexuml_library.evaluation.maintenance.rul_head_architecture_summary \
    --head-seeds "${seeds[@]}" \
    --output .experiments/rul/head-architectures/summary.json
else
  echo "Partial architecture grid: validation summary not generated."
fi
