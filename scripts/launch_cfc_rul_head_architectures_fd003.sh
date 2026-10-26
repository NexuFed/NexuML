#!/usr/bin/env bash
set -euo pipefail

# Replicate the locked FD001 estimator comparison on FD003 using native NexuML.
# Train the FD003 seed-42 upstream models only when artifacts are missing.
mkdir -p .experiments/rul/head-architectures/fd003

upstream="maintenance-rul-controlled-fd003-patchformer-s42"
upstream_run="rul-FD003-controlled-patchformer-seed42"
upstream_manifest=".experiments/rul/audits/${upstream_run}.checkpoint.json"
flow_upstream="maintenance-rul-patchformer-sequence-cfc-fd003-s42"
flow_run="rul-FD003-patchformer-sequence-cfc-flow-seed42"
flow_endpoints=".experiments/rul/prepared/${flow_run}-endpoints.pt"
flow_checkpoints=".experiments/rul/checkpoints/${flow_run}"

if [[ ! -f "$upstream_manifest" ]]; then
  echo "Training missing FD003 PatchFormer upstream seed-42 checkpoint."
  config=".experiments/rul/head-architectures/fd003/${upstream}.yaml"
  nexuml resolve "$upstream" --output "$config"
  nexuml train --config "$config"
fi

# Do not treat last.ckpt as a validation-selected Flow checkpoint.
if [[ ! -s "$flow_endpoints" ]] ||
    [[ ! -d "$flow_checkpoints" ]] ||
    [[ -z "$(find "$flow_checkpoints" -maxdepth 1 -type f \
      -name '*.ckpt' ! -name 'last.ckpt' -print -quit 2>/dev/null)" ]]; then
  echo "Training missing FD003 CfC sequence-Flow upstream seed-42 checkpoint."
  config=".experiments/rul/head-architectures/fd003/${flow_upstream}.yaml"
  nexuml resolve "$flow_upstream" --output "$config"
  nexuml train --config "$config"
fi

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
      scenario="maintenance-rul-cfc-head-${head}-${arm}-fd003-s${seed}"
      config=".experiments/rul/head-architectures/fd003/${scenario}.yaml"
      nexuml resolve "$scenario" --output "$config"
      nexuml train --config "$config"
    done
  done
done

# Comparison operates only on training/validation curves and provenance.
if [[ " ${heads[*]} " == *" transformer "* &&
      " ${heads[*]} " == *" mamba "* &&
      " ${heads[*]} " == *" cfc "* &&
      " ${arms[*]} " == *" encoder "* &&
      " ${arms[*]} " == *" flow "* ]]; then
  python -m nexuml_library.evaluation.maintenance.rul_head_architecture_summary \
    --subset FD003 --head-seeds "${seeds[@]}" \
    --output .experiments/rul/head-architectures/fd003/summary.json
else
  echo "Partial architecture grid: validation summary not generated."
fi
