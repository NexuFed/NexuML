#!/usr/bin/env bash
set -euo pipefail

# Run the controlled seven-option FD003 CfC enhancement ablation, all native.
# The upstream seed-42 PatchFormer and CfC endpoint normalizer are shared.
mkdir -p .experiments/rul/enhanced-cfc/fd003

upstream="maintenance-rul-controlled-fd003-patchformer-s42"
upstream_run="rul-FD003-controlled-patchformer-seed42"
patchformer_audit=".experiments/rul/audits/${upstream_run}.checkpoint.json"
if [[ ! -s "$patchformer_audit" ]]; then
  upstream_config=".experiments/rul/enhanced-cfc/fd003/${upstream}.yaml"
  nexuml resolve "$upstream" --output "$upstream_config"
  nexuml train --config "$upstream_config"
fi

# Exact FD003 train-endpoint normalization must exist for all readouts;
# this flow scenario produces it using the upstream FD003 encoder.
flow_upstream="maintenance-rul-patchformer-sequence-cfc-fd003-s42"
flow_run="rul-FD003-patchformer-sequence-cfc-flow-seed42"
endpoints=".experiments/rul/prepared/${flow_run}-endpoints.pt"
if [[ ! -s "$endpoints" ]]; then
  flow_config=".experiments/rul/enhanced-cfc/fd003/${flow_upstream}.yaml"
  nexuml resolve "$flow_upstream" --output "$flow_config"
  nexuml train --config "$flow_config"
fi

read -r -a seeds <<< "${HEAD_SEEDS:-42 43 44}"
read -r -a variants <<< "${VARIANTS:-baseline finetune graph trend multiscale segmentation trajectory uncertainty all all_no_finetune all_no_graph all_no_trend all_no_multiscale all_no_segmentation all_no_trajectory all_no_uncertainty}"

for seed in "${seeds[@]}"; do
  case "$seed" in
    42|43|44) ;;
    *) echo "Invalid enhanced CfC head seed: $seed" >&2; exit 1 ;;
  esac
  for variant in "${variants[@]}"; do
    case "$variant" in
      baseline|finetune|graph|trend|multiscale|segmentation|trajectory|uncertainty|all|all_no_finetune|all_no_graph|all_no_trend|all_no_multiscale|all_no_segmentation|all_no_trajectory|all_no_uncertainty) ;;
      *) echo "Invalid enhanced CfC variant: $variant" >&2; exit 1 ;;
    esac
    scenario="maintenance-rul-enhanced-cfc-fd003-${variant}-s${seed}"
    config=".experiments/rul/enhanced-cfc/fd003/${scenario}.yaml"
    nexuml resolve "$scenario" --output "$config"
    nexuml train --config "$config"
  done
done

# A full factorial is 128 combinations; this intentional 16-config study
# isolates each single addition and its contribution to the combined model.
# Partial grids do not produce a misleading full-comparison summary.
if [[ "${#variants[@]}" == 16 ]]; then
  python -m nexuml_library.evaluation.maintenance.rul_enhanced_summary \
    --subset FD003 --head-seeds "${seeds[@]}"
else
  echo "Partial enhancement grid: summary not generated."
fi
