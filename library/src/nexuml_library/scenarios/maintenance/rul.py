"""Supervised RUL benchmark using only conventional NexuML training.

Build typed scenarios, persist resolved YAML, then invoke
`nexuml build CONFIG` and `nexuml train --config CONFIG`.
No custom trainer, Lightning loop or holdout-evaluation runner is needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback
from nexuml.core.discovery import scenario
from nexuml.core.types import (
    DataSpec,
    DatasetSpec,
    EvalAlgorithmSpec,
    EvaluationSpec,
    LayerSpec,
    LoaderSpec,
    PipelineSpec,
    ScenarioSpec,
)
from nexuml_library.data.maintenance.rul import PreparedRULWindows, read_rul_artifact
from nexuml_library.evaluation.maintenance.rul_evaluator import RULBenchmarkEvaluator
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.rul_forecaster import (
    RULRegressionObjective,
    TemporalRULForecaster,
)
from nexuml_library.layers.model.rul_metrics import RULRegressionMetrics
from nexuml_library.scenarios.training.defaults import (
    default_logging,
    default_training,
)


@scenario("maintenance-rul-benchmark")
def maintenance_rul_benchmark(
    prepared_file: str | Path | None = None,
    *,
    architecture: Literal[
        "feature_mlp", "lstm", "tcn", "transformer", "tsmixer"
    ] = "tcn",
    objective: Literal["mse", "mae", "huber", "quantile"] = "huber",
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    hidden_dim: int = 96,
    depth: int = 3,
    dropout: float = 0.1,
    lr: float = 3e-4,
    batch_size: int = 64,
    max_epochs: int = 150,
    patience: int = 20,
    seed: int | None = None,
    checkpoint_root: str | Path = ".experiments/rul/checkpoints",
) -> ScenarioSpec:
    """Build an official NexuML regression scenario on prepared RUL trajectories.

    Test windows never enter optimizer fitting or checkpoint selection.
    The input dataset's already-normalized targets remain normalized for the
    initial benchmark; target_unit and target_scale are saved alongside data.
    The default scenario can be resolved before C-MAPSS data preparation; it
    requires the prepared artifact before actual training.

    Args:
        prepared_file: Immutable artifact from rul_prepare. If omitted, use
            NEXUML_RUL_PREPARED_FILE or the conventional FD001 location.
        architecture: Temporal backbone or feature MLP baseline.
        objective: Point regression or multi-quantile pinball loss.
        quantiles: Ascending pinball probabilities when using quantile objective.
        hidden_dim: Shared backbone dimension.
        depth: Temporal encoder depth.
        dropout: Standard architecture dropout.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        max_epochs: Upper bound on optimization epochs.
        patience: Validation loss early-stopping patience.
        seed: Optional required split seed; defaults to the prepared artifact seed.
        checkpoint_root: User-specified experiment output path.

    Returns:
        Fully typed scenario suitable for NexuML YAML resolution and training.

    Raises:
        ValueError: If settings, dimensions or split seed are inconsistent.
        FileNotFoundError: If an explicitly selected prepared artifact is missing.
    """
    if min(hidden_dim, depth, batch_size, max_epochs, patience) <= 0 or lr <= 0:
        raise ValueError("RUL training dimensions and learning rate must be positive")
    configured_file = os.environ.get("NEXUML_RUL_PREPARED_FILE")
    selected_file = (
        prepared_file
        if prepared_file is not None
        else configured_file
        if configured_file is not None
        else ".experiments/rul/prepared/cmapss-FD001-seed42.pt"
    )
    file_path = Path(selected_file).expanduser().resolve()
    if file_path.is_file():
        payload = read_rul_artifact(file_path)
        settings = payload["settings"]
        _, length, channels = payload["features"].shape
    elif prepared_file is None and configured_file is None:
        # Known FD001 defaults make the registered scenario resolvable without
        # the raw dataset. The normal datasource still requires real data to train.
        settings = {
            "dataset": "cmapss",
            "subset": "FD001",
            "seed": 42,
            "target_scale": 125.0,
        }
        length, channels = 30, 24
    else:
        raise FileNotFoundError(f"Prepared RUL artifact does not exist: {file_path}")
    resolved_seed = int(settings["seed"]) if seed is None else seed
    if int(settings["seed"]) != resolved_seed:
        raise ValueError("RUL scenario seed must equal prepared artifact split seed")
    q = list(quantiles)
    if objective == "quantile" and (
        not q or sorted(set(q)) != q or any(not 0 < p < 1 for p in q)
    ):
        raise ValueError("Quantile levels must be unique ascending probabilities")
    output_dim = len(q) if objective == "quantile" else 1
    run_name = (
        f"rul-{settings['dataset']}-{settings.get('subset', 'all')}-"
        f"{architecture}-{objective}-seed{resolved_seed}"
    )
    checkpoint_dir = Path(checkpoint_root).expanduser().resolve() / run_name
    data = DataSpec(
        datasets=[
            DatasetSpec(
                source=PreparedRULWindows(path=str(file_path)),
                modality="sensor",
                split_type="keep",
            )
        ],
        loader=LoaderSpec(num_workers=0, shuffle_train=True),
        feature_key="rul_history",
        input_shapes={"rul_history": [length, channels]},
    )
    model = TemporalRULForecaster(
        architecture=architecture,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        output_dim=output_dim,
        initialization_seed=resolved_seed,
    )
    pipeline = PipelineSpec(
        stages={
            "RULModel": [
                LayerSpec(
                    component=model,
                    keys_in=["rul_history"],
                    keys_out=["predicted_rul"],
                )
            ],
            "RULLoss": [
                LayerSpec(
                    component=RULRegressionObjective(
                        objective=objective, quantiles=q
                    ),
                    keys_in=["predicted_rul"],
                    keys_out=["rul_loss"],
                    label_key="rul",
                ),
                LayerSpec(
                    component=RULRegressionMetrics(
                        target_scale=float(settings["target_scale"]),
                        prediction_index=(
                            min(range(len(q)), key=lambda i: abs(q[i] - 0.5))
                            if objective == "quantile"
                            else 0
                        ),
                    ),
                    keys_in=["predicted_rul"],
                    keys_out=["rul_mae", "rul_rmse", "rul_late_error"],
                    label_key="rul",
                ),
            ],
        }
    )
    return ScenarioSpec(
        name=run_name,
        data=data,
        pipeline=pipeline,
        training=default_training(
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_keys={"rul_loss": 1.0},
            metric_keys=["rul_mae", "rul_rmse", "rul_late_error"],
        ),
        evaluation=EvaluationSpec(
            metrics=[],
            algorithms=[
                EvalAlgorithmSpec(
                    algorithm=RULBenchmarkEvaluator(
                        target_scale=float(settings["target_scale"]),
                        point_index=(
                            min(range(len(q)), key=lambda i: abs(q[i] - 0.5))
                            if objective == "quantile" else 0
                        ),
                        lower_index=0 if objective == "quantile" else None,
                        upper_index=len(q) - 1 if objective == "quantile" else None,
                    ),
                    feature_key="predicted_rul",
                    label_key="rul",
                )
            ],
            test_result_metrics="all",
        ),
        logging=default_logging(
            experiment_name="NEX-233 Remaining Useful Life",
            run_name=run_name,
            use_mlflow=False,
        ),
        callbacks=[
            callback(RULSeedOnFitStart, seed=resolved_seed),
            callback(
                ModelCheckpoint,
                dirpath=str(checkpoint_dir),
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            ),
            callback(
                EarlyStopping, monitor="val/loss",
                patience=patience, mode="min",
            ),
        ],
    )


@scenario("maintenance-rul-feature-mlp")
def maintenance_rul_feature_mlp() -> ScenarioSpec:
    """Construct the feature-window MLP RUL benchmark as a registered NexuML scenario.

    Returns:
        Typed feature-window MLP regression scenario for native resolve and train.
    """
    return maintenance_rul_benchmark(architecture="feature_mlp")


@scenario("maintenance-rul-lstm")
def maintenance_rul_lstm() -> ScenarioSpec:
    """Construct the LSTM RUL benchmark as a registered NexuML scenario.

    Returns:
        Typed LSTM regression scenario for native resolve and train.
    """
    return maintenance_rul_benchmark(architecture="lstm")


@scenario("maintenance-rul-tcn")
def maintenance_rul_tcn() -> ScenarioSpec:
    """Construct the causal TCN RUL benchmark as a registered NexuML scenario.

    Returns:
        Typed causal TCN regression scenario for native resolve and train.
    """
    return maintenance_rul_benchmark(architecture="tcn")


@scenario("maintenance-rul-transformer")
def maintenance_rul_transformer() -> ScenarioSpec:
    """Construct the causal Transformer RUL benchmark as a registered NexuML scenario.

    Returns:
        Typed causal Transformer regression scenario for native resolve and train.
    """
    return maintenance_rul_benchmark(architecture="transformer")


@scenario("maintenance-rul-tsmixer")
def maintenance_rul_tsmixer() -> ScenarioSpec:
    """Construct the TSMixer RUL benchmark as a registered NexuML scenario.

    Returns:
        Typed TSMixer regression scenario for native resolve and train.
    """
    return maintenance_rul_benchmark(architecture="tsmixer")


@scenario("maintenance-rul-quantile-tcn")
def maintenance_rul_quantile_tcn() -> ScenarioSpec:
    """Construct the quantile TCN RUL benchmark as a registered scenario.

    Returns:
        Typed quantile regression scenario for native resolve and train.
    """
    return maintenance_rul_benchmark(architecture="tcn", objective="quantile")
