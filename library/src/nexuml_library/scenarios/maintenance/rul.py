"""Supervised RUL benchmark using only conventional NexuML training.

Build typed scenarios, persist resolved YAML, then invoke
`nexuml build CONFIG` and `nexuml train --config CONFIG`.
No custom trainer, Lightning loop or holdout-evaluation runner is needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from nexuml import callback, optimizer
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
from nexuml_library.data.maintenance.rul import (
    AutoCMapSSRULWindows,
    PreparedRULWindows,
    read_rul_artifact,
)
from nexuml_library.evaluation.maintenance.rul_evaluator import RULBenchmarkEvaluator
from nexuml_library.evaluation.maintenance.rul_paper_callbacks import Goel2026PlateauLR
from nexuml_library.evaluation.maintenance.rul_seed import RULSeedOnFitStart
from nexuml_library.layers.model.rul_forecaster import (
    RULRegressionObjective,
    TemporalRULForecaster,
)
from nexuml_library.layers.model.rul_paper_cnn import Goel2026CNNForecaster
from nexuml_library.layers.model.rul_paper_lstm import Goel2026LSTMForecaster
from nexuml_library.layers.model.rul_metrics import RULRegressionMetrics
from nexuml_library.scenarios.training.defaults import (
    default_logging,
    default_training,
)


@scenario("maintenance-rul-benchmark")
def maintenance_rul_benchmark(
    prepared_file: str | Path | None = None,
    *,
    download: bool = True,
    dataset_root: str | Path | None = None,
    cmapss_window: int = 30,
    cmapss_stride: int = 1,
    cmapss_sensor_ids: list[int] | None = None,
    cmapss_normalization: Literal["zscore", "minmax"] = "zscore",
    cmapss_rul_cap: float = 125.0,
    cmapss_profile: str | None = None,
    cmapss_padding_mode: Literal["repeat_first", "zero"] = "repeat_first",
    cmapss_train_full_windows_only: bool = False,
    cmapss_normalize_targets: bool = True,
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
    checkpoint_root: str | Path | None = None,
) -> ScenarioSpec:
    """Build an official NexuML regression scenario on prepared RUL trajectories.

    Test windows never enter optimizer fitting or checkpoint selection.
    The input dataset's already-normalized targets remain normalized for the
    initial benchmark; target_unit and target_scale are saved alongside data.
    The default scenario can be resolved before C-MAPSS data preparation; it
    requires the prepared artifact before actual training.

    Args:
        prepared_file: Immutable artifact from rul_prepare. If omitted, use
            NEXUML_RUL_PREPARED_FILE or automatically retrieve official FD001.
        download: Auto-download missing official C-MAPSS source data.
        dataset_root: Local raw-data cache; defaults to NEXUML_CMAPSS_ROOT.
        cmapss_window: Number of past cycles for automatic data preparation.
        cmapss_stride: Stride for training windows.
        cmapss_sensor_ids: Optional selected sensor IDs, excluding settings.
        cmapss_normalization: Train-only feature normalization method.
        cmapss_rul_cap: Training and currently evaluation label clipping threshold.
        cmapss_profile: Cache and run label distinguishing published protocols.
        cmapss_padding_mode: How to fill incomplete sensor windows.
        cmapss_train_full_windows_only: Exclude incomplete training windows.
        cmapss_normalize_targets: Divide targets by cap for optimization.
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
        checkpoint_root: Optional checkpoint root; defaults to
            NEXUML_RUL_CHECKPOINT_ROOT or .experiments/rul/checkpoints.

    Returns:
        Fully typed scenario suitable for NexuML YAML resolution and training.

    Raises:
        ValueError: If settings, dimensions or split seed are inconsistent.
        FileNotFoundError: If an explicitly selected prepared artifact is missing.
    """
    if min(hidden_dim, depth, batch_size, max_epochs, patience) <= 0 or lr <= 0:
        raise ValueError("RUL training dimensions and learning rate must be positive")
    configured_file = (os.environ.get("NEXUML_RUL_PREPARED_FILE") or "").strip() or None
    if (
        cmapss_window <= 0
        or cmapss_stride <= 0
        or cmapss_rul_cap <= 0
        or (
            cmapss_profile is not None
            and (
                not cmapss_profile
                or any(
                    char not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                    for char in cmapss_profile
                )
            )
        )
    ):
        raise ValueError("Invalid paper-profile C-MAPSS preparation settings")
    default_seed = 42 if seed is None else seed
    default_artifact = (
        f".experiments/rul/prepared/cmapss-FD001-seed{default_seed}-{cmapss_profile}.pt"
        if cmapss_profile is not None
        else f".experiments/rul/prepared/cmapss-FD001-seed{default_seed}.pt"
    )
    selected_file = (
        prepared_file
        if prepared_file is not None
        else configured_file
        if configured_file is not None
        else default_artifact
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
            "seed": default_seed,
            "target_scale": float(
                cmapss_rul_cap if cmapss_normalize_targets else 1.0
            ),
        }
        length = cmapss_window
        channels = 24 if cmapss_sensor_ids is None else len(cmapss_sensor_ids)
    else:
        raise FileNotFoundError(
            f"Prepared RUL artifact is not a file: {file_path}. "
            "To enable automatic NASA download, omit prepared_file and "
            "unset NEXUML_RUL_PREPARED_FILE."
        )
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
    selected_checkpoint_root = (
        checkpoint_root
        if checkpoint_root is not None
        else os.environ.get("NEXUML_RUL_CHECKPOINT_ROOT", ".experiments/rul/checkpoints")
    )
    if cmapss_profile is not None and prepared_file is None and configured_file is None:
        run_name = f"{run_name}-{cmapss_profile}"
    checkpoint_dir = Path(selected_checkpoint_root).expanduser().resolve() / run_name
    source_definition: AutoCMapSSRULWindows | PreparedRULWindows
    if prepared_file is None and configured_file is None:
        configured_root = (os.environ.get("NEXUML_CMAPSS_ROOT") or "").strip()
        selected_root = (
            dataset_root
            if dataset_root is not None
            else configured_root or "data/cmapss"
        )
        source_definition = AutoCMapSSRULWindows(
            root=str(selected_root),
            artifact_path=str(file_path),
            seed=resolved_seed,
            window=cmapss_window,
            stride=cmapss_stride,
            rul_cap=cmapss_rul_cap,
            sensor_ids=cmapss_sensor_ids,
            normalization=cmapss_normalization,
            padding_mode=cmapss_padding_mode,
            train_full_windows_only=cmapss_train_full_windows_only,
            normalize_targets=cmapss_normalize_targets,
            download=download,
        )
    else:
        source_definition = PreparedRULWindows(path=str(file_path))
    data = DataSpec(
        datasets=[
            DatasetSpec(
                source=source_definition,
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

# A single ordinary ScenarioSpec can train several independent model heads
# on the same prepared data, but NexuML checkpoints and early stopping act
# on the shared run. This is not six independently tuned training sessions.
_RULArchitecture = Literal["feature_mlp", "lstm", "tcn", "transformer", "tsmixer"]
_RUL_COMPOSITE_ARCHITECTURES: tuple[_RULArchitecture, ...] = (
    "feature_mlp",
    "lstm",
    "tcn",
    "transformer",
    "tsmixer",
)


@scenario("maintenance-rul-all")
def maintenance_rul_all(
    prepared_file: str | Path | None = None,
    *,
    download: bool = True,
    dataset_root: str | Path | None = None,
    hidden_dim: int = 96,
    depth: int = 3,
    dropout: float = 0.1,
    lr: float = 3e-4,
    batch_size: int = 64,
    max_epochs: int = 150,
    patience: int = 20,
    seed: int | None = None,
    checkpoint_root: str | Path | None = None,
) -> ScenarioSpec:
    """Train six RUL heads together using one native NexuML scenario.

    The point-regression models and quantile TCN receive the same sensor
    windows but have disjoint parameters, predictions, losses and metrics.
    They share one optimizer, aggregate validation loss and checkpoint, so
    this scenario is not a substitute for independently selected baselines.

    Args:
        prepared_file: Immutable RUL artifact or the standard environment override.
        download: Automatically retrieve missing official C-MAPSS data.
        dataset_root: Local NASA raw-data cache for first-time preparation.
        hidden_dim: Width of each model backbone.
        depth: Depth of each model backbone.
        dropout: Regularization applied to each backbone.
        lr: Shared optimizer learning rate.
        batch_size: Shared minibatch size.
        max_epochs: Maximum number of training epochs.
        patience: Early stopping patience for the aggregate loss.
        seed: Optional split seed; inherited from the prepared artifact by default.
        checkpoint_root: Optional NexuML checkpoint root.

    Returns:
        Composite ScenarioSpec for native nexuml resolve and train.

    Raises:
        ValueError: If the source artifact or requested configuration is invalid.
        FileNotFoundError: If an explicitly selected prepared artifact does not exist.
    """
    baseline = maintenance_rul_benchmark(
        prepared_file=prepared_file,
        download=download,
        dataset_root=dataset_root,
        architecture="tcn",
        objective="huber",
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        checkpoint_root=checkpoint_root,
    )
    baseline_model = baseline.pipeline.stages["RULModel"][0].component
    assert isinstance(baseline_model, TemporalRULForecaster)
    resolved_seed = baseline_model.initialization_seed
    baseline_evaluator = baseline.evaluation.algorithms[0].algorithm
    assert isinstance(baseline_evaluator, RULBenchmarkEvaluator)
    target_scale = baseline_evaluator.target_scale

    models: list[LayerSpec] = []
    objectives_and_metrics: list[LayerSpec] = []
    evaluators: list[EvalAlgorithmSpec] = []
    loss_keys: dict[str, float] = {}
    metric_keys: list[str] = []
    branches: list[tuple[str, _RULArchitecture, bool]] = [
        (name, name, False) for name in _RUL_COMPOSITE_ARCHITECTURES
    ]
    branches.append(("quantile_tcn", "tcn", True))
    for name, architecture, quantile in branches:
        prediction_key = f"predicted_rul_{name}"
        loss_key = f"rul_loss_{name}"
        local_metric_keys = [f"rul_mae_{name}", f"rul_rmse_{name}", f"rul_late_error_{name}"]
        models.append(
            LayerSpec(
                component=TemporalRULForecaster(
                    architecture=architecture,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    dropout=dropout,
                    output_dim=3 if quantile else 1,
                    initialization_seed=resolved_seed,
                ),
                keys_in=["rul_history"],
                keys_out=[prediction_key],
            )
        )
        objectives_and_metrics.extend(
            [
                LayerSpec(
                    component=RULRegressionObjective(objective="quantile" if quantile else "huber"),
                    keys_in=[prediction_key],
                    keys_out=[loss_key],
                    label_key="rul",
                ),
                LayerSpec(
                    component=RULRegressionMetrics(
                        target_scale=target_scale,
                        prediction_index=1 if quantile else 0,
                    ),
                    keys_in=[prediction_key],
                    keys_out=local_metric_keys,
                    label_key="rul",
                ),
            ]
        )
        loss_keys[loss_key] = 1.0 / 6.0
        metric_keys.extend(local_metric_keys)
        evaluators.append(
            EvalAlgorithmSpec(
                name=f"rul_{name}",
                algorithm=RULBenchmarkEvaluator(
                    target_scale=target_scale,
                    point_index=1 if quantile else 0,
                    lower_index=0 if quantile else None,
                    upper_index=2 if quantile else None,
                ),
                feature_key=prediction_key,
                label_key="rul",
            )
        )

    run_name = baseline.name.replace("-tcn-huber-seed", "-all-seed")
    selected_checkpoint_root = (
        checkpoint_root
        if checkpoint_root is not None
        else os.environ.get("NEXUML_RUL_CHECKPOINT_ROOT", ".experiments/rul/checkpoints")
    )
    checkpoint_dir = Path(selected_checkpoint_root).expanduser().resolve() / run_name
    logging = (
        baseline.logging.model_copy(update={"run_name": run_name})
        if baseline.logging is not None
        else None
    )
    return baseline.model_copy(
        update={
            "name": run_name,
            "pipeline": PipelineSpec(
                stages={"RULModels": models, "RULLoss": objectives_and_metrics}
            ),
            "training": baseline.training.model_copy(
                update={"loss_keys": loss_keys, "metric_keys": metric_keys}
            ),
            "evaluation": baseline.evaluation.model_copy(update={"algorithms": evaluators}),
            "logging": logging,
            "callbacks": [
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
                    EarlyStopping,
                    monitor="val/loss",
                    patience=patience,
                    mode="min",
                ),
            ],
        }
    )


# Published data and optimizer settings are not themselves architecture reproductions.
# Both research papers include dedicated model modules absent from TemporalRULForecaster.
_LU2025_SENSORS: list[int] = [
    2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21
]


@scenario("maintenance-rul-lu2025-tcn-protocol")
def maintenance_rul_lu2025_tcn_protocol() -> ScenarioSpec:
    """Match Lu et al. (2025) FD001 preprocessing/training settings with a TCN proxy.

    This is a parameter/protocol control, not Lu's AD-TCN-MSCDIM architecture.
    The original paper's adaptive dilation and cross-dimension modules are not
    part of NexuML's generic TCN. Unreported loss/depth/dropout choices remain
    NexuML proxies and cannot support an exact published-result comparison.

    Returns:
        Native NexuML scenario with paper-parameterized FD001 TCN baseline.
    """
    return maintenance_rul_benchmark(
        architecture="tcn",
        objective="mse",
        cmapss_window=40,
        cmapss_stride=1,
        cmapss_sensor_ids=_LU2025_SENSORS,
        cmapss_normalization="minmax",
        cmapss_rul_cap=125.0,
        cmapss_profile="lu2025-tcn-protocol",
        hidden_dim=32,
        batch_size=256,
        lr=1e-3,
        max_epochs=300,
    )


@scenario("maintenance-rul-cen2024-tsmixer-proxy")
def maintenance_rul_cen2024_tsmixer_proxy() -> ScenarioSpec:
    """Use Cen et al. (2024) FD001 protocol with NexuML's TSMixer proxy.

    The paper uses MFSSCINet, not TSMixer; its feature/sequence attention
    and SCINet interaction modules are NOT reproduced here. This scenario
    only aligns confirmed data, objective, optimizer and batching choices.

    Returns:
        Native NexuML parameter-controlled TSMixer baseline, not MFSSCINet.
    """
    return maintenance_rul_benchmark(
        architecture="tsmixer",
        objective="mse",
        cmapss_window=28,
        cmapss_stride=1,
        cmapss_sensor_ids=_LU2025_SENSORS,
        cmapss_normalization="minmax",
        cmapss_rul_cap=120.0,
        cmapss_profile="cen2024-tsmixer-proxy",
        batch_size=256,
        lr=0.01,
    )


@scenario("maintenance-rul-goel2026-lstm-protocol")
def maintenance_rul_goel2026_lstm_protocol() -> ScenarioSpec:
    """Apply the Goel et al. (2026) FD001 LSTM architecture and training recipe.

    Implements their one-layer 32-unit LSTM, Dropout(0.5), 32-8-8-1
    regressor, RMSprop and validation plateau schedule. The published
    evaluation can be compared against capped and official raw RUL labels;
    unknown publication details still preclude an exact-reproduction claim.

    Returns:
        Native NexuML FD001 scenario with paper-aligned LSTM and optimizer.
    """
    base = maintenance_rul_benchmark(
        architecture="lstm",
        objective="mse",
        cmapss_window=30,
        cmapss_stride=1,
        cmapss_sensor_ids=_LU2025_SENSORS,
        cmapss_normalization="zscore",
        cmapss_rul_cap=130.0,
        cmapss_profile="goel2026-lstm-protocol-raw",
        cmapss_padding_mode="zero",
        cmapss_train_full_windows_only=True,
        cmapss_normalize_targets=False,
        hidden_dim=32,
        depth=1,
        dropout=0.5,
        batch_size=64,
        lr=1e-3,
        max_epochs=200,
        patience=20,
        seed=42,
    )
    paper_layer = LayerSpec(
        component=Goel2026LSTMForecaster(
            hidden_dim=32,
            dropout=0.5,
            initialization_seed=42,
        ),
        keys_in=["rul_history"],
        keys_out=["predicted_rul"],
    )
    paper_evaluator = base.evaluation.algorithms[0].model_copy(
        update={
            "algorithm": RULBenchmarkEvaluator(
                target_scale=1.0,
                include_nasa_score=True,
            ),
        }
    )
    raw_evaluator = EvalAlgorithmSpec(
        name="rul_official_raw",
        algorithm=RULBenchmarkEvaluator(
            target_scale=1.0,
            include_nasa_score=True,
        ),
        feature_key="predicted_rul",
        label_key="rul_raw",
    )
    return base.model_copy(
        update={
            "evaluation": base.evaluation.model_copy(
                update={"algorithms": [paper_evaluator, raw_evaluator]}
            ),
            "pipeline": PipelineSpec(
                stages={
                    "RULModel": [paper_layer],
                    "RULLoss": base.pipeline.stages["RULLoss"],
                }
            ),
            "training": base.training.model_copy(
                update={
                    "optimizer": optimizer(
                        torch.optim.RMSprop, lr=1e-3, weight_decay=1e-5
                    ),
                }
            ),
            "callbacks": [
                *base.callbacks,
                callback(Goel2026PlateauLR, factor=0.5, patience=5),
            ],
        }
    )


@scenario("maintenance-rul-goel2026-cnn-protocol")
def maintenance_rul_goel2026_cnn_protocol() -> ScenarioSpec:
    """Apply Goel et al. (2026) FD001 CNN architecture and training protocol.

    Matches their three 3x1 convolution blocks, dropout, 128-unit regression
    head, and the same native RMSprop and plateau schedule as their LSTM.

    Returns:
        Native NexuML scenario with paper-aligned CNN model and data source.
    """
    baseline = maintenance_rul_goel2026_lstm_protocol()
    run_name = baseline.name.replace("-lstm-", "-cnn-")
    checkpoint_root = (
        (os.environ.get("NEXUML_RUL_CHECKPOINT_ROOT") or "").strip()
        or ".experiments/rul/checkpoints"
    )
    checkpoint_dir = Path(checkpoint_root).expanduser().resolve() / run_name
    cnn_layer = LayerSpec(
        component=Goel2026CNNForecaster(dropout=0.3, initialization_seed=42),
        keys_in=["rul_history"],
        keys_out=["predicted_rul"],
    )
    return baseline.model_copy(
        update={
            "name": run_name,
            "logging": (
                baseline.logging.model_copy(update={"run_name": run_name})
                if baseline.logging is not None
                else None
            ),
            "pipeline": PipelineSpec(
                stages={
                    "RULModel": [cnn_layer],
                    "RULLoss": baseline.pipeline.stages["RULLoss"],
                }
            ),
            "callbacks": [
                callback(RULSeedOnFitStart, seed=42),
                callback(
                    ModelCheckpoint,
                    dirpath=str(checkpoint_dir),
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                ),
                callback(EarlyStopping, monitor="val/loss", patience=20, mode="min"),
                callback(Goel2026PlateauLR, factor=0.5, patience=5),
            ],
        }
    )
