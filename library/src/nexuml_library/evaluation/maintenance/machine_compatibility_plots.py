"""Optional descriptive plots for NEX-233 machine compatibility reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact


def plot_machine_compatibility(
    report: dict[str, Any], prepared_file: str | Path, output_dir: str | Path
) -> None:
    """Plot machine distance matrices and train-fitted healthy PCA projection.

    PCA is fitted on healthy **training** embeddings only, then applied to
    healthy validation embeddings. Test embeddings are never indexed.
    PCA is descriptive and must not define pseudo-type decisions.

    Args:
        report: Result of diagnose_machine_compatibility.
        prepared_file: The same immutable prepared artifact used for analysis.
        output_dir: Directory for generated PNG files.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = list(report["machines"])

    for key, title, file in (
        ("healthy_swd", "Healthy FISHER: inter-machine sliced W1", "healthy_swd.png"),
        ("faulty_swd", "Faulty FISHER: inter-machine sliced W1", "faulty_swd.png"),
        (
            "mean_fault_direction_cosine",
            "Inter-machine mean fault-direction cosine",
            "fault_direction_cosine.png",
        ),
    ):
        matrix = report[key]["matrix"]
        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = "coolwarm" if key == "mean_fault_direction_cosine" else "viridis"
        vmin = -1.0 if key == "mean_fault_direction_cosine" else None
        vmax = 1.0 if key == "mean_fault_direction_cosine" else None
        values = [[float("nan") if v is None else v for v in row] for row in matrix]
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                label = "n/a" if value is None else f"{value:.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=9)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output / file, dpi=180)
        plt.close(fig)

    payload = read_prepared_artifact(prepared_file)
    rows = cast(list[dict[str, Any]], payload["rows"])
    features = cast(Tensor, payload["features"])
    train_indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    if len(train_indices) < 3:
        return
    fit_features = features[train_indices, 0].float().numpy()
    model = PCA(n_components=2, svd_solver="randomized", random_state=report["seed"])
    model.fit(fit_features)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, info in report["machines"].items():
        indices = [
            i for i, row in enumerate(rows)
            if row["group_id"] == name and row["split"] in {"train", "val"}
        ]
        if not indices:
            continue
        coordinates = model.transform(features[indices, 0].float().numpy())
        is_validation = info["split"] == "val"
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=12,
            alpha=0.65,
            label=f"{name} ({info['split']})",
            marker="x" if is_validation else "o",
        )
    ax.set(
        xlabel="Healthy train-fitted PCA component 1",
        ylabel="Healthy train-fitted PCA component 2",
        title="Healthy embeddings: exploratory machine geometry",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "healthy_train_fitted_pca.png", dpi=180)
    plt.close(fig)
