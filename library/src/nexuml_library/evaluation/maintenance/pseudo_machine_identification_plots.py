"""Optional visual diagnostics for NEX-233 known-machine and pseudo-type routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_identification_report(report: dict[str, Any], output_dir: str | Path) -> None:
    """Plot held-out classifier confusion and state-specific recognition.

    The first seed provides interpretable confusion matrices. Aggregate
    accuracy bars combine repeated split seeds. No new model is fitted.

    Args:
        report: JSON-compatible pseudo-machine-identification report.
        output_dir: Destination for descriptive plots.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    first = report["runs"][0]
    for target_name, target in first["targets"].items():
        labels = target["class_labels"]
        for condition in ("healthy", "faulty"):
            score = target["classifiers"]["healthy_only"][f"assessment_{condition}"]
            confusion = np.asarray(score["confusion_matrix"], dtype=np.int64)
            fig, ax = plt.subplots(figsize=(6, 5))
            image = ax.imshow(confusion, cmap="Blues")
            fig.colorbar(image, ax=ax)
            ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
            ax.set_yticks(range(len(labels)), labels)
            for row in range(len(labels)):
                for column in range(len(labels)):
                    ax.text(
                        column, row, str(confusion[row, column]), ha="center", va="center"
                    )
            ax.set(
                xlabel="Predicted",
                ylabel="True",
                title=f"{target_name}: healthy-trained on {condition} (seed "
                f"{first['seed']})",
            )
            fig.tight_layout()
            fig.savefig(output / f"{target_name}-{condition}-confusion.png", dpi=180)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    regimes = ("healthy_only", "faulty_only", "mixed")
    x = np.arange(len(regimes))
    width = 0.32
    group = report["aggregate"]["machine_identity"]
    for offset, condition in ((-0.5, "healthy"), (0.5, "faulty")):
        series = [
            group[regime][f"assessment_{condition}"]["accuracy"]["mean"]
            for regime in regimes
        ]
        ax.bar(x + offset * width, series, width, label=f"Assessment {condition}")
    ax.set(
        xticks=x,
        xticklabels=["Healthy-trained", "Faulty-trained", "Mixed-trained"],
        ylim=(0.0, 1.05),
        ylabel="Mean identity recognition accuracy",
        title="Known-machine routing across acoustic conditions",
    )
    ax.axhline(
        1.0 / len(report["machine_groups"]), linestyle="--", label="Balanced chance"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "identity-condition-comparison.png", dpi=180)
    plt.close(fig)

    swd = first["healthy_train_swd"]
    names = swd["labels"]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(swd["matrix"], cmap="viridis")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    for row in range(len(names)):
        for column in range(len(names)):
            ax.text(
                column, row, f"{swd['matrix'][row][column]:.2f}", ha="center", va="center"
            )
    ax.set_title("Training-healthy FISHER sliced W1: pseudo-type hypothesis")
    fig.tight_layout()
    fig.savefig(output / "healthy-training-machine-swd.png", dpi=180)
    plt.close(fig)
