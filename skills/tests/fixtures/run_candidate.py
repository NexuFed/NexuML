"""Fixed CPU research fixture; copy into an external starter project before use."""

import argparse
import hashlib
import json
import re
import sys
import time
from importlib.metadata import distributions
from pathlib import Path

import torch

from nexuml.core.config import ResolvedConfig
from nexuml.training.lightning import NexuSession

from agent_demo.scenarios import regression


def main():
    """Execute one bounded native attempt and retain evidence, including failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("lr", type=float)
    args = parser.parse_args()
    assert re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id), "Invalid run ID"
    assert 0.001 <= args.lr <= 0.03, "Learning rate outside the contract"
    runs = Path("runs")
    runs.mkdir(exist_ok=True)
    assert len(list(runs.iterdir())) < 3, "Three-attempt total budget exhausted"
    destination = runs / args.run_id
    destination.mkdir()  # existing IDs must not overwrite evidence
    started = time.monotonic()
    evidence = {"id": args.run_id, "lr": args.lr, "state": "running", "command": sys.argv}
    path = destination / "metrics.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    try:
        torch.set_num_threads(1)
        scenario = regression()
        scenario = scenario.model_copy(
            update={
                "name": f"research-{args.run_id}",
                "training": scenario.training.model_copy(update={"lr": args.lr}),
            }
        )
        config = ResolvedConfig.from_scenario(scenario)
        (destination / "scenario.yaml").write_text(config.to_yaml())
        evidence["source_sha256"] = {
            str(file): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in [*sorted(Path("src/agent_demo").glob("*.py")), Path(__file__).resolve()]
        }
        evidence["environment"] = {
            "python": sys.version,
            "distributions": sorted(
                (item.metadata["Name"], item.version) for item in distributions()
            ),
        }
        session = NexuSession(
            scenario,
            log_dir=destination / "logs",
            enable_progress_bar=False,
            enable_loggers=False,
        )
        session.data_module.setup()
        train = list(session.data_module.train_dataloader().dataset.indices)
        validation = list(session.data_module.val_dataloader().dataset.indices)
        assert len(session.data_module.test_dataloader().dataset) == 0
        assert (len(train), len(validation)) == (48, 16)
        assert not set(train) & set(validation)
        assert set(train) | set(validation) == set(range(64))
        evidence["partitions"] = {"train": train, "validation": validation, "test": []}
        # Deliberately use supported phase methods, not run(), which includes test().
        session.fit()
        results = session.validate()
        evidence["validation_loss"] = results[0]["val/loss"]
        evidence["epochs"] = session.trainer.current_epoch
        evidence["global_step"] = session.trainer.global_step
        torch.save(session.pipeline.state_dict(), destination / "model.pt")
        evidence["state"] = "completed"
    except Exception as error:
        evidence.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        evidence["elapsed_seconds"] = time.monotonic() - started
        path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
