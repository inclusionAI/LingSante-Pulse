"""One shared representation, four public cohorts, fixed repeated linear probes."""

from pathlib import Path
import argparse
import subprocess
import sys
import json
import numpy as np
import lingsante_pulse.cgm.evaluation.protocol as protocol

ROOT = Path(__file__).resolve().parents[1] / "assets"
COHORTS = ("cgmacros", "shanghai_t2dm", "stanford", "hall")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--embeddings-root",
        type=Path,
        help="Evaluate existing cohort/embeddings.npz files instead of extracting",
    )
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    inputs = {c: a.data_root / c / (c + "_glucofm_windows_v1.npz") for c in COHORTS}
    if not a.embeddings_root:
        for path in inputs.values():
            if not path.is_file():
                raise FileNotFoundError(path)
    a.output.mkdir(parents=True)
    data = {}
    for cohort, path in inputs.items():
        directory = a.output / cohort
        directory.mkdir()
        embedding = directory / "embeddings.npz"
        if a.embeddings_root:
            embedding = a.embeddings_root / cohort / "embeddings.npz"
        else:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lingsante_pulse.cgm.inference.pipeline",
                    "--input",
                    str(path),
                    "--output",
                    str(embedding),
                    "--cohort",
                    cohort,
                    "--checkpoint-dir",
                    str(a.checkpoint_dir),
                    "--device",
                    a.device,
                ],
                check=True,
            )
        with np.load(embedding, allow_pickle=False) as archive:
            visits = archive["subject_id"].astype(str)
            people = (
                np.array([v.split("_")[0] for v in visits])
                if cohort == "shanghai_t2dm"
                else visits
            )
            data[cohort] = {
                "x": archive["embedding"].astype(np.float64),
                "people": people,
                "labels": {
                    k[6:]: archive[k] for k in archive.files if k.startswith("label_")
                },
            }
    results = {}
    for index, (cohort, d) in enumerate(data.items()):
        tasks = {k: [] for k in d["labels"]}
        for repeat in range(10):
            folds, _ = protocol.common_folds(
                d["people"], d["labels"], 5, 43 + repeat + 100 * index
            )
            for fold, (train, test) in enumerate(folds):
                for task, y in sorted(d["labels"].items()):
                    metrics, _, warnings = protocol.probe(
                        d["x"][train], y[train], d["x"][test], y[test]
                    )
                    tasks[task].append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "metrics": metrics,
                            "convergence_warnings": warnings,
                        }
                    )
        (a.output / cohort / "linear_probe.json").write_text(
            json.dumps(tasks, indent=2) + "\n"
        )
        for task, rows in tasks.items():
            results[cohort + "/" + task] = {
                m: float(np.mean([r["metrics"][m] for r in rows]))
                for m in protocol.METRICS
            }
    summary = {
        "protocol": "fixed checkpoint on common biological-person outer folds",
        "tasks": results,
        "overall": {
            m: float(np.mean([v[m] for v in results.values()]))
            for m in protocol.METRICS
        },
    }
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
