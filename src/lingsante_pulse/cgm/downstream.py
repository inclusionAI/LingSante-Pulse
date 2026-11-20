"""Example classifier deployment, separate from the repeated benchmark protocol."""

import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["fit", "predict"])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--label", help="NPZ key, e.g. label_diabetes; used only for fit")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    with np.load(a.input, allow_pickle=False) as data:
        x = np.asarray(data["embedding"], dtype=np.float64)
        ids = data["window_id"]
        if a.mode == "fit":
            if not a.label or a.label not in data:
                p.error("fit requires an existing --label key")
            y = data[a.label]
            valid = np.isfinite(y) & (y >= 0)
            if len(np.unique(y[valid])) < 2:
                p.error("Need at least two labeled classes")
            if a.model.exists():
                raise FileExistsError(a.model)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0, solver="lbfgs", max_iter=1000, random_state=43
                ),
            )
            model.fit(x[valid], y[valid])
            a.model.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, a.model)
        else:
            if a.output is None:
                p.error("predict requires --output")
            if a.output.exists():
                raise FileExistsError(a.output)
            # Only load classifiers created locally or from a trusted source.
            model = joblib.load(a.model)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                a.output,
                window_id=ids,
                classes=model.classes_,
                probability=model.predict_proba(x),
                prediction=model.predict(x),
            )


if __name__ == "__main__":
    main()
