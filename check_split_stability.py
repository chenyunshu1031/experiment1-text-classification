"""Audit the locked SVM on five predeclared stratified train/validation splits.

Seeds 43-47 are sensitivity checks, not a new hyperparameter search. Each run
fits its TF-IDF vocabulary/IDF on its own training rows and trains exactly 70
epochs with the already selected SVM parameters. No test CSV is opened, and
none of these results can replace the seed-42 submission by this script.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from compare_models import CLASSES, DATA, DATA_SHA, HERE, TFIDF, measure, sha


SPLIT_SEEDS = (43, 44, 45, 46, 47)
MODEL_SEED = 42
EPOCHS = 70
ALPHA = 1e-4
ETA0 = 2.0


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_source():
    assert sha(DATA) == DATA_SHA, "Original labeled CSV changed"
    frame = pd.read_csv(DATA)
    assert frame.shape == (7368, 2) and list(frame.columns) == ["text", "target"]
    assert not frame.isna().any().any()
    labels = frame.target.to_numpy()
    np.testing.assert_array_equal(np.unique(labels), CLASSES)
    return frame, labels


def make_estimator():
    return SGDClassifier(
        loss="hinge", penalty="l2", alpha=ALPHA,
        learning_rate="constant", eta0=ETA0, average=True,
        random_state=MODEL_SEED, shuffle=True, max_iter=1, tol=None,
        early_stopping=False, n_jobs=1,
    )


def run_split(frame, labels, seed, output):
    tr, va = train_test_split(
        np.arange(len(frame)), test_size=0.2,
        random_state=seed, stratify=labels,
    )
    assert len(tr) == 5894 and len(va) == 1474
    assert set(tr).isdisjoint(set(va))
    split_sha = hashlib.sha256(tr.tobytes() + va.tobytes()).hexdigest()

    vec = TfidfVectorizer(**TFIDF)
    xtr = vec.fit_transform(frame.iloc[tr].text).tocsr()
    vocabulary, idf = vec.vocabulary_.copy(), vec.idf_.copy()
    xva = vec.transform(frame.iloc[va].text).tocsr()
    assert vocabulary == vec.vocabulary_ and np.array_equal(idf, vec.idf_)
    assert xtr.shape == (5894, 20000) and xva.shape == (1474, 20000)
    ytr, yva = labels[tr], labels[va]

    model = make_estimator()
    history = []
    folder = output / f"split_seed_{seed}"
    folder.mkdir()
    for epoch in range(1, EPOCHS + 1):
        model.partial_fit(xtr, ytr, classes=CLASSES if epoch == 1 else None)
        training, _ = measure(model, xtr, ytr, "hinge")
        validation, _ = measure(model, xva, yva, "hinge")
        row = {
            "epoch": epoch,
            "train_hinge_loss": training["data_loss"],
            "val_hinge_loss": validation["data_loss"],
            "train_accuracy": training["accuracy"],
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
        }
        history.append(row)
        print(f"split_seed={seed} | Epoch {epoch:03d}/{EPOCHS} | "
              f"train_hinge_loss={row['train_hinge_loss']:.6f} | "
              f"val_hinge_loss={row['val_hinge_loss']:.6f} | "
              f"val_macro_f1={row['val_macro_f1']:.6f}", flush=True)

    pd.DataFrame(history).to_csv(folder / "epoch_history.csv", index=False)
    np.savez_compressed(folder / "split_indices.npz", train_idx=tr, val_idx=va)
    joblib.dump(vec, folder / "vectorizer.joblib")
    joblib.dump(model, folder / "model.joblib")
    final = history[-1]
    result = {
        "split_seed": seed,
        "split_sha256": split_sha,
        "train_rows": len(tr),
        "validation_rows": len(va),
        "model_seed": MODEL_SEED,
        "epochs": EPOCHS,
        "alpha": ALPHA,
        "eta0": ETA0,
        "average": True,
        "train_accuracy": final["train_accuracy"],
        "val_accuracy": final["val_accuracy"],
        "val_macro_f1": final["val_macro_f1"],
        "train_hinge_loss": final["train_hinge_loss"],
        "val_hinge_loss": final["val_hinge_loss"],
        "model_sha256": sha(folder / "model.joblib"),
        "vectorizer_sha256": sha(folder / "vectorizer.joblib"),
    }
    save_json(folder / "metrics.json", result)
    return result


def main(args):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    frame, labels = load_source()
    print(f"Locked split seeds={SPLIT_SEEDS}; model seed={MODEL_SEED}; "
          f"alpha={ALPHA}; eta0={ETA0}; average=True; epochs={EPOCHS}.", flush=True)
    print("Only the labeled CSV is read. New results do not select or replace the submission.", flush=True)
    if args.check_only:
        print("Preflight passed; no features or models fit, no result file written.", flush=True)
        return

    output = HERE / "split_stability_checks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", {
        "purpose": "locked_configuration_split_sensitivity_only",
        "labeled_data_sha256": DATA_SHA,
        "split_seeds": SPLIT_SEEDS,
        "split": "stratified_random_80_20",
        "model_seed": MODEL_SEED,
        "model": "SGDClassifier_hinge_linear_SVM",
        "alpha": ALPHA, "learning_rate": "constant", "eta0": ETA0,
        "average": True, "fixed_epochs_no_reselection": EPOCHS,
        "tfidf": TFIDF,
        "test_csv_opened": False,
        "submission_csv_changed": False,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "script_sha256": sha(Path(__file__)),
    })
    rows = [run_split(frame, labels, seed, output) for seed in SPLIT_SEEDS]
    pd.DataFrame(rows).to_csv(output / "split_results.csv", index=False)
    scores = np.asarray([row["val_macro_f1"] for row in rows], dtype=float)
    accuracies = np.asarray([row["val_accuracy"] for row in rows], dtype=float)
    summary = {
        "split_seeds": SPLIT_SEEDS,
        "n_splits": len(rows),
        "validation_macro_f1_mean": float(scores.mean()),
        "validation_macro_f1_sample_std": float(scores.std(ddof=1)),
        "validation_macro_f1_min": float(scores.min()),
        "validation_macro_f1_max": float(scores.max()),
        "validation_accuracy_mean": float(accuracies.mean()),
        "validation_accuracy_sample_std": float(accuracies.std(ddof=1)),
        "original_seed_42_score_excluded_from_aggregate": True,
        "hyperparameters_or_epochs_selected_from_new_splits": False,
    }
    save_json(output / "summary.json", summary)
    print("\nFixed-configuration sensitivity scores:", flush=True)
    print(pd.DataFrame(rows)[["split_seed", "val_accuracy", "val_macro_f1"]].to_string(index=False), flush=True)
    print(f"Macro-F1 mean={scores.mean():.6f}; sample std={scores.std(ddof=1):.6f}; "
          f"range=[{scores.min():.6f}, {scores.max():.6f}]", flush=True)
    print(f"Fresh results: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    with threadpool_limits(limits=2):
        main(parser.parse_args())
