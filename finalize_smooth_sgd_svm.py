"""Validate and finalize the averaged-SGD linear SVM, without overwriting old work.

Use --check, then --validate, then --finalize --validation-dir PATH.
Validation never opens the unlabeled test CSV. Finalization requires a saved
validation checkpoint and creates a fresh full_refit_* subdirectory.
"""

import argparse
import copy
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
from sklearn.metrics import classification_report
from threadpoolctl import threadpool_limits

from compare_models import (
    CLASSES, DATA, DATA_SHA, HERE, SEED, SPLIT_SHA, TFIDF,
    measure, prepare, sha,
)


ROOT = HERE / "sgd_smooth_final"
TEST = DATA.with_name("test_data_unlabeled.csv")
TEST_SHA = "4882c04b008c055f351f1c9b25c45d601a454077af37e79375601f2465b41615"
ALPHA = 1e-4
ETA0 = 2.0
MAX_EPOCHS = 120


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def estimator():
    return SGDClassifier(
        loss="hinge", penalty="l2", alpha=ALPHA,
        learning_rate="constant", eta0=ETA0, average=True,
        random_state=SEED, shuffle=True, max_iter=1, tol=None,
        early_stopping=False, n_jobs=1,
    )


def validate():
    _, tr, va, xtr, xva, ytr, yva, vec = prepare()
    output = ROOT / stamp()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "source_sha256": DATA_SHA, "split_sha256": SPLIT_SHA,
        "split": "stratified_random_80_20", "seed": SEED,
        "tfidf": TFIDF, "algorithm": "SGDClassifier_hinge_linear_SVM",
        "alpha": ALPHA, "learning_rate": "constant", "eta0": ETA0,
        "average": True, "max_validation_epochs": MAX_EPOCHS,
        "checkpoint_rule": "max_validation_macro_f1_then_accuracy_then_earlier_epoch",
        "selection_informed_by_exploratory_schedule_comparison": True,
        "test_csv_opened": False,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "script_sha256": sha(Path(__file__)),
    }
    save_json(output / "protocol.json", protocol)
    np.savez_compressed(output / "split_indices.npz", train_idx=tr, val_idx=va)
    joblib.dump(vec, output / "validation_vectorizer.joblib")
    model = estimator()
    rows = []
    best_key, best_epoch, best_model = None, None, None
    with (output / "epoch_log.txt").open("w", encoding="utf-8") as log:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.partial_fit(xtr, ytr, classes=CLASSES if epoch == 1 else None)
            train, _ = measure(model, xtr, ytr, "hinge")
            validation, _ = measure(model, xva, yva, "hinge")
            row = {
                "epoch": epoch,
                "train_hinge_loss": train["data_loss"],
                "train_accuracy": train["accuracy"],
                "val_hinge_loss": validation["data_loss"],
                "val_accuracy": validation["accuracy"],
                "val_macro_f1": validation["macro_f1"],
            }
            rows.append(row)
            line = (f"Epoch {epoch:03d}/{MAX_EPOCHS:03d} | "
                    f"train_hinge_loss={row['train_hinge_loss']:.6f} | "
                    f"val_hinge_loss={row['val_hinge_loss']:.6f} | "
                    f"val_macro_f1={row['val_macro_f1']:.6f}")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            key = (validation["macro_f1"], validation["accuracy"], -epoch)
            if best_key is None or key > best_key:
                best_key, best_epoch, best_model = key, epoch, copy.deepcopy(model)
    history = pd.DataFrame(rows)
    history.to_csv(output / "epoch_history.csv", index=False)
    joblib.dump(best_model, output / "validation_model.joblib")
    loaded = joblib.load(output / "validation_model.joblib")
    np.testing.assert_array_equal(loaded.predict(xva), best_model.predict(xva))
    train, _ = measure(loaded, xtr, ytr, "hinge")
    val, prediction = measure(loaded, xva, yva, "hinge")
    np.savez_compressed(output / "validation_predictions.npz", target=yva, prediction=prediction)
    (output / "classification_report.txt").write_text(
        classification_report(yva, prediction, digits=5, zero_division=0), encoding="utf-8")
    selection = {
        "selected_epoch": int(best_epoch), "max_epochs": MAX_EPOCHS,
        "alpha": ALPHA, "eta0": ETA0, "average": True,
        "train_accuracy": train["accuracy"], "val_accuracy": val["accuracy"],
        "val_macro_f1": val["macro_f1"],
        "train_hinge_loss": train["data_loss"],
        "val_hinge_loss": val["data_loss"],
        "epoch_history_sha256": sha(output / "epoch_history.csv"),
        "validation_model_sha256": sha(output / "validation_model.joblib"),
    }
    save_json(output / "locked_selection.json", selection)
    (output / "VALIDATION_COMPLETE.txt").write_text(
        "Validation checkpoint locked. No test data used.\n", encoding="utf-8")
    print(f"\nLOCKED: epoch={best_epoch}, val_accuracy={val['accuracy']:.6f}, "
          f"val_macro_f1={val['macro_f1']:.6f}", flush=True)
    print(f"Validation directory: {output}", flush=True)
    print("No test CSV opened; no prediction CSV written.", flush=True)


def finalize(validation_dir):
    folder = Path(validation_dir).resolve()
    if not folder.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Validation directory must be inside {ROOT}")
    if not (folder / "VALIDATION_COMPLETE.txt").is_file():
        raise ValueError("Validation run is incomplete")
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    selected = json.loads((folder / "locked_selection.json").read_text(encoding="utf-8"))
    history_file = folder / "epoch_history.csv"
    history = pd.read_csv(history_file)
    assert protocol["source_sha256"] == DATA_SHA
    assert protocol["split_sha256"] == SPLIT_SHA
    assert protocol["tfidf"] == {key: list(value) if isinstance(value, tuple) else value
                                  for key, value in TFIDF.items()}
    assert selected["epoch_history_sha256"] == sha(history_file)
    assert selected["alpha"] == ALPHA and selected["eta0"] == ETA0
    assert selected["average"] is True and selected["max_epochs"] == MAX_EPOCHS
    assert len(history) == MAX_EPOCHS
    chosen_row = history.loc[history.epoch.eq(selected["selected_epoch"])].iloc[0]
    assert abs(float(chosen_row.val_macro_f1) - selected["val_macro_f1"]) < 1e-12
    assert abs(float(chosen_row.val_accuracy) - selected["val_accuracy"]) < 1e-12
    assert int(selected["selected_epoch"]) == int(history.sort_values(
        ["val_macro_f1", "val_accuracy", "epoch"],
        ascending=[False, False, True], kind="stable").iloc[0].epoch)
    print(f"Locked checkpoint verified: epoch={selected['selected_epoch']}, "
          f"val_macro_f1={selected['val_macro_f1']:.6f}", flush=True)

    # Only now read the teacher's unlabeled test CSV.
    assert sha(TEST) == TEST_SHA, "Teacher's unlabeled test CSV changed"
    teacher = pd.read_csv(TEST)
    assert teacher.shape == (2457, 1) and list(teacher.columns) == ["text"]
    assert not teacher.isna().any().any()
    assert sha(DATA) == DATA_SHA
    labeled = pd.read_csv(DATA)
    assert labeled.shape == (7368, 2) and list(labeled.columns) == ["text", "target"]
    y = labeled.target.to_numpy()
    np.testing.assert_array_equal(np.unique(y), CLASSES)
    vectorizer = TfidfVectorizer(**TFIDF)
    xfull = vectorizer.fit_transform(labeled.text).tocsr()
    vocabulary, idf = vectorizer.vocabulary_.copy(), vectorizer.idf_.copy()
    xtest = vectorizer.transform(teacher.text).tocsr()
    assert vectorizer.vocabulary_ == vocabulary and np.array_equal(vectorizer.idf_, idf)
    assert xfull.shape == (7368, 20000) and xtest.shape == (2457, 20000)
    output = folder / f"full_refit_{stamp()}"
    output.mkdir(exist_ok=False)
    model = estimator()
    rows = []
    count = int(selected["selected_epoch"])
    for epoch in range(1, count + 1):
        model.partial_fit(xfull, y, classes=CLASSES if epoch == 1 else None)
        train, _ = measure(model, xfull, y, "hinge")
        rows.append({"epoch": epoch, "train_hinge_loss": train["data_loss"],
                     "train_accuracy": train["accuracy"]})
        print(f"Full refit {epoch:03d}/{count:03d} | "
              f"train_hinge_loss={train['data_loss']:.6f}", flush=True)
    pd.DataFrame(rows).to_csv(output / "full_refit_epoch_history.csv", index=False)
    prediction = model.predict(xtest)
    assert prediction.shape == (2457,) and set(prediction).issubset(set(CLASSES))
    result = output / "predictions.csv"
    pd.DataFrame({"prediction": prediction}).to_csv(result, index=False, header=False)
    reloaded = pd.read_csv(result, header=None)
    assert reloaded.shape == (2457, 1)
    np.testing.assert_array_equal(reloaded.iloc[:, 0].to_numpy(), prediction)
    joblib.dump(model, output / "model.joblib")
    joblib.dump(vectorizer, output / "vectorizer.joblib")
    replay = joblib.load(output / "model.joblib").predict(
        joblib.load(output / "vectorizer.joblib").transform(teacher.text))
    np.testing.assert_array_equal(replay, prediction)
    manifest = {
        "locked_validation_directory": str(folder),
        "selected_epoch": count, "alpha": ALPHA,
        "learning_rate": "constant", "eta0": ETA0, "average": True,
        "source_sha256": DATA_SHA, "test_sha256": TEST_SHA,
        "prediction_count": len(prediction),
        "prediction_class_counts": {str(k): int(v) for k, v in
                                    pd.Series(prediction).value_counts().sort_index().items()},
        "predictions_sha256": sha(result),
        "official_test_accuracy_available": False,
        "old_prediction_files_overwritten": False,
    }
    save_json(output / "manifest.json", manifest)
    (output / "COMPLETE.txt").write_text(
        "Full labeled refit and unlabeled prediction complete.\n", encoding="utf-8")
    print(f"\nPredictions: {len(prediction)}; classes: {manifest['prediction_class_counts']}",
          flush=True)
    print(f"New submission CSV: {result}", flush=True)
    print("No test score can be computed because test labels are unavailable.", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--validation-dir", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.check:
            prepare()
            print("Check passed. No model trained, no test opened, no files written.", flush=True)
        elif args.validate:
            validate()
        else:
            if args.validation_dir is None:
                parser.error("--finalize requires --validation-dir")
            finalize(args.validation_dir)


if __name__ == "__main__":
    main()
