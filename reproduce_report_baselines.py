"""Reproduce two historical report comparisons on the locked validation split.

The 5000-feature NB result and the 20-epoch SVM alpha scan were exploratory
validation experiments. They are not the final averaged-SVM submission model.
This script never opens the unlabeled test CSV or changes submission files.
"""

import argparse
import copy
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
from sklearn.naive_bayes import MultinomialNB
from threadpoolctl import threadpool_limits

from compare_models import (
    CLASSES, DATA_SHA, HERE, SEED, SPLIT_SHA, TFIDF, measure, prepare, sha,
)


ALPHAS = (1e-5, 1e-4, 1e-3)
SVM_EPOCHS = 20
RAW5000_TFIDF = dict(
    max_features=5000, lowercase=True, analyzer="word",
    token_pattern=r"(?u)\b\w\w+\b", stop_words=None,
    ngram_range=(1, 1), min_df=1, max_df=1.0,
    binary=False, use_idf=True, smooth_idf=True,
    sublinear_tf=False, norm="l2", dtype=np.float64,
)
REFERENCE = {
    "nb_raw5000": {"val_macro_f1": 0.8921926312556477, "val_accuracy": 0.8914518317503393},
    "svm_1e-05": {"selected_epoch": 20, "val_macro_f1": 0.9384131050404502},
    "svm_0.0001": {"selected_epoch": 4, "val_macro_f1": 0.9445998225942513},
    "svm_0.001": {"selected_epoch": 9, "val_macro_f1": 0.9305180887281607},
}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def reference_check(key, result):
    expected = REFERENCE[key]
    checks = {}
    for field, target in expected.items():
        observed = result[field]
        checks[field] = bool(np.isclose(observed, target, rtol=0, atol=1e-8))
    return checks


def fit_raw5000_nb(frame, tr, va, ytr, yva, output):
    vec = TfidfVectorizer(**RAW5000_TFIDF)
    xtr = vec.fit_transform(frame.iloc[tr].text).tocsr()
    vocabulary, idf = vec.vocabulary_.copy(), vec.idf_.copy()
    xva = vec.transform(frame.iloc[va].text).tocsr()
    assert vocabulary == vec.vocabulary_ and np.array_equal(idf, vec.idf_)
    assert xtr.shape == (5894, 5000) and xva.shape == (1474, 5000)
    model = MultinomialNB(alpha=0.1)
    model.fit(xtr, ytr)
    training, _ = measure(model, xtr, ytr, "nb")
    validation, predictions = measure(model, xva, yva, "nb")
    result = {
        "experiment": "nb_raw5000",
        "tfidf": {k: (str(v) if k == "dtype" else v) for k, v in RAW5000_TFIDF.items()},
        "alpha": 0.1,
        "train_accuracy": training["accuracy"],
        "val_accuracy": validation["accuracy"],
        "val_macro_f1": validation["macro_f1"],
        "train_log_loss": training["data_loss"],
        "val_log_loss": validation["data_loss"],
        "reference_match": None,
    }
    result["reference_match"] = reference_check("nb_raw5000", result)
    folder = output / "nb_raw5000"
    folder.mkdir()
    joblib.dump(vec, folder / "vectorizer.joblib")
    joblib.dump(model, folder / "model.joblib")
    np.savez_compressed(folder / "validation_predictions.npz", target=yva, prediction=predictions)
    save_json(folder / "metrics.json", result)
    print(f"Raw 5000-dimensional NB: validation Macro-F1={validation['macro_f1']:.6f}; "
          f"reference_match={result['reference_match']}", flush=True)
    return result


def fit_early_svm(xtr, xva, ytr, yva, output):
    results = []
    for alpha in ALPHAS:
        model = SGDClassifier(
            loss="hinge", penalty="l2", alpha=alpha,
            learning_rate="optimal", random_state=SEED, shuffle=True,
            max_iter=1, tol=None, early_stopping=False, n_jobs=1,
        )
        history = []
        best_key, best_model, best_epoch = None, None, None
        for epoch in range(1, SVM_EPOCHS + 1):
            model.partial_fit(xtr, ytr, classes=CLASSES if epoch == 1 else None)
            training, _ = measure(model, xtr, ytr, "hinge")
            validation, _ = measure(model, xva, yva, "hinge")
            row = {
                "epoch": epoch, "train_hinge_loss": training["data_loss"],
                "train_accuracy": training["accuracy"],
                "val_hinge_loss": validation["data_loss"],
                "val_accuracy": validation["accuracy"],
                "val_macro_f1": validation["macro_f1"],
            }
            history.append(row)
            print(f"alpha={alpha:g} | Epoch {epoch:02d}/{SVM_EPOCHS} | "
                  f"train_hinge_loss={row['train_hinge_loss']:.6f} | "
                  f"val_hinge_loss={row['val_hinge_loss']:.6f} | "
                  f"val_macro_f1={row['val_macro_f1']:.6f}", flush=True)
            key = (validation["macro_f1"], validation["accuracy"], -epoch)
            if best_key is None or key > best_key:
                best_key, best_model, best_epoch = key, copy.deepcopy(model), epoch

        selected_train, _ = measure(best_model, xtr, ytr, "hinge")
        selected_val, predictions = measure(best_model, xva, yva, "hinge")
        result = {
            "alpha": alpha, "learning_rate": "optimal", "average": False,
            "selected_epoch": best_epoch, "epochs_run": SVM_EPOCHS,
            "selection_rule": "max_val_macro_f1_then_accuracy_then_earlier_epoch",
            "train_accuracy": selected_train["accuracy"],
            "val_accuracy": selected_val["accuracy"],
            "val_macro_f1": selected_val["macro_f1"],
            "train_hinge_loss": selected_train["data_loss"],
            "val_hinge_loss": selected_val["data_loss"],
        }
        result["reference_match"] = reference_check(f"svm_{alpha:g}", result)
        folder = output / f"svm_alpha_{alpha:g}"
        folder.mkdir()
        pd.DataFrame(history).to_csv(folder / "epoch_history.csv", index=False)
        joblib.dump(best_model, folder / "validation_model.joblib")
        np.savez_compressed(folder / "validation_predictions.npz", target=yva, prediction=predictions)
        save_json(folder / "metrics.json", result)
        print(f"alpha={alpha:g} | selected_epoch={best_epoch} | "
              f"val_macro_f1={selected_val['macro_f1']:.6f} | "
              f"reference_match={result['reference_match']}", flush=True)
        results.append(result)
    pd.DataFrame(results).to_csv(output / "early_svm_comparison.csv", index=False)
    return results


def main(args):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    frame, tr, va, xtr, xva, ytr, yva, _ = prepare()
    assert hashlib.sha256(tr.tobytes() + va.tobytes()).hexdigest() == SPLIT_SHA
    print("Historical report comparisons only; no unlabeled test CSV or submission file opened.", flush=True)
    print(f"Models={args.models}; seed={SEED}; source={DATA_SHA[:16]}; "
          f"split={SPLIT_SHA[:16]}; final TF-IDF={TFIDF}", flush=True)
    if args.check_only:
        print("Preflight passed; no model trained and no result file written.", flush=True)
        return
    output = HERE / "report_baseline_reproduction" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", {
        "purpose": "historical_validation_report_comparisons_not_final_submission",
        "source_sha256": DATA_SHA, "split_sha256": SPLIT_SHA,
        "seed": SEED, "models": args.models,
        "early_svm_alphas": ALPHAS, "early_svm_max_epochs": SVM_EPOCHS,
        "early_svm_tfidf": TFIDF,
        "raw5000_tfidf": {k: (str(v) if k == "dtype" else v) for k, v in RAW5000_TFIDF.items()},
        "test_csv_opened": False, "submission_csv_changed": False,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "script_sha256": sha(Path(__file__)),
    })
    np.savez_compressed(output / "split_indices.npz", train_idx=tr, val_idx=va)
    results = {}
    if "nb" in args.models:
        results["nb"] = fit_raw5000_nb(frame, tr, va, ytr, yva, output)
    if "svm" in args.models:
        results["svm"] = fit_early_svm(xtr, xva, ytr, yva, output)
    save_json(output / "summary.json", results)
    matches = []
    if "nb" in results:
        matches.extend(results["nb"]["reference_match"].values())
    if "svm" in results:
        for row in results["svm"]:
            matches.extend(row["reference_match"].values())
    print(f"All historical reference values reproduced: {all(matches)}", flush=True)
    print(f"Fresh results: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--models", nargs="+", choices=("nb", "svm"), default=["nb", "svm"])
    with threadpool_limits(limits=2):
        main(parser.parse_args())
