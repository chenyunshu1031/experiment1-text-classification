"""Controlled MLP architecture ablation on the locked validation split.

Only hidden_layer_sizes changes. The teacher's unlabeled test file and final
submission are not opened or modified.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from compare_models import (DATA_SHA, HERE, SEED, SPLIT_SHA, TFIDF,
                            prepare, train_one)


ARCHITECTURES = ((128,), (64, 32))
MAX_EPOCHS = 120


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(hidden):
    return MLPClassifier(
        hidden_layer_sizes=hidden, alpha=0.0001, activation="relu",
        solver="sgd", batch_size=64, learning_rate="constant",
        learning_rate_init=0.01, momentum=0.9, nesterovs_momentum=True,
        random_state=SEED, shuffle=True, max_iter=1, early_stopping=False,
    )


def summary_row(label, hidden, metrics):
    return {
        "architecture": label,
        "hidden_layer_sizes": str(hidden),
        "epochs_run": metrics["epochs_run"],
        "selected_epoch": metrics["selected_epoch"],
        "train_accuracy": metrics["train"]["accuracy"],
        "train_log_loss": metrics["train"]["multiclass_log_loss"],
        "val_accuracy": metrics["validation"]["accuracy"],
        "val_macro_f1": metrics["validation"]["macro_f1"],
        "val_log_loss": metrics["validation"]["multiclass_log_loss"],
        "fit_seconds": metrics["seconds"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    reference = make_model((64,)).get_params()
    _, _, _, xtr, xva, ytr, yva, _ = prepare()
    print("Controlled variable: hidden_layer_sizes only", flush=True)
    print("Baseline=(64,); candidates=(128,), (64,32); max_epochs=120", flush=True)
    print("Same data, split, TF-IDF, SGD settings, seed and checkpoint rule.", flush=True)
    print("Teacher test and final prediction CSV are not opened.", flush=True)
    if args.check_only:
        return

    output = HERE / "mlp_architecture_ablation" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps({
        "data_sha256": DATA_SHA,
        "split_sha256": SPLIT_SHA,
        "seed": SEED,
        "tfidf": TFIDF,
        "architectures": [[64], [128], [64, 32]],
        "max_epochs": MAX_EPOCHS,
        "selection": "validation multiclass log loss; min_delta=0.0005; patience=10",
        "teacher_test_read": False,
        "final_submission_modified": False,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "script_sha256": file_sha(Path(__file__)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    with threadpool_limits(limits=2):
        for hidden in ((64,),) + ARCHITECTURES:
            model = make_model(hidden)
            candidate = model.get_params().copy()
            expected = reference.copy()
            candidate.pop("hidden_layer_sizes")
            expected.pop("hidden_layer_sizes")
            assert candidate == expected, "An uncontrolled model parameter changed"
            name = "hidden_" + "_".join(map(str, hidden))
            folder = output / name
            print(f"\nTraining {name}", flush=True)
            train_one(name, "mlp", model, xtr, xva, ytr, yva,
                      MAX_EPOCHS, folder)
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            assert tuple(metrics["parameters"]["hidden_layer_sizes"]) == hidden
            rows.append(summary_row("baseline" if hidden == (64,) else "candidate",
                                    hidden, metrics))
            pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)

    table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    print("\nArchitecture comparison (same validation split):", flush=True)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"), flush=True)
    print(f"Results: {output}", flush=True)
    (output / "COMPLETE.txt").write_text(
        "Controlled MLP architecture comparison complete.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
