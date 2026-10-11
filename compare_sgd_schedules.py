"""Explore gentler SGD schedules on the locked validation split only.

This is an exploratory comparison, not a replacement for the submitted model.
The teacher's unlabeled test set and existing prediction CSV are never read.
"""

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, log_loss
from threadpoolctl import threadpool_limits

from compare_models import CLASSES, HERE, SEED, prepare


OUT = HERE / "sgd_schedule_exploration" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
SCHEDULES = (
    ("optimal_averaged", "optimal", None, True),
    ("constant_0p1", "constant", 0.1, False),
    ("constant_0p1_averaged", "constant", 0.1, True),
    ("constant_1_averaged", "constant", 1.0, True),
    ("constant_2_averaged", "constant", 2.0, True),
)


def metrics(model, x, y, kind):
    decision = model.decision_function(x)
    signs = np.where(y[:, None] == CLASSES[None, :], 1.0, -1.0)
    if kind == "svm":
        data_loss = float(np.maximum(0.0, 1.0 - signs * decision).mean())
    else:
        data_loss = float(np.logaddexp(0.0, -signs * decision).mean())
    score = float(f1_score(y, model.predict(x), average="macro"))
    multiclass_loss = (float(log_loss(y, model.predict_proba(x), labels=CLASSES))
                       if kind == "logistic" else None)
    return data_loss, multiclass_loss, score


def main(epochs, only):
    with threadpool_limits(limits=2):
        _, _, _, xtr, xva, ytr, yva, _ = prepare()
        OUT.mkdir(parents=True, exist_ok=False)
        summary = []
        for kind, alpha, loss_name in (("logistic", 1e-5, "log_loss"),
                                       ("svm", 1e-4, "hinge")):
            for label, learning_rate, eta0, average in SCHEDULES:
                if only and label not in only:
                    continue
                ident = f"{kind}_{label}"
                kwargs = dict(loss=loss_name, penalty="l2", alpha=alpha,
                              learning_rate=learning_rate, average=average,
                              random_state=SEED, shuffle=True, max_iter=1,
                              tol=None, early_stopping=False, n_jobs=1)
                if eta0 is not None:
                    kwargs["eta0"] = eta0
                model = SGDClassifier(**kwargs)
                rows = []
                print(f"\n{ident}: {kwargs}", flush=True)
                for epoch in range(1, epochs + 1):
                    model.partial_fit(xtr, ytr, classes=CLASSES if epoch == 1 else None)
                    tr_loss, tr_multiclass, _ = metrics(model, xtr, ytr, kind)
                    va_loss, va_multiclass, va_f1 = metrics(model, xva, yva, kind)
                    row = dict(epoch=epoch, train_data_loss=tr_loss,
                               val_data_loss=va_loss, train_multiclass_log_loss=tr_multiclass,
                               val_multiclass_log_loss=va_multiclass,
                               val_macro_f1=va_f1)
                    rows.append(row)
                    print(f"Epoch {epoch:03d}/{epochs:03d} | train_loss={tr_loss:.6f} | "
                          f"val_loss={va_loss:.6f} | val_macro_f1={va_f1:.6f}", flush=True)
                table = pd.DataFrame(rows)
                table.to_csv(OUT / f"{ident}_history.csv", index=False)
                chosen = table.sort_values(["val_macro_f1", "epoch"],
                                           ascending=[False, True]).iloc[0]
                key = "val_multiclass_log_loss" if kind == "logistic" else "val_data_loss"
                summary.append(dict(model=kind, schedule=label,
                                    best_f1_epoch=int(chosen.epoch),
                                    best_val_macro_f1=float(chosen.val_macro_f1),
                                    val_loss_at_best_f1=float(chosen[key]),
                                    first_val_loss=float(table[key].iloc[0]),
                                    final_val_loss=float(table[key].iloc[-1]),
                                    min_val_loss=float(table[key].min()),
                                    min_val_loss_epoch=int(table.loc[table[key].idxmin(), "epoch"])))
        result = pd.DataFrame(summary).sort_values(["model", "best_val_macro_f1"],
                                                    ascending=[True, False])
        result.to_csv(OUT / "summary.csv", index=False)
        print("\nSchedule comparison (not a submission change):", flush=True)
        print(result.to_string(index=False), flush=True)
        print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--only", nargs="*", default=[])
    arguments = parser.parse_args()
    main(arguments.epochs, set(arguments.only))
