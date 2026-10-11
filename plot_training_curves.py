"""Recreate the report's three-panel training/validation loss figure.

This script uses the small, tracked CSV snapshots in results_summary/ and
does not load course data, fit a model, or read the unlabeled test set.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "results_summary"
SPECS = (
    ("Logistic classification", "logistic_epoch_history.csv",
     "train_multiclass_log_loss", "val_multiclass_log_loss", 20,
     0.9350240424061529, "Log loss"),
    ("Linear SVM", "svm_epoch_history.csv",
     "train_hinge_loss", "val_hinge_loss", 70,
     0.9444402010987307, "Hinge loss"),
    ("MLP", "mlp_epoch_history.csv",
     "train_multiclass_log_loss", "val_multiclass_log_loss", 119,
     0.9364774544950641, "Log loss"),
)


def load_history(name, filename, train_col, val_col, selected_epoch, expected_f1):
    path = SUMMARY / filename
    frame = pd.read_csv(path)
    required = {"epoch", "val_macro_f1", train_col, val_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    epochs = frame["epoch"].to_numpy()
    if len(frame) != 120 or not np.array_equal(epochs, np.arange(1, 121)):
        raise ValueError(f"{path}: expected exactly epochs 1–120")
    if not np.isfinite(frame[[train_col, val_col, "val_macro_f1"]].to_numpy()).all():
        raise ValueError(f"{path}: non-finite values")
    observed_f1 = float(frame.loc[frame.epoch.eq(selected_epoch), "val_macro_f1"].iloc[0])
    if not np.isclose(observed_f1, expected_f1, rtol=0, atol=1e-12):
        raise ValueError(f"{name}: selected-epoch F1 differs from recorded result")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate the CSVs without creating an image")
    parser.add_argument("--output", type=Path, default=SUMMARY / "loss_curves.png",
                        help="Output PNG path")
    args = parser.parse_args()

    histories = [
        load_history(name, filename, train_col, val_col, selected_epoch, expected_f1)
        for name, filename, train_col, val_col, selected_epoch, expected_f1, _ in SPECS
    ]
    print("Three 120-epoch histories and selected checkpoints verified.", flush=True)
    if args.check_only:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.25), constrained_layout=True)
    for ax, spec, frame in zip(axes, SPECS, histories):
        name, _, train_col, val_col, selected_epoch, _, loss_label = spec
        ax.plot(frame.epoch, frame[train_col], color="#1f77b4", label="Training", lw=1.8)
        ax.plot(frame.epoch, frame[val_col], color="#d95319", label="Validation", lw=1.8)
        ax.axvline(selected_epoch, color="0.35", linestyle=":", lw=1.2)
        selected = frame.loc[frame.epoch.eq(selected_epoch)].iloc[0]
        ax.scatter([selected_epoch], [selected[val_col]], color="#d95319", s=22, zorder=3)
        ax.set(title=f"{name} (epoch {selected_epoch})", xlabel="Epoch", ylabel=loss_label,
               xlim=(1, 120), ylim=(0, None))
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
