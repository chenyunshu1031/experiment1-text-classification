"""Controlled preprocessing ablation for the locked SGD linear-SVM recipe.

This is an exploratory validation experiment. It never opens the teacher's
unlabeled test CSV or overwrites the existing final prediction file.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from compare_models import CLASSES, SEED, SPLIT_SHA, TFIDF, measure, prepare


HERE = Path(__file__).resolve().parent
ALPHA = 0.0001
EPOCHS = 4
PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
BLANK_LINE = re.compile(r"\r?\n[ \t]*\r?\n")


def estimator(alpha):
    # Historical four-epoch ablation configuration, distinct from final model.
    return SGDClassifier(loss="hinge", penalty="l2", alpha=alpha,
                         learning_rate="optimal", random_state=SEED,
                         shuffle=True, max_iter=1, tol=None,
                         early_stopping=False, n_jobs=1)


def body_only(text):
    boundary = BLANK_LINE.search(text)
    return text[boundary.end():] if boundary else text


def subject_and_body(text):
    boundary = BLANK_LINE.search(text)
    if boundary is None:
        return text
    header = text[:boundary.start()].splitlines()
    for index, line in enumerate(header):
        match = re.match(r"^Subject:[ \t]*(.*)$", line, flags=re.IGNORECASE)
        if match:
            parts = [match.group(1).strip()]
            for continuation in header[index + 1:]:
                if not continuation.startswith((" ", "\t")):
                    break
                parts.append(continuation.strip())
            subject = " ".join(parts).strip()
            return subject + "\n\n" + text[boundary.end():] if subject else text[boundary.end():]
    return text[boundary.end():]


def remove_punctuation(text):
    # Replace with a space so words on either side are not joined together.
    return PUNCTUATION.sub(" ", text)


def build_stemmer():
    # Reuse sklearn's own lowercasing, accent removal, tokenization and
    # English-stop-word removal before stemming each remaining unigram.
    unigram_config = {**TFIDF, "ngram_range": (1, 1)}
    analyzer = TfidfVectorizer(**unigram_config).build_analyzer()
    porter = PorterStemmer()

    def stem_text(text):
        return " ".join(porter.stem(token) for token in analyzer(text))

    return stem_text


def fit_features(train_text, val_text, config):
    vectorizer = TfidfVectorizer(**config)
    x_train = vectorizer.fit_transform(train_text).tocsr()
    vocabulary = vectorizer.vocabulary_.copy()
    idf = vectorizer.idf_.copy()
    x_val = vectorizer.transform(val_text).tocsr()
    assert vectorizer.vocabulary_ == vocabulary
    np.testing.assert_array_equal(vectorizer.idf_, idf)
    return x_train, x_val, vectorizer


def main():
    frame, train_idx, val_idx, baseline_train, baseline_val, y_train, y_val, _ = prepare()
    print("Locked split:", SPLIT_SHA[:16], "| seed:", SEED)
    print("Locked SGD-SVM: alpha=0.0001; exactly 4 epochs; no epoch reselection")
    print("No teacher test CSV is opened; existing predictions are not changed.")

    stem_text = build_stemmer()
    stemmed_config = {**TFIDF, "lowercase": False, "strip_accents": None,
                      "stop_words": None}
    variants = [
        ("current_stopwords", lambda text: text, TFIDF),
        ("without_stopwords", lambda text: text, {**TFIDF, "stop_words": None}),
        ("body_only", body_only, TFIDF),
        ("subject_and_body", subject_and_body, TFIDF),
        ("punctuation_removed", remove_punctuation, TFIDF),
        ("porter_stemming", stem_text, stemmed_config),
        ("subject_body_stemming", lambda text: stem_text(subject_and_body(text)), stemmed_config),
    ]

    out = HERE / "preprocessing_ablation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    (out / "protocol.json").write_text(json.dumps({
        "split_sha256": SPLIT_SHA, "seed": SEED,
        "train_rows": len(train_idx), "validation_rows": len(val_idx),
        "model": "SGDClassifier(loss=hinge, penalty=l2, alpha=0.0001)",
        "epochs": EPOCHS, "selection": "none; frozen model hyperparameters and epochs",
        "variants": [name for name, _, _ in variants],
        "tfidf_baseline": TFIDF,
        "teacher_test_used": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    train_raw = frame.iloc[train_idx].text.tolist()
    val_raw = frame.iloc[val_idx].text.tolist()
    summaries = []
    for name, transform, config in variants:
        print(f"\n=== {name} ===", flush=True)
        train_text = [transform(text) for text in train_raw]
        val_text = [transform(text) for text in val_raw]
        if name == "current_stopwords":
            x_train, x_val = baseline_train, baseline_val
            vocabulary_size = x_train.shape[1]
            feature_change_train = 0
            feature_change_val = 0
        else:
            x_train, x_val, vectorizer = fit_features(train_text, val_text, config)
            vocabulary_size = len(vectorizer.vocabulary_)
            feature_change_train = None
            feature_change_val = None
            if name == "punctuation_removed":
                feature_change_train = int((x_train != baseline_train).nnz)
                feature_change_val = int((x_val != baseline_val).nnz)
        assert x_train.shape[0] == len(y_train) and x_val.shape[0] == len(y_val)
        model = estimator(ALPHA)
        history = []
        for epoch in range(1, EPOCHS + 1):
            model.partial_fit(x_train, y_train, classes=CLASSES if epoch == 1 else None)
            train_metrics, _ = measure(model, x_train, y_train, "hinge")
            val_metrics, _ = measure(model, x_val, y_val, "hinge")
            row = {
                "variant": name, "epoch": epoch,
                "train_hinge_loss": train_metrics["data_loss"],
                "val_hinge_loss": val_metrics["data_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            history.append(row)
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} | "
                  f"train_hinge_loss={row['train_hinge_loss']:.6f} | "
                  f"val_hinge_loss={row['val_hinge_loss']:.6f} | "
                  f"val_macro_f1={row['val_macro_f1']:.6f}", flush=True)
        pd.DataFrame(history).to_csv(out / f"{name}_epoch_history.csv", index=False)
        final = history[-1]
        summaries.append({
            "variant": name,
            "train_changed_text_rows": sum(a != b for a, b in zip(train_raw, train_text)),
            "val_changed_text_rows": sum(a != b for a, b in zip(val_raw, val_text)),
            "train_empty_text_rows": sum(not text.strip() for text in train_text),
            "val_empty_text_rows": sum(not text.strip() for text in val_text),
            "vocabulary_size": vocabulary_size,
            "feature_difference_train_nnz": feature_change_train,
            "feature_difference_val_nnz": feature_change_val,
            "val_accuracy": final["val_accuracy"],
            "val_macro_f1": final["val_macro_f1"],
            "train_hinge_loss": final["train_hinge_loss"],
            "val_hinge_loss": final["val_hinge_loss"],
        })
    table = pd.DataFrame(summaries)
    baseline_f1 = table.loc[table.variant == "current_stopwords", "val_macro_f1"].iloc[0]
    table["delta_macro_f1_vs_current"] = table.val_macro_f1 - baseline_f1
    table.to_csv(out / "comparison.csv", index=False)
    print("\nFixed-configuration preprocessing comparison:")
    print(table[["variant", "val_accuracy", "val_macro_f1", "delta_macro_f1_vs_current",
                 "train_empty_text_rows", "val_empty_text_rows"]].to_string(index=False))
    print("Results:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
