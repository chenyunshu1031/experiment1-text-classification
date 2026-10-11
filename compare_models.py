"""Compare four text classifiers on a fixed train/validation split.

Run from any directory. The teacher test CSV is never opened. Results are
written under sgd_retest/ in a fresh timestamped directory.

MultinomialNB has no SGD solver. SGDClassifier(hinge) is a linear SVM family
method, not SVC; SGDClassifier(log_loss) is OvR logistic classification, not
the original multinomial LogisticRegression(lbfgs). MLP uses solver='sgd'.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('EXPERIMENT1_DATA_DIR', HERE / 'data'))
DATA = DATA_DIR / 'train_data.csv'
DATA_SHA = '484667d165458da7f73ba36d172d67e14864e1aa5ab1ee6262e2966dd295c863'
SPLIT_SHA = '77593837c46c1db66d096edb63ef1fde73be7f10ffe3a2f491efe17931534ecd'
SEED = 42
CLASSES = np.arange(10)
TFIDF = dict(lowercase=True, stop_words='english', strip_accents='unicode',
             sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.95,
             max_features=20000)
LINEAR_ALPHAS = (0.00001, 0.0001, 0.001)
MLP_RATES = (0.001, 0.01, 0.1)
MIN_DELTA = 0.0005
PATIENCE = 10


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def prepare():
    assert sha(DATA) == DATA_SHA, 'Original source data changed'
    frame = pd.read_csv(DATA)
    assert frame.shape == (7368, 2) and list(frame.columns) == ['text', 'target']
    assert not frame.isna().any().any()
    labels = frame.target.to_numpy()
    np.testing.assert_array_equal(np.unique(labels), CLASSES)
    tr, va = train_test_split(np.arange(len(frame)), test_size=0.2,
                              random_state=SEED, stratify=labels)
    assert hashlib.sha256(tr.tobytes() + va.tobytes()).hexdigest() == SPLIT_SHA
    vec = TfidfVectorizer(**TFIDF)
    xtr = vec.fit_transform(frame.iloc[tr].text).tocsr()
    old_vocabulary, old_idf = vec.vocabulary_.copy(), vec.idf_.copy()
    xva = vec.transform(frame.iloc[va].text).tocsr()
    assert vec.vocabulary_ == old_vocabulary and np.array_equal(vec.idf_, old_idf)
    np.testing.assert_allclose(old_idf, np.log((1 + len(tr)) /
                               (1 + np.asarray((xtr > 0).sum(axis=0)).ravel())) + 1,
                               rtol=1e-12, atol=1e-12)
    assert xtr.shape == (5894, 20000) and xva.shape == (1474, 20000)
    print(f'Data and features verified: source={DATA_SHA[:16]}, split={SPLIT_SHA[:16]}, '
          f'features={xtr.shape}/{xva.shape}.', flush=True)
    print('Vocabulary and IDF fitted on training rows only; no test CSV opened.', flush=True)
    return frame, tr, va, xtr, xva, labels[tr], labels[va], vec


def measure(model, x, y, kind):
    pred = model.predict(x)
    result = {'accuracy': float(accuracy_score(y, pred)),
              'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
              'multiclass_log_loss': None, 'data_loss': None, 'loss_name': None}
    if kind in ('hinge', 'log_loss'):
        decision = model.decision_function(x)
        assert np.isfinite(decision).all()
        signs = np.where(y[:, None] == model.classes_[None, :], 1.0, -1.0)
        if kind == 'hinge':
            result['data_loss'] = float(np.maximum(0.0, 1.0 - signs * decision).mean())
            result['loss_name'] = 'mean_ovr_binary_hinge_loss'
        else:
            result['data_loss'] = float(np.logaddexp(0.0, -signs * decision).mean())
            result['loss_name'] = 'mean_ovr_binary_logistic_loss'
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(x)
        assert np.isfinite(prob).all()
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, rtol=1e-10, atol=1e-10)
        result['multiclass_log_loss'] = float(log_loss(y, prob, labels=CLASSES))
        if kind in ('mlp', 'nb'):
            result['data_loss'] = result['multiclass_log_loss']
            result['loss_name'] = 'multiclass_log_loss'
    return result, pred


def train_one(identifier, kind, model, xtr, xva, ytr, yva, max_epochs, folder):
    folder.mkdir(parents=True, exist_ok=False)
    if kind == 'nb':
        started = perf_counter()
        model.fit(xtr, ytr)
        duration = perf_counter() - started
        train, _ = measure(model, xtr, ytr, kind)
        val, pred = measure(model, xva, yva, kind)
        print(f'{identifier} | count fit | train_log_loss={train["data_loss"]:.6f} | '
              f'val_log_loss={val["data_loss"]:.6f} | val_macro_f1={val["macro_f1"]:.6f}', flush=True)
        selected_epoch = None
        epochs_run = 0
        stop_reason = 'count_based_fit_no_sgd'
        chosen = model
    else:
        history = []
        best_model, best_loss, best_epoch, stale = None, float('inf'), 0, 0
        started = perf_counter()
        with (folder / 'epoch_log.txt').open('w', encoding='utf-8') as log:
            for epoch in range(1, max_epochs + 1):
                model.partial_fit(xtr, ytr, classes=CLASSES if epoch == 1 else None)
                train, _ = measure(model, xtr, ytr, kind)
                val, _ = measure(model, xva, yva, kind)
                stopping_loss = (val['multiclass_log_loss'] if kind in ('log_loss', 'mlp')
                                 else val['data_loss'])
                row = {'epoch': epoch, 'train_data_loss': train['data_loss'],
                       'val_data_loss': val['data_loss'],
                       'train_multiclass_log_loss': train['multiclass_log_loss'],
                       'val_multiclass_log_loss': val['multiclass_log_loss'],
                       'val_accuracy': val['accuracy'], 'val_macro_f1': val['macro_f1'],
                       'stopping_loss': stopping_loss,
                       'optimizer_loss': float(model.loss_) if kind == 'mlp' else None}
                history.append(row)
                line = (f'{identifier} | Epoch {epoch:03d}/{max_epochs} | '
                        f'train_{train["loss_name"]}={train["data_loss"]:.6f} | '
                        f'val_{val["loss_name"]}={val["data_loss"]:.6f} | '
                        f'val_macro_f1={val["macro_f1"]:.6f}')
                if kind == 'mlp':
                    line += f' | optimizer_loss={model.loss_:.6f}'
                print(line, flush=True)
                log.write(line + '\n')
                log.flush()
                if stopping_loss < best_loss - MIN_DELTA:
                    best_model, best_loss, best_epoch, stale = copy.deepcopy(model), stopping_loss, epoch, 0
                else:
                    stale += 1
                if stale >= PATIENCE:
                    break
        duration = perf_counter() - started
        pd.DataFrame(history).to_csv(folder / 'epoch_history.csv', index=False)
        chosen = best_model
        selected_epoch, epochs_run = best_epoch, len(history)
        stop_reason = 'validation_patience' if stale >= PATIENCE else 'epoch_budget_not_converged'
        train, _ = measure(chosen, xtr, ytr, kind)
        val, pred = measure(chosen, xva, yva, kind)
        print(f'{identifier} | selected={selected_epoch}/{epochs_run} | '
              f'val_macro_f1={val["macro_f1"]:.6f} | stop={stop_reason}', flush=True)
    assert len(pred) == len(yva)
    joblib.dump(chosen, folder / 'model.joblib')
    np.savez_compressed(folder / 'validation_predictions.npz', target=yva, prediction=pred)
    (folder / 'classification_report.txt').write_text(
        classification_report(yva, pred, digits=5, zero_division=0), encoding='utf-8')
    metrics = {'id': identifier, 'algorithm_family': kind,
               'estimator_class': type(chosen).__name__,
               'parameters': chosen.get_params(),
               'train': train, 'validation': val,
               'selected_epoch': selected_epoch, 'epochs_run': epochs_run,
               'stop_reason': stop_reason, 'seconds': duration,
               'model_sha256': sha(folder / 'model.joblib')}
    save_json(folder / 'metrics.json', metrics)
    return {'id': identifier, 'algorithm_family': kind,
            'estimator_class': metrics['estimator_class'],
            'train_accuracy': train['accuracy'],
            'val_accuracy': val['accuracy'],
            'val_macro_f1': val['macro_f1'],
            'train_data_loss': train['data_loss'],
            'val_data_loss': val['data_loss'],
            'loss_name': val['loss_name'],
            'val_multiclass_log_loss': val['multiclass_log_loss'],
            'selected_epoch': selected_epoch, 'epochs_run': epochs_run,
            'stop_reason': stop_reason, 'seconds': duration}


def main(args):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    _, tr, va, xtr, xva, ytr, yva, vectorizer = prepare()
    print('NB cannot use SGD; it is fitted by counts as a reference.', flush=True)
    print('Linear SGD uses one-vs-rest; original multinomial LR and SVC are different estimators.', flush=True)
    print(f'Linear alpha candidates={LINEAR_ALPHAS}; MLP learning rates={MLP_RATES}; '
          f'MLP max_epochs={args.mlp_max_epochs}; linear max_epochs={args.linear_max_epochs}; '
          f'patience={PATIENCE}; min_delta={MIN_DELTA}.', flush=True)
    if args.check_only:
        print('Check only: no model fit, no output written.', flush=True)
        return
    out = HERE / 'sgd_retest' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'protocol.json', {
        'data_sha256': DATA_SHA, 'split_sha256': SPLIT_SHA,
        'seed': SEED, 'tfidf': TFIDF,
        'linear_alpha_grid': LINEAR_ALPHAS, 'mlp_rate_grid': MLP_RATES,
        'mlp_hidden_layer_sizes': [64], 'mlp_alpha': 0.0001,
        'mlp_batch_size': 64, 'mlp_momentum': 0.9,
        'linear_learning_rate': 'optimal',
        'linear_max_epochs': args.linear_max_epochs,
        'mlp_max_epochs': args.mlp_max_epochs,
        'patience': PATIENCE, 'min_delta': MIN_DELTA,
        'ranking_metric': 'validation_macro_f1_at_predeclared_val_loss_checkpoint',
        'test_data_read': False, 'teacher_submission_untouched': True,
        'nb_does_not_use_sgd': True,
        'linear_estimator_differs_from_original_LR_and_SVC': True,
        'script_sha256': sha(Path(__file__)),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__},
    })
    np.savez_compressed(out / 'indices.npz', train_idx=tr, val_idx=va)
    joblib.dump(vectorizer, out / 'tfidf_vectorizer.joblib')
    rows = []
    nb = MultinomialNB(alpha=0.1)
    rows.append(train_one('NB_alpha0.1_count', 'nb', nb, xtr, xva, ytr, yva,
                          0, out / 'NB_alpha0.1_count'))
    pd.DataFrame(rows).to_csv(out / 'partial_results.csv', index=False)
    for kind, loss in (('hinge', 'hinge'), ('log_loss', 'log_loss')):
        for alpha in LINEAR_ALPHAS:
            name = f'{kind}_SGD_alpha{alpha:g}'
            model = SGDClassifier(loss=loss, penalty='l2', alpha=alpha,
                                  learning_rate='optimal', random_state=SEED,
                                  shuffle=True, max_iter=1, tol=None,
                                  early_stopping=False, n_jobs=1)
            rows.append(train_one(name, kind, model, xtr, xva, ytr, yva,
                                  args.linear_max_epochs, out / name))
            pd.DataFrame(rows).to_csv(out / 'partial_results.csv', index=False)
    for rate in MLP_RATES:
        name = f'MLP_SGD_eta{rate:g}'
        model = MLPClassifier(hidden_layer_sizes=(64,), alpha=0.0001,
                              activation='relu', solver='sgd', batch_size=64,
                              learning_rate='constant', learning_rate_init=rate,
                              momentum=0.9, nesterovs_momentum=True,
                              random_state=SEED, shuffle=True, max_iter=1,
                              early_stopping=False)
        rows.append(train_one(name, 'mlp', model, xtr, xva, ytr, yva,
                              args.mlp_max_epochs, out / name))
        pd.DataFrame(rows).to_csv(out / 'partial_results.csv', index=False)
    table = pd.DataFrame(rows).sort_values(['val_macro_f1', 'val_accuracy'], ascending=False)
    table.to_csv(out / 'comparison.csv', index=False)
    save_json(out / 'outcome.json', {
        'selected_by_macro_f1': rows[int(table.index[0])],
        'all_configurations': rows,
        'no_hidden_test_score': True,
        'no_submission_csv_created': True})
    (out / 'COMPLETE.txt').write_text('SGD comparison complete; no test prediction.\n', encoding='utf-8')
    print('\nComplete SGD comparison:', flush=True)
    print(table.to_string(index=False, float_format=lambda v: f'{v:.6f}'), flush=True)
    print('Results:', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--linear-max-epochs', type=int, default=80)
    parser.add_argument('--mlp-max-epochs', type=int, default=120)
    with threadpool_limits(limits=2):
        main(parser.parse_args())
