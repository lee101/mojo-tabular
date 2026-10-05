import argparse, hashlib, json, time
from pathlib import Path
import joblib, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tabular_bench import DATA, SEED, load, prep

TRIALS = 16
GB_TRIALS = 6


def gbdt(rng):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    for _ in range(GB_TRIALS):
        lr = float(10 ** rng.uniform(-2, -0.7)); n = int(rng.integers(150, 600)); d = int(rng.integers(3, 9))
        yield 'xgb', dict(learning_rate=lr, n_estimators=n, max_depth=d), XGBClassifier(
            learning_rate=lr, n_estimators=n, max_depth=d, subsample=0.8, colsample_bytree=0.8, n_jobs=2, random_state=SEED, verbosity=0)
        yield 'lgbm', dict(learning_rate=lr, n_estimators=n, num_leaves=2 ** d), LGBMClassifier(
            learning_rate=lr, n_estimators=n, num_leaves=2 ** d, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, random_state=SEED, verbose=-1)


def hgb(rng):
    return dict(learning_rate=float(10 ** rng.uniform(-2, -0.7)), max_iter=int(rng.integers(100, 600)),
                max_leaf_nodes=int(rng.integers(8, 64)), min_samples_leaf=int(rng.integers(5, 60)),
                l2_regularization=float(10 ** rng.uniform(-3, 1)), max_features=float(rng.uniform(0.5, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--datasets', nargs='*')
    ap.add_argument('--gbdt', action='store_true')
    ap.add_argument('--artifacts', type=Path, required=True)
    a = ap.parse_args()
    man = json.loads((DATA / 'manifest.json').read_text())
    out = dict(kind='tuned_ensemble', seed=SEED, metric='roc_auc', trials=TRIALS,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), datasets={})
    a.artifacts.mkdir(parents=True, exist_ok=True)
    for name, m in man.items():
        if a.datasets and name not in a.datasets:
            continue
        X, y = load(name, m['target'])
        if y is None:
            continue
        idx = np.arange(len(y))
        tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
        va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=SEED)
        rng = np.random.default_rng(SEED)
        t0 = time.monotonic()
        cands = []
        for i in range(TRIALS):
            p = hgb(rng)
            mdl = make_pipeline(prep(X), HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **p)).fit(X.iloc[tr], y[tr])
            cands.append(('hgb', p, mdl, mdl.predict_proba(X.iloc[va])[:, 1]))
        for c in (0.1, 1.0):
            mdl = make_pipeline(prep(X), LogisticRegression(C=c, max_iter=500)).fit(X.iloc[tr], y[tr])
            cands.append(('logreg', dict(C=c), mdl, mdl.predict_proba(X.iloc[va])[:, 1]))
        mdl = make_pipeline(prep(X), ExtraTreesClassifier(300, min_samples_leaf=2, n_jobs=2, random_state=SEED)).fit(X.iloc[tr], y[tr])
        cands.append(('et', {}, mdl, mdl.predict_proba(X.iloc[va])[:, 1]))
        if a.gbdt:
            for kind, p, est in gbdt(np.random.default_rng(SEED + 1)):
                mdl = make_pipeline(prep(X), est).fit(X.iloc[tr], y[tr])
                cands.append((kind, p, mdl, mdl.predict_proba(X.iloc[va])[:, 1]))
        aucs = [roc_auc_score(y[va], c[3]) for c in cands]
        order = np.argsort(aucs)[::-1]
        top = [cands[i] for i in order[:3]]
        ens_va = np.mean([c[3] for c in top], axis=0)
        ens_auc = roc_auc_score(y[va], ens_va)
        single = cands[order[0]]
        use_ens = ens_auc > aucs[order[0]]
        members = top if use_ens else [single]
        pred = lambda i: np.mean([c[2].predict_proba(X.iloc[i])[:, 1] for c in members], axis=0)
        path = a.artifacts / f'{name}.joblib'
        joblib.dump(dict(members=[c[2] for c in members], columns=list(X.columns)), path, compress=3)
        out['datasets'][name] = dict(
            val_auc=round(max(ens_auc, aucs[order[0]]), 4), test_auc=round(roc_auc_score(y[te], pred(te)), 4),
            ensemble=use_ens, best_single_kind=single[0], members=[dict(kind=c[0], params=c[1]) for c in members],
            artifact_bytes=path.stat().st_size, artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            search_s=round(time.monotonic() - t0, 1))
        print(name, out['datasets'][name], flush=True)
    a.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
