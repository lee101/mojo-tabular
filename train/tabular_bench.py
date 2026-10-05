import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path('/nvme0n1-disk/cache/foundry/tabular')
SEED = 42


def load(name, target):
    df = pd.read_parquet(DATA / f'{name}.parquet').dropna(subset=[target])
    y = df[target]
    y = (y == sorted(y.astype(str).unique())[-1]).astype(int).to_numpy() if y.nunique() == 2 else None
    X = df.drop(columns=[target])
    return X, y


def prep(X):
    cat = [c for c in X.columns if X[c].dtype.name in ('category', 'object')]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ('n', make_pipeline(__import__('sklearn.impute', fromlist=['x']).SimpleImputer(strategy='median'), StandardScaler()), num),
        ('c', make_pipeline(__import__('sklearn.impute', fromlist=['x']).SimpleImputer(strategy='most_frequent'),
                            OneHotEncoder(handle_unknown='ignore', min_frequency=5, sparse_output=False)), cat)])


def zoo(X):
    return {
        'prior': lambda: DummyClassifier(strategy='prior'),
        'logreg': lambda: make_pipeline(prep(X), LogisticRegression(max_iter=300)),
        'rf': lambda: make_pipeline(prep(X), RandomForestClassifier(100, n_jobs=2, random_state=SEED)),
        'hgb': lambda: make_pipeline(prep(X), HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)),
        'mlp': lambda: make_pipeline(prep(X), MLPClassifier((128, 64), early_stopping=True, max_iter=60, random_state=SEED)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    manifest = json.loads((DATA / 'manifest.json').read_text())
    out = dict(kind='real_data_baselines', seed=SEED, split='60/20/20 stratified', metric='roc_auc',
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), datasets={})
    for name, m in manifest.items():
        X, y = load(name, m['target'])
        if y is None:
            continue
        idx = np.arange(len(y))
        tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
        va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=SEED)
        res = {}
        for k, mk in zoo(X).items():
            t = time.monotonic()
            mdl = mk().fit(X.iloc[tr], y[tr])
            fit = time.monotonic() - t
            p = lambda i: mdl.predict_proba(X.iloc[i])[:, 1]
            res[k] = dict(val_auc=round(roc_auc_score(y[va], p(va)), 4),
                          test_auc=round(roc_auc_score(y[te], p(te)), 4), fit_s=round(fit, 2))
            print(name, k, res[k], flush=True)
        best = max((k for k in res if k != 'prior'), key=lambda k: res[k]['val_auc'])
        out['datasets'][name] = dict(rows=len(y), sha256=m['sha256'], results=res, selected_by_val=best)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
