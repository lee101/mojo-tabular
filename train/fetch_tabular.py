import json, hashlib, sys
from pathlib import Path
from sklearn.datasets import fetch_openml
OUT = Path('/nvme0n1-disk/cache/foundry/tabular')
OUT.mkdir(parents=True, exist_ok=True)
SETS = {'adult': 1590, 'credit-g': 31, 'bank-marketing': 1461, 'phoneme': 1489, 'diabetes': 37}
meta = {}
for name, did in SETS.items():
    b = fetch_openml(data_id=did, as_frame=True, parser='auto', data_home=str(OUT / 'openml'))
    df = b.frame
    p = OUT / f'{name}.parquet'
    df.to_parquet(p)
    meta[name] = dict(openml_id=did, rows=len(df), cols=df.shape[1], target=b.target.name,
                      sha256=hashlib.sha256(p.read_bytes()).hexdigest(), license=b.details.get('licence'),
                      version=b.details.get('version'))
    print(name, meta[name], flush=True)
(OUT / 'manifest.json').write_text(json.dumps(meta, indent=2))
