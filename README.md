---
license: mit
library_name: scikit-learn
pipeline_tag: tabular-classification
tags: [tabular, cpu, tiny, cog, app-nz]
---

# mojo-tabular

Tiny CPU tabular classifiers (one per dataset), total 12 MB, no GPU. Trained with scikit-learn;
`mojo-` is the planned optimization path, not a Mojo implementation yet.

## Results (ROC-AUC, seed 42, 60/20/20 stratified split, selection on validation only)

| dataset (OpenML) | val | test | untuned best baseline test | MB |
|---|---|---|---|---|
| adult | 0.929 | 0.9282 | 0.9272 | 2.64 |
| credit-g | 0.8294 | 0.7986 | 0.7958 | 0.00 |
| bank-marketing | 0.9354 | 0.9384 | 0.9375 | 2.67 |
| phoneme | 0.9523 | 0.9546 | 0.9554 | 5.50 |
| diabetes | 0.8528 | 0.8001 | 0.8031 | 1.09 |

Search: 16 random HistGradientBoosting configs + logistic regression + extra trees; top-3 averaged
when validation improves. Single split, single seed: small sets (credit-g n=1000, diabetes n=768) have
large validation/test gaps and are noisy. No comparison against XGBoost/CatBoost/TabPFN has been run,
so **no frontier claim is made**. Adding tuned XGBoost 3.4.1 and LightGBM 4.7.0 (6 configs each) to the pool changed test AUC by at most +0.001 (adult 0.929, bank-marketing 0.9386), i.e. these results sit at the plateau of strong boosted trees; the shipped models use sklearn only to keep the image small. CatBoost and TabPFN were not compared (CatBoost hung under the 4 GiB address-space limit). Data: public OpenML sets (license listed as Public).
Reproduce: `train/fetch_tabular.py`, `train/tabular_tune.py` (`--gbdt` needs xgboost/lightgbm), artifact hashes in `results/tuned.json`.

## Use

```python
import joblib, pandas as pd
b = joblib.load("models/adult.joblib")  # needs scikit-learn==1.8.0
p = sum(m.predict_proba(df[b["columns"]])[:, 1] for m in b["members"]) / len(b["members"])
```

## Deploy on app.nz (Cog, CPU)

```bash
cd cog && python3 -m unittest discover -s tests          # contract test, no network
docker build -t ghcr.io/lee101/mojo-tabular:latest .     # 381 MB (python:3.12-slim), weights baked
docker run --rm -p 5000:5000 --memory 512m ghcr.io/lee101/mojo-tabular:latest   # ~80 MB RSS, ready in <10 s
```

[![Deploy to app.nz](https://app.nz/deploy-button.svg)](https://app.nz/deploy?image=ghcr.io/lee101/mojo-tabular:latest&name=mojo-tabular&hardware=cpu&idleSeconds=60)

`POST /predictions {"input": {"dataset": "adult", "rows": "[{...}]"}}` returns a JSON list of probabilities
(max 1000 rows). Do not use the default `cog build` image: its apt layer alone is 1.78 GB (2.33 GB total).
Links: GitHub `lee101/mojo-tabular`, Hugging Face `lee101/mojo-tabular`.

## Limitations

Trained on public benchmark data only; not validated for credit, hiring or medical decisions.
