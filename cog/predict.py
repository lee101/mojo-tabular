import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODELS = Path(__file__).parent / 'models'
MAX_ROWS = 1000


class Predictor:
    def setup(self):
        self.models = {p.stem: joblib.load(p) for p in sorted(MODELS.glob('*.joblib'))}

    def predict(self, dataset: str, rows: str) -> str:
        if dataset not in self.models:
            raise ValueError(f'dataset must be one of {sorted(self.models)}')
        data = json.loads(rows)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list) or not data or len(data) > MAX_ROWS:
            raise ValueError(f'rows must be 1..{MAX_ROWS} JSON objects')
        bundle = self.models[dataset]
        missing = [c for c in bundle['columns'] if c not in data[0]]
        if missing:
            raise ValueError(f'missing columns: {missing}')
        df = pd.DataFrame(data).reindex(columns=bundle['columns'])
        p = np.mean([m.predict_proba(df)[:, 1] for m in bundle['members']], axis=0)
        return json.dumps([round(float(x), 6) for x in p])
