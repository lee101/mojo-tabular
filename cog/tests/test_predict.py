import json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from predict import Predictor

DATA = Path('/nvme0n1-disk/cache/foundry/tabular')


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Predictor()
        cls.p.setup()

    def test_all_models(self):
        for name, target in [('adult', 'class'), ('credit-g', 'class'), ('bank-marketing', 'Class'),
                             ('phoneme', 'Class'), ('diabetes', 'class')]:
            f = DATA / f'{name}.parquet'
            if not f.exists():
                self.skipTest('dataset cache absent')
            df = pd.read_parquet(f).drop(columns=[target]).head(5)
            rows = df.astype(object).where(df.notna(), None).to_json(orient='records')
            out = json.loads(self.p.predict(name, rows))
            self.assertEqual(len(out), 5)
            self.assertTrue(all(0 <= x <= 1 for x in out))

    def test_errors(self):
        with self.assertRaises(ValueError):
            self.p.predict('nope', '[]')
        with self.assertRaises(ValueError):
            self.p.predict('adult', '[{"age": 1}]')


if __name__ == '__main__':
    unittest.main()
