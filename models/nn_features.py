"""Feature preparation for the neural members (v35 RealMLP, v36 TabM).

`feature_engineering` and `NumericalPreprocessor` below are copied VERBATIM out of
v35_realmlp.py (yekenot's notebook code, kernel ps-s6-e9-realmlp-pytorch) so that v36
sees exactly the inputs v35 sees. v35 runs its whole pipeline at import time and cannot
be imported as a module, hence this file; v35 itself is untouched. If the notebook is
re-pulled and v35 regenerated, re-copy the two blocks below (v35 lines 63-169 and
183-216) and `load_frames` (v35 lines 34-56 + 171-176).
"""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

ID = 'id'
TARGET = 'Will_Buy_EV'
TFMS = ["median_center", "robust_scale", 'smooth_clip']   # v35's CONFIG["tfms"]

# module-level state the verbatim block reads (v35 has these as script globals)
orig = None
cat_cols = []
num_cols = []
category_map = {}
important_combos = [
    ('Annual_Income_USD', 'Range_Anxiety_Level'),
    ('Age', 'Range_Anxiety_Level'),
    ('Annual_Income_USD', 'Income_/_100_floor_'),
]

# ── verbatim from v35_realmlp.py lines 63-169 ─────────────────────────────────
def feature_engineering(df, fit=False):
    # Fill NaNs
    for col in cat_cols:
        df[col] = df[col].fillna("missing")
    for col in num_cols:
        df[col] = df[col].fillna(0.0)

    # Categorize string cats
    for col in cat_cols:
        if fit:
            codes, uniques = df[col].factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = df[col].map(code_map).fillna(-1).astype('int32')
        df[col] = codes
        df[col] = df[col].astype('category')

    # Arithmetic interaction
    df['_Daily_Commute_km_/_Age'] = (df['Daily_Commute_km'] / (df['Age'] + 1e-6)).astype('float32')
    df['Income_/_100_floor_']  = np.floor(df['Annual_Income_USD'] / 100.0).astype('float32').astype('category')
    df['Income_/_1000_floor_']  = np.floor(df['Annual_Income_USD'] / 1000.0).astype('float32').astype('category')
    df['Income_/_10000_floor_']  = np.floor(df['Annual_Income_USD'] / 10000.0).astype('float32').astype('category')
    df['Daily_km_/_5_floor_']  = np.floor(df['Daily_Commute_km'] / 5.0).astype('float32').astype('category')

    # Categorize numericals
    for col in [i for i in num_cols if i not in ['Annual_Income_USD', 'Charging_Stations_Near_Home']]:
        cat_name = f"{col}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype('category')

    # Digit extraction
    for col in ['Daily_Commute_km']:
        decimal_name = f"_{col}_decimal"
        df[decimal_name] = (df[col] % 1).round(2).astype('float32')
    df['Annual_Income_USD_is_multiple_10_'] = (np.floor(df['Annual_Income_USD']) % 10 == 0).astype('category')

    # Target encoding from orig
    for col in ['Annual_Income_USD']:
        orig_enc_name = f"_{col}_mean_target_orig"
        df[orig_enc_name] = (
            df[col]
            .map(orig.groupby(col)[TARGET].mean())
            .fillna(orig[TARGET].mean())
            .astype('float32')
        )

    # Count encoding
    for col in ['Annual_Income_USD']:
        count_name = f"_{col}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].astype(object).map(count_map).fillna(0).astype('int32')

    # Discretize numericals
    bin_config = {'Annual_Income_USD': [400, 600, 800, 900, 1100]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ['quantile']:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode='ordinal',
                        strategy=strategy,
                        subsample=None
                    )
                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned
                df[bin_name] = df[bin_name].astype('category')

    # Create interaction categories
    combo_names = []
    for cols in important_combos:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype('category')   

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]
    return df, new_cat_cols, new_num_cols, combo_names

# ── verbatim from v35_realmlp.py lines 183-216 ────────────────────────────────
class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Applies a configurable sequence of numerical transforms from CONFIG["tfms"].
    Supported: 'median_center', 'robust_scale', 'smooth_clip', 'l2_normalize'.
    'one_hot' and 'embedding' are recognised but skipped (handled by the model).
    """

    def __init__(self, tfms):
        self._tfms = [t for t in tfms
                      if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]

    def fit(self, X: np.ndarray, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X

# ── our wrappers ──────────────────────────────────────────────────────────────
def load_frames(smoke_rows=None, smoke_test_rows=5000):
    """v35 lines 34-56 + 171-176, as a function. Returns X, y, X_test, test_id, cat_cols,
    num_cols, combo_names — the exact frames v35's fold loop iterates over."""
    global orig, cat_cols, num_cols, category_map
    category_map = {}
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/orig/EV_Adoption_and_Range_Anxiety_Dataset.csv")
    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1})
    orig[TARGET] = orig[TARGET].map({'No': 0, 'Yes': 1})
    X = train.drop([ID, TARGET], axis=1)
    y = train[TARGET]
    X_test = test.drop([ID], axis=1); test_id = test[ID]
    del train, test
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    if smoke_rows:
        X = X.iloc[:smoke_rows].reset_index(drop=True); y = y.iloc[:smoke_rows].reset_index(drop=True)
        X_test = X_test.iloc[:smoke_test_rows].reset_index(drop=True); test_id = test_id.iloc[:smoke_test_rows]
    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
    X_test, _, _, _ = feature_engineering(X_test, fit=False)
    cat_cols = cat_cols + new_cat_cols; num_cols = num_cols + new_num_cols
    return X, y, X_test, test_id, cat_cols, num_cols, combo_names

def fold_te(X_tr, X_val, X_tst, y_tr, combo_names, seed):
    """v35's per-fold nested target encoding of the combo keys (its fold loop, ~line 880):
    fit rows inner-CV'd, validation and test encoded from the whole fit set. In place."""
    enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=seed)
    te_names = [f"_{c}TE" for c in combo_names]
    X_tr[te_names] = enc.fit_transform(X_tr[combo_names], y_tr)
    X_val[te_names] = enc.transform(X_val[combo_names])
    X_tst[te_names] = enc.transform(X_tst[combo_names])
    return te_names
