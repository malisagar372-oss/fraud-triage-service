"""train.py - Part B: fraud model, class imbalance and alert threshold.

    from train import build_final_model
    python train.py        # prints every Part B table
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare import EXCLUDED_COLS, add_features, clean_transactions, load_raw

SEED = 42
TARGET = "is_fraud"
THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]
MIN_RECALL = 0.65
PARAM_GRID = {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}


# ---------------------------------------------------------------- data
def load_features():
    """Clean + engineer, return (X one-hot encoded, y, full table)."""
    df = add_features(clean_transactions(load_raw()))
    y = df[TARGET].astype(int)
    X = pd.get_dummies(df.drop(columns=EXCLUDED_COLS), dtype=int)
    return X, y, df


def split(X, y):
    """Stratified 60/20/20 split with two successive calls."""
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def fill_median(median, *frames):
    """Fill distance_from_home_km with the training median (copies, no in-place change)."""
    return [f.assign(distance_from_home_km=f["distance_from_home_km"].fillna(median)) for f in frames]


def prepared_splits():
    X, y, _ = load_features()
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(X, y)
    median = float(X_tr["distance_from_home_km"].median())    # learned on training rows only
    X_tr, X_va, X_te = fill_median(median, X_tr, X_va, X_te)
    return X, X_tr, X_va, X_te, y_tr, y_va, y_te, median


# ---------------------------------------------------------------- models
def lr_pipeline(class_weight=None):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight))])


def rf(**params):
    return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=SEED, **params)


def cv_folds():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def grid_search(X_tr, y_tr):
    gs = GridSearchCV(rf(), PARAM_GRID, scoring="f1", cv=cv_folds(), n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs


def oversample(X_tr, y_tr):
    """Random over-sampling of training fraud rows up to the non-fraud count."""
    tr = X_tr.assign(**{TARGET: y_tr})
    legit = tr[tr[TARGET] == 0]
    fraud = tr[tr[TARGET] == 1].sample(n=len(legit), replace=True, random_state=SEED)
    big = pd.concat([legit, fraud])
    return big.drop(columns=TARGET), big[TARGET]


def threshold_table(y_true, proba):
    rows = []
    for t in THRESHOLDS:
        pred = (proba >= t).astype(int)
        rows.append({"threshold": t,
                     "precision": precision_score(y_true, pred, zero_division=0),
                     "recall": recall_score(y_true, pred),
                     "f1": f1_score(y_true, pred),
                     "flagged": int(pred.sum())})
    return pd.DataFrame(rows)


def choose_threshold(table):
    ok = table[table["recall"] >= MIN_RECALL]
    return float(ok["threshold"].max()) if len(ok) else 0.1


def metrics(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "flagged": int(pred.sum())}


# ---------------------------------------------------------------- builder
def build_final_model():
    """Load, clean, engineer, encode, split, median-fill, grid-search; return the final model."""
    X, X_tr, X_va, X_te, y_tr, y_va, y_te, median = prepared_splits()
    model = grid_search(X_tr, y_tr).best_estimator_
    proba_va = model.predict_proba(X_va)[:, 1]
    t_star = choose_threshold(threshold_table(y_va, proba_va))
    return {"model": model, "columns": list(X.columns), "median": median, "threshold": t_star}


# ---------------------------------------------------------------- report
def fmt(d):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def main():
    pd.set_option("display.width", 160)
    X, X_tr, X_va, X_te, y_tr, y_va, y_te, median = prepared_splits()

    print("=== Input columns (26, pinned order) ===")
    print(len(X.columns), list(X.columns))

    print("\n=== Split ===")
    print(f"{'split':<12}{'rows':>6}{'fraud':>7}")
    for name, yy in (("train", y_tr), ("validation", y_va), ("test", y_te)):
        print(f"{name:<12}{len(yy):>6}{int(yy.sum()):>7}")
    print("training median of distance_from_home_km:", median)

    print("\n=== Imbalance treatments: logistic regression, validation, threshold 0.5 ===")
    X_os, y_os = oversample(X_tr, y_tr)
    variants = {"plain": (lr_pipeline(), X_tr, y_tr),
                "balanced": (lr_pipeline("balanced"), X_tr, y_tr),
                "over-sampled": (lr_pipeline(), X_os, y_os)}
    rows = []
    for name, (pipe, Xf, yf) in variants.items():
        pipe.fit(Xf, yf)
        rows.append({"variant": name, **fmt(metrics(y_va, pipe.predict(X_va)))})
    print(pd.DataFrame(rows).drop(columns="flagged").to_string(index=False))
    print(f"over-sampled training table: {len(X_os)} rows "
          f"({int((y_os == 0).sum())} non-fraud + {int((y_os == 1).sum())} fraud)")

    print("\n=== 5-fold CV on the training split (F1) ===")
    cv = cv_folds()
    for name, est in (("balanced logistic regression", lr_pipeline("balanced")),
                      ("random forest (untuned, balanced)", rf())):
        s = cross_val_score(est, X_tr, y_tr, scoring="f1", cv=cv, n_jobs=-1)
        print(f"{name:<36} mean {s.mean():.3f}  std {s.std():.3f}  folds {np.round(s, 3).tolist()}")
    gs = grid_search(X_tr, y_tr)
    best_i = gs.best_index_
    print(f"grid search: {len(gs.cv_results_['params'])} combinations x 5 folds")
    print(f"best params: {gs.best_params_} (n_estimators=200 fixed)")
    print(f"best mean CV F1: {gs.best_score_:.3f}  std {gs.cv_results_['std_test_score'][best_i]:.3f}")
    model = gs.best_estimator_

    print("\n=== Threshold table, tuned forest, validation split ===")
    proba_va = model.predict_proba(X_va)[:, 1]
    table = threshold_table(y_va, proba_va)
    print(table.round(3).to_string(index=False))
    t_star = choose_threshold(table)
    print(f"t_star = {t_star} (highest threshold with validation recall >= {MIN_RECALL})")

    print(f"\n=== Final test at t_star = {t_star} (used once) ===")
    m = metrics(y_te, (model.predict_proba(X_te)[:, 1] >= t_star).astype(int))
    print(fmt(m))
    print(f"test payments: {len(y_te)}, flagged: {m['flagged']}, "
          f"legitimate held per fraud caught: {m['FP'] / m['TP']:.2f}, frauds missed: {m['FN']}")

    print("\n=== Leakage experiment: validation F1 at 0.5 ===")
    clean_f1 = f1_score(y_va, (proba_va >= 0.5).astype(int))
    _, _, df = load_features()
    leak_tr = X_tr.assign(chargeback_filed=df.loc[X_tr.index, "chargeback_filed"])
    leak_va = X_va.assign(chargeback_filed=df.loc[X_va.index, "chargeback_filed"])
    leaky = rf(**gs.best_params_).fit(leak_tr, y_tr)
    leaky_f1 = f1_score(y_va, (leaky.predict_proba(leak_va)[:, 1] >= 0.5).astype(int))
    print(f"clean tuned forest (26 cols): {clean_f1:.3f}")
    print(f"leaky forest (27 cols, chargeback_filed last): {leaky_f1:.3f}")
    print(f"gap: {leaky_f1 - clean_f1:+.3f}")

    print("\n=== Top 5 feature importances (tuned forest) ===")
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    for name, v in imp.head(5).items():
        print(f"{name:<26}{v:.3f}")


if __name__ == "__main__":
    main()
