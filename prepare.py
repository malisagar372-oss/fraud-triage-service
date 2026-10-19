"""prepare.py - Part A: audit, clean and engineer the raw payment export.

    from prepare import clean_transactions, add_features
    python prepare.py      # prints the audit table and the Part A checkpoints
"""
import pandas as pd

RAW_FILE = "transactions_raw.csv"
TEXT_COLS = ["merchant_category", "city"]
AMOUNT_MIN, AMOUNT_MAX = 1, 1_000_000
EXCLUDED_COLS = ["txn_id", "customer_id", "chargeback_filed", "is_fraud"]


def load_raw(path=RAW_FILE):
    return pd.read_csv(path)


def normalise_text(df):
    """Strip surrounding spaces and title-case merchant_category and city (in place on df)."""
    for col in TEXT_COLS:
        df[col] = df[col].astype(str).str.strip().str.title()
    return df


def clean_transactions(raw_df):
    """Return a cleaned copy of raw_df. The input is never modified."""
    df = raw_df.copy()
    df = df.drop_duplicates()                                    # (a)
    df = normalise_text(df)                                      # (b)
    df = df[df["amount_inr"].notna()]                            # (c)
    df = df[(df["amount_inr"] >= AMOUNT_MIN) & (df["amount_inr"] <= AMOUNT_MAX)]  # (d)
    return df.reset_index(drop=True)                             # (e)


def add_features(df):
    """Return a copy with amount_vs_avg_ratio, is_night and high_value added."""
    out = df.copy()
    out["amount_vs_avg_ratio"] = out["amount_inr"] / out["avg_amount_30d_inr"]
    out["is_night"] = (out["hour_of_day"] <= 5).astype(int)
    out["high_value"] = (out["amount_inr"] > 50_000).astype(int)
    return out


def audit(raw):
    dedup = raw.drop_duplicates()
    amt = dedup["amount_inr"]
    return {
        "rows as read": len(raw),
        "columns as read": raw.shape[1],
        "exact duplicate rows": int(raw.duplicated().sum()),
        "rows after de-duplication": len(dedup),
        "missing amount_inr (after de-dup)": int(amt.isna().sum()),
        "missing distance_from_home_km (after de-dup)": int(dedup["distance_from_home_km"].isna().sum()),
        "amount_inr < 1 or > 1,000,000 (after de-dup)": int(((amt < AMOUNT_MIN) | (amt > AMOUNT_MAX)).sum()),
        "distinct raw spellings of merchant_category": raw["merchant_category"].nunique(),
        "distinct raw spellings of city": raw["city"].nunique(),
    }


def main():
    raw = load_raw()
    raw_copy = raw.copy()

    print("=== Part A: audit of transactions_raw.csv ===")
    for k, v in audit(raw).items():
        print(f"{k:<48} {v:>6}")

    clean = clean_transactions(raw)
    feat = add_features(clean)
    print("\n=== Cleaned table ===")
    print("rows:", len(clean), "| duplicate rows:", int(clean.duplicated().sum()))
    print("merchant_category counts:", clean["merchant_category"].value_counts().to_dict())
    print("city counts:", clean["city"].value_counts().to_dict())
    print("payment_mode counts:", clean["payment_mode"].value_counts().to_dict())
    print(f"amount_inr min {clean['amount_inr'].min():.2f}, max {clean['amount_inr'].max():.2f}")
    print("remaining missing distance_from_home_km:", int(clean["distance_from_home_km"].isna().sum()))

    print("\n=== Engineered features ===")
    print("is_night sum:", int(feat["is_night"].sum()))
    print("high_value sum:", int(feat["high_value"].sum()))
    print(f"amount_vs_avg_ratio max: {feat['amount_vs_avg_ratio'].max():.2f}")

    print("\n=== Leakage audit (chargeback_filed) ===")
    print(f"{'chargeback_filed':<18}{'rows':>6}{'fraud':>7}")
    for v in (1, 0):
        sub = clean[clean["chargeback_filed"] == v]
        print(f"{v:<18}{len(sub):>6}{int(sub['is_fraud'].sum()):>7}")
    inputs = [c for c in feat.columns if c not in EXCLUDED_COLS]
    print("excluded columns:", EXCLUDED_COLS)
    print("model-input columns (before one-hot):", inputs)

    print("\n=== Why accuracy misleads ===")
    n_fraud = int(clean["is_fraud"].sum())
    n = len(clean)
    print(f"fraud rows: {n_fraud} of {n} ({n_fraud / n:.2%})")
    print(f"'never fraud' rule: accuracy {(n - n_fraud) / n:.2%}, recall 0")

    print("\n=== Checks ===")
    one = clean_transactions(raw.head(1))
    add_features(one)
    print("one-row DataFrame works:", len(one) == 1)
    print("input DataFrame unchanged:", raw.equals(raw_copy))


if __name__ == "__main__":
    main()
