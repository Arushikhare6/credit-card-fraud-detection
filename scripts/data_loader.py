import os
import pandas as pd

from config import RAW_DATA_PATH


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at {path}.\n"
            "Download 'creditcard.csv' from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the data/ directory, or pass a custom path "
            "to load_data()."
        )

    df = pd.read_csv(path)

    expected_cols = {"Time", "Amount", "Class"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    return df


def basic_summary(df: pd.DataFrame) -> dict:
    """Return a small dict of dataset-level facts, useful for README/report generation."""
    n_total = len(df)
    n_fraud = int(df["Class"].sum())
    n_legit = n_total - n_fraud

    return {
        "n_transactions": n_total,
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "fraud_pct": round(100 * n_fraud / n_total, 4),
        "imbalance_ratio": round(n_legit / n_fraud, 1) if n_fraud else None,
        "n_features": df.shape[1] - 1,  # excluding target
        "missing_values": int(df.isnull().sum().sum()),
    }


if __name__ == "__main__":
    data = load_data()
    print(basic_summary(data))