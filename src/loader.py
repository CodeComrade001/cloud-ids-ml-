# src/loader.py
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # 🔴 Normalize column names FIRST
    df.columns = df.columns.str.strip().str.lower()

    if "intrusion" in df.columns:
        df = df.rename(columns={"intrusion": "target"})
    elif "attack_detected" in df.columns:
        df = df.rename(columns={"attack_detected": "target"})
    elif "label" in df.columns:
        df = df.rename(columns={"label": "target"})
    else:
        raise ValueError(f"No valid label column found: {df.columns.tolist()}")

    return df

def standardize_target(df):
    if "target" not in df.columns:
        raise ValueError("Target column missing after load_data")

    def map_target(x):
        val = str(x).lower()
        if val in ["0", "normal", "benign"]:
            return 0
        return 1

    df["target"] = df["target"].apply(map_target)

    return df

def validate_dataset(df):
    assert "target" in df.columns

    if df["target"].isna().any():
        raise ValueError("NaN in target")

    if df["target"].nunique() != 2:
        raise ValueError("Target must be binary")

    if df.empty:
        raise ValueError("Empty dataset")

    return df

