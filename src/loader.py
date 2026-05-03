# src/loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Identify label column
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
        raise ValueError("Missing 'target' column")

    # Clean target
    df["target"] = df["target"].astype(str).str.strip().str.lower()

    # Multi-class encoding
    le = LabelEncoder()
    df["target_multi"] = le.fit_transform(df["target"])

    classes = list(le.classes_)
    encoded = np.asarray(le.transform(le.classes_)).tolist()
    label_mapping = dict(zip(classes, encoded))

    print("\n🎯 Label Mapping:")
    for k, v in label_mapping.items():
        print(f"{k} → {v}")

    # Binary target (SAFE VERSION)
    unique_labels = df["target"].unique()

    if len(unique_labels) == 2:
        print("✅ Binary classification detected")
        df["target_binary"] = df["target_multi"]

    else:
        print(f"⚠️ Multi-class detected ({len(unique_labels)} classes)")

        # Safer binary mapping
        normal_keywords = ["normal", "benign"]

        def map_binary(x):
            x = str(x).lower()
            return 0 if any(k in x for k in normal_keywords) else 1

        df["target_binary"] = df["target"].apply(map_binary)

        # Safety check
        if df["target_binary"].nunique() < 2:
            raise ValueError("Binary mapping failed: only one class detected")

    return df, label_mapping


def validate_dataset(df):
    if "target" not in df.columns:
        raise ValueError("Missing target column")

    if df["target"].isna().any():
        raise ValueError("NaN in target")

    if df.empty:
        raise ValueError("Empty dataset")

    return df