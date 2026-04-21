# src/preprocess.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(df):
    try:
        # -------------------------
        # 1. SAFETY CHECK (CRITICAL)
        # -------------------------
        if "target" not in df.columns:
            raise ValueError(f"Missing 'target' column. Columns: {df.columns.tolist()}")

        # -------------------------
        # 2. DROP MISSING LABELS
        # -------------------------
        df = df.dropna(subset=["target"])

        # optional debug log
        if df.empty:
            raise ValueError("Dataset is empty after removing NaN targets")

        # -------------------------
        # 3. SPLIT
        # -------------------------
        X = df.drop("target", axis=1)
        y = df["target"]

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    except Exception as e:
        raise RuntimeError(f"Data splitting failed: {e}")

def scale_data(X_train, X_test):
    scaler = StandardScaler()

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test