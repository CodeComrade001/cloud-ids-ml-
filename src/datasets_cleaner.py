import pandas as pd
import numpy as np

def clean_cybersecurity_intrusion_datasets(path, save_path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset 1: {e}")

    try:
        # 1. Drop identifier
        if "session_id" in df.columns:
            df = df.drop(columns=["session_id"])

        # 2. Force numeric conversion where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Handle missing values safely
        for col in df.columns:

            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric column
                df[col] = df[col].fillna(df[col].median())

            else:
                # Categorical column
                if df[col].mode().empty:
                    df[col] = df[col].fillna("Unknown")
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        # 4. Encode categorical features
        categorical_cols = ["protocol_type", "encryption_used", "browser_type"]
        existing_cols = [col for col in categorical_cols if col in df.columns]

        df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

        # 5. Remove duplicates
        before = df.shape[0]
        df = df.drop_duplicates()
        duplicates_removed = before - df.shape[0]

        # 6. Final validation
        if df.empty:
            raise ValueError("Dataset became empty after cleaning")

        df.to_csv(save_path, index=False)

        return {
            "status": "success",
            "rows": df.shape[0],
            "columns": df.shape[1],
            "duplicates_removed": duplicates_removed
        }

    except Exception as e:
        raise RuntimeError(f"Cleaning dataset 1 failed: {e}")
import pandas as pd

def clean_iot_intrusion_dataset(path, save_path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    try:
        # -------------------------
        # 0. NORMALIZE COLUMN NAMES
        # -------------------------
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # -------------------------
        # 1. VALIDATE LABEL
        # -------------------------
        if "label" not in df.columns:
            raise ValueError(f"Missing 'label' column. Found: {df.columns.tolist()}")

        labels = df["label"].copy()

        # -------------------------
        # 2. DROP USELESS / HIGH-RISK COLUMNS
        # -------------------------
        drop_cols = ["source_ip", "destination_ip"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # -------------------------
        # 3. SEPARATE FEATURES
        # -------------------------
        feature_cols = [col for col in df.columns if col != "label"]

        # -------------------------
        # 4. FORCE NUMERIC CONVERSION
        # -------------------------
        df[feature_cols] = df[feature_cols].apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        )

        # -------------------------
        # 5. DROP BAD COLUMNS (too many NaNs)
        # -------------------------
        threshold = 0.5  # 50% rule (matches your methodology)
        cols_to_drop = [
            col for col in feature_cols
            if df[col].isna().mean() > threshold
        ]

        df = df.drop(columns=cols_to_drop)
        feature_cols = [col for col in df.columns if col != "label"]

        # -------------------------
        # 6. FILL REMAINING NaNs
        # -------------------------
        for col in feature_cols:
            df[col] = df[col].fillna(df[col].median())

        # -------------------------
        # 7. RESTORE LABEL
        # -------------------------
        df["label"] = labels
        df = df[df["label"].notna()]

        # -------------------------
        # 8. REMOVE DUPLICATES
        # -------------------------
        before = len(df)
        df = df.drop_duplicates()
        duplicates_removed = before - len(df)

        # -------------------------
        # 9. FINAL NUMERIC CHECK (CRITICAL)
        # -------------------------
        non_numeric = df.drop(columns=["label"]).select_dtypes(exclude=["int64", "float64"])

        if len(non_numeric.columns) > 0:
            raise ValueError(f"Non-numeric columns remain: {non_numeric.columns.tolist()}")

        # -------------------------
        # 10. FINAL VALIDATION
        # -------------------------
        if df["label"].isna().any():
            raise ValueError("Label column still contains NaN")

        if df.empty:
            raise ValueError("Dataset became empty after cleaning")

        # -------------------------
        # 11. SAVE
        # -------------------------
        df.to_csv(save_path, index=False)

        return {
            "status": "success",
            "rows": df.shape[0],
            "columns": df.shape[1],
            "duplicates_removed": duplicates_removed,
            "dropped_columns_due_to_nan": cols_to_drop
        }

    except Exception as e:
        raise RuntimeError(f"Cleaning failed: {e}")

def clean_network_logs_dataset(path, save_path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    try:
        # 1. Fix column names
        df.columns = df.columns.str.strip().str.replace(" ", "_")

        # 2. Handle missing values safely
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        for col in df.columns:

            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric column
                df[col] = df[col].fillna(df[col].median())

            else:
                if df[col].mode().empty:
                    df[col] = df[col].fillna("Unknown")
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        # 3. Remove mostly-zero columns
        zero_ratio = (df == 0).sum() / len(df)
        cols_to_drop = zero_ratio[zero_ratio > 0.5].index.tolist()
        df = df.drop(columns=cols_to_drop)

        # 4. Remove highly correlated features
        corr_matrix = df.corr(numeric_only=True).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.9)]
        df = df.drop(columns=high_corr_cols)

        # 5. Remove duplicates
        df = df.drop_duplicates()

        # 6. Final validation
        if df.empty:
            raise ValueError("Dataset became empty after cleaning")

        df.to_csv(save_path, index=False)

        return {
            "status": "success",
            "rows": df.shape[0],
            "columns": df.shape[1],
            "dropped_zero_cols": cols_to_drop,
            "dropped_corr_cols": high_corr_cols
        }

    except Exception as e:
        raise RuntimeError(f"Cleaning pipeline failed: {e}")

