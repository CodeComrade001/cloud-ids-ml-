import pandas as pd
import numpy as np


import pandas as pd
import numpy as np

def clean_dataset_1(path, save_path):
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

def clean_dataset_2(path, save_path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset 2: {e}")

    try:
        # 1. Drop IP addresses (high cardinality, no generalization)
        drop_cols = ["Source_IP", "Destination_IP"]
        existing_drop = [col for col in drop_cols if col in df.columns]
        df = df.drop(columns=existing_drop)

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

        # 3. Encode categorical features
        categorical_cols = ["Request_Type", "Protocol", "User_Agent", "Status", "Scan_Type"]
        existing_cols = [col for col in categorical_cols if col in df.columns]

        df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

        # 4. Remove duplicates
        before = df.shape[0]
        df = df.drop_duplicates()
        duplicates_removed = before - df.shape[0]

        # 5. Validate dataset
        if df.empty:
            raise ValueError("Dataset 2 became empty after cleaning")

        df.to_csv(save_path, index=False)

        return {
            "status": "success",
            "rows": df.shape[0],
            "columns": df.shape[1],
            "dropped_columns": existing_drop,
            "duplicates_removed": duplicates_removed
        }

    except Exception as e:
        raise RuntimeError(f"Cleaning dataset 2 failed: {e}")

def clean_dataset_3(path, save_path):
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


def load_data(path):
    return pd.read_csv(path)