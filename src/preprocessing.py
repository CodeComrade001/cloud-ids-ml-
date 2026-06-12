# src/preprocessing.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def analyze_dataset(df, target_col):
    print("\n📊 Shape:", df.shape)

    print("\n📊 Class Distribution:")
    print(df[target_col].value_counts())

    print("\n📊 Class Distribution (%):")
    print(df[target_col].value_counts(normalize=True))

    dup = df.duplicated().sum()
    print(f"\n⚠️ Duplicates: {dup}")


def safe_stratify(y):
    if (y.value_counts() < 2).any():
        print("⚠️ Stratify disabled (rare class)")
        return None
    return y


def split_data(
    df,
    target_col,
    test_size=0.30,
    val_size=0.00,
    time_column=None,
    dataset_name=None,
    iot_sample_frac=0.10
):
    if target_col not in df.columns:
        raise ValueError(f"Missing {target_col}")

    df = df.dropna(subset=[target_col])

    if df.empty:
        raise ValueError("Empty dataset")

    # remove target columns from features
    X = df.drop(
        columns=["target", "target_multi", "target_binary"],
        errors="ignore"
    )

    y = df[target_col]

    # ==================================================
    # TIME-BASED SPLIT
    # ==================================================
    if time_column and time_column in df.columns:

        df = df.sort_values(time_column)

        train_end = int(len(df) * (1 - test_size))
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:]

        def split_xy(dataframe):
            X_part = dataframe.drop(
                columns=["target", "target_multi", "target_binary"],
                errors="ignore"
            )
            y_part = dataframe[target_col]
            return X_part, y_part

        X_train, y_train = split_xy(train_df)
        X_test, y_test = split_xy(test_df)

    # ==================================================
    # RANDOM SPLIT
    # ==================================================
    else:

        stratify_y = safe_stratify(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=stratify_y,
            random_state=42
        )

    # ==================================================
    # IoT TRAIN REDUCTION
    # ==================================================
    if dataset_name == "ioT_context":

        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=iot_sample_frac,
            stratify=safe_stratify(y_train),
            random_state=42
        )

    # ==================================================
    # KEEP PIPELINE COMPATIBILITY
    # ==================================================
    X_val = X_train.copy()
    y_val = y_train.copy()

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_val, X_test