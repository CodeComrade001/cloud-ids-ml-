def preprocess(df):
    # handle missing, duplicates
    return df

def split_data(df):
    from sklearn.model_selection import train_test_split
    X = df.drop("label", axis=1)
    y = df["label"]
    return train_test_split(X, y, test_size=0.3, stratify=y)