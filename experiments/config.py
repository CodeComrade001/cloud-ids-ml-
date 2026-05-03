CONFIGS = [
    {"name": "baseline", "features": False, "pca": False},
    {"name": "features_only", "features": True, "pca": False},
    {"name": "pca_only", "features": False, "pca": True},
    {"name": "features_pca", "features": True, "pca": True},
]

REQUIRED_SCHEMA = {
    "features": "numeric",
    "target": "binary (0/1)",
    "no_nan": True
}

# =========================
# 7) REDUCE PARAMETER GRIDS (models.py)
# Example smaller grids
# =========================

PARAM_GRIDS = {
    "KNN": {
        "n_neighbors": [3, 5, 7]
    },

    "SVM": {
        "C": [1, 10],
        "kernel": ["rbf"]
    },

    "MLP": {
        "hidden_layer_sizes": [(50,), (100,)],
        "max_iter": [200]
    },

    "RF": {
        "n_estimators": [50, 100],
        "max_depth": [None, 10]
    }
}

# sequential manual training
# 1. KNN
MODELS = ["KNN"]
# 2. SVM
# MODELS = ["SVM"]
# 3. MLP
# MODELS = ["MLP"]
# 4. RF
# The line `# MODELS = ["RF"]` is a commented-out line in the code. This means that it is not
# currently active or being used in the program. It is likely used as a placeholder or for testing
# purposes. If you want to use the Random Forest (RF) model for training, you can uncomment this line
# by removing the `#` at the beginning and run the program with this configuration.
# MODELS = ["RF"]

# for full testing
# MODELS = ["KNN", "SVM", "MLP", "RF"]