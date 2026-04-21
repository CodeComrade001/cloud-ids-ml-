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

# sequential manual training
# 1. KNN
MODELS = ["KNN"]
# 2. SVM
# MODELS = ["SVM"]
# 3. MLP
# MODELS = ["MLP"]
# 4. RF
# MODELS = ["RF"]

# for full testing
# MODELS = ["KNN", "SVM", "MLP", "RF"]