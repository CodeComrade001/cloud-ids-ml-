CONFIGS = [
    {"name": "baseline", "features": False, "pca": False},
    {"name": "features_only", "features": True, "pca": False},
    {"name": "pca_only", "features": False, "pca": True},
    {"name": "features_pca", "features": True, "pca": True},
]

MODELS = ["KNN"]
# for full testing
# MODELS = ["KNN", "SVM", "MLP", "RF"]