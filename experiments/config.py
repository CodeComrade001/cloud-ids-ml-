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


# ==================================================
# SEQUENTIAL MANUAL TRAINING (BEST FOR LOW RAM PC)
# Run ONE model at a time
# ==================================================

# 1. KNN :::::::::::======= true
MODELS = ["KNN"]

# 2. SVM :::::::::::======= true
# MODELS = ["SVM"]

# 3. MLP :::::::::::======= true
# MODELS = ["MLP"]

# 4. Random Forest :::::::::::======= true
# MODELS = ["RF"]

# 5. Logistic Regression
# MODELS = ["LR"]

# 6. Decision Tree :::::::::::======= true
# MODELS = ["DT"]

# 7. Naive Bayes  :::::::::::======= true
# MODELS = ["NB"]

# 8. Gradient Boosting   :::::::::::======= true
# MODELS = ["GB"]

# 9. Voting Ensemble
# Uses multiple models internally, so heavier than single models
# MODELS = ["VOTE"]


# ==================================================
# LIGHT BATCH TESTING (2 at a time recommended)
# ==================================================

# MODELS = ["LR", "NB"]
# MODELS = ["DT", "RF"]
# MODELS = ["GB", "MLP"]
# MODELS = ["KNN", "SVM"]


# ==================================================
# FULL TESTING (ONLY IF MACHINE CAN HANDLE IT)
# ==================================================

# MODELS = [
#     "KNN",
#     "SVM",
#     "MLP",
#     "RF",
#     "LR",
#     "DT",
#     "NB",
#     "GB",
#     "VOTE"
# ]