from experiments.config import CONFIGS, MODELS
from src.models import PARAM_GRIDS
from src.pipeline import run_pipeline
from experiments.logger import log

# ==================================================
# SELECT ALL RESULTS FILE (MATCH ACTIVE MODEL)
# ==================================================

# 1. KNN
# ALL_RESULTS_DIR = "results/summary/all_knn_results.csv"

# 2. SVM
# ALL_RESULTS_DIR = "results/summary/all_svm_results.csv"

# 3. MLP
ALL_RESULTS_DIR = "results/summary/all_mlp_results.csv"

# 4. Random Forest
# ALL_RESULTS_DIR = "results/summary/all_rf_results.csv"

# 5. Logistic Regression
# ALL_RESULTS_DIR = "results/summary/all_lr_results.csv"

# 6. Decision Tree
# ALL_RESULTS_DIR = "results/summary/all_dt_results.csv"

# 7. Naive Bayes
# ALL_RESULTS_DIR = "results/summary/all_nb_results.csv"

# 8. Gradient Boosting
# ALL_RESULTS_DIR = "results/summary/all_gb_results.csv"

# 9. Voting Ensemble
# ALL_RESULTS_DIR = "results/summary/all_vote_results.csv"

def run_all():
    results = run_pipeline(CONFIGS, MODELS, PARAM_GRIDS)
    
    results.to_csv(ALL_RESULTS_DIR, index=False)