from experiments.config import CONFIGS, MODELS
from src.models import PARAM_GRIDS
from src.pipeline import run_pipeline
from experiments.logger import log

# ALL_RESULTS_DIR = "results/summary/all_knn_results.csv"
# ALL_RESULTS_DIR = "results/summary/all_svm_results.csv"
ALL_RESULTS_DIR = "results/summary/all_mlp_results.csv"
# ALL_RESULTS_DIR = "results/summary/all_rf_results.csv"

def run_all():
    results = run_pipeline(CONFIGS, MODELS, PARAM_GRIDS)
    
    results.to_csv(ALL_RESULTS_DIR, index=False)