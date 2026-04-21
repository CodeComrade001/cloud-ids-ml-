from experiments.config import CONFIGS, MODELS
from src.datasets_cleaner import clean_iot_intrusion_dataset
from src.pipeline import run_pipeline
from experiments.logger import log

def run_all():
    results = run_pipeline(CONFIGS, MODELS)
    # log(results, model=MODELS, metrics=results)
    