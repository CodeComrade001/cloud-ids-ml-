from experiments.config import CONFIGS, MODELS
from src.pipeline import run_pipeline
from experiments.logger import log

def run_all():
    # for config in CONFIGS:
    #     for model in MODELS:
    #         print(f"Running {model} with {config['name']}")

            results = run_pipeline("KVN", "none")

            log("KVN", "none", results)