import os
import json
import pandas as pd
from datetime import datetime

class ExperimentLogger:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def create_run(self, dataset, config, model):
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(self.base_dir, dataset, config, model, run_id)
        os.makedirs(path, exist_ok=True)
        return path

    def save_metrics(self, path, metrics):
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def save_predictions(self, path, y_true, y_pred):
        df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred
        })
        df.to_csv(os.path.join(path, "predictions.csv"), index=False)

    def save_config(self, path, config_data):
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_data, f, indent=4)