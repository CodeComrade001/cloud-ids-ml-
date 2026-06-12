import pandas as pd
import os

def log(config, model, metrics):
    os.makedirs("results/summary", exist_ok=True)

    row = {
        "config": config,
        "model": model,
        **metrics
    }

    df = pd.DataFrame([row])

    file = "results/summary/all_results.csv"

    if os.path.exists(file):
        df.to_csv(file, mode="a", header=False, index=False)
    else:
        df.to_csv(file, index=False)