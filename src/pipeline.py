import numpy as np
np.random.seed(42)
import pandas as pd
import os
from experiments.ExperimentLogger import ExperimentLogger
from src.loader import load_data, standardize_target, validate_dataset
from src.preprocessing import split_data, scale_data
from src.feature_engineering import FeatureEngineer
from src.dimensionality import apply_pca
from src.models import PARAM_GRIDS, get_model
from src.trainer import train
from src.evaluator import evaluate




DATASETS = {
    "baseline_dataset": "data/cleaned/cybersecurity_intrusion_data_cleaned.csv",
    "network_logs_dataset": "data/cleaned/Network_logs_cleaned.csv",
    "ioT_context": "data/cleaned/IoT_Intrusion_cleaned.csv"
    # "ioT_context": "data/raw/IoT_Intrusion.csv"
}

RESULTS_FILE = "results/summary/KNN_training_results.csv"


def run_pipeline(configurations, model_names):
    try:
        logger = ExperimentLogger("results")
        results = []
        for dataset_name, dataset_path in DATASETS.items():

            print(f"\n=== Processing {dataset_name} ===")

            # -------------------------
            # 1. LOAD DATA
            # -------------------------
            try:
                df = load_data(dataset_path)
                df = standardize_target(df)
                df = validate_dataset(df)
            except Exception as e:
                print(f"❌ Skipping {dataset_name}: {e}")
                continue
            
              # -------------------------
            # 2. SPILT DATA
            # -------------------------
            X_train, X_test, y_train, y_test = split_data(df)

            # -------------------------
            # 3. SCALE DATA
            # -------------------------
            X_train, X_test = scale_data(X_train, X_test)

            for config in configurations:

                print(f"  -> Config: {config['name']}")
                if y_train.isna().any():
                    print(f"❌ Invalid labels in {dataset_name}")
                    
                print("Class distribution:\n", y_train.value_counts())

                if y_train.nunique() < 2:
                    raise ValueError(f"{dataset_name} has only one class")

                if X_train.isna().any().any():
                    raise ValueError(f"{dataset_name} has NaN in features")

                # IMPORTANT: isolate each config run
                X_tr, X_te = X_train.copy(), X_test.copy()

                # -------------------------
                # 2. FEATURE ENGINEERING
                # -------------------------
                if config["features"]:
                    fe = FeatureEngineer()
                    X_tr = fe.fit_transform(X_tr)
                    X_te = fe.transform(X_te)

                # -------------------------
                # 3. DIMENSIONALITY REDUCTION
                # -------------------------
                if config["pca"]:
                    X_tr, X_te = apply_pca(X_tr, X_te)

                for model_name in model_names:

                    print(f"    -> Model: {model_name} training started...")

                    # -------------------------
                    # 4. MODEL CREATION
                    # -------------------------
                    model = get_model(model_name)

                    print(f"    -> model fetch/ get_model() found : {model}")
                    # -------------------------
                    # 5. TRAINING
                    # -------------------------
                    params = PARAM_GRIDS.get(model_name, {})
                    print(f"    -> Param fetching/ PARAM_GRIDS() found : {params}")
                    
                    trained_model = train(model, params, X_tr, y_train)
                    print(f"    -> Model training/ trained_model() found : {trained_model}")

                    # -------------------------
                    # 6. PREDICTION
                    # -------------------------
                    y_pred = trained_model.predict(X_te)
                    print(f"    ->model prediction/ Prediction/ y_pred() found : {y_pred[:5]}")  # print first 5 predictions for sanity check`

                    # -------------------------
                    # 7. EVALUATION
                    # -------------------------
                    metrics = evaluate(trained_model, X_te, y_test)
                    print(f"    ->model Evaluation/ metrics() found : {metrics}")

                    # -------------------------
                    # 8. LOGGING (CRITICAL)
                    # -------------------------
                    run_path = logger.create_run(dataset_name, config["name"], model_name)

                    logger.save_metrics(run_path, metrics)
                    logger.save_predictions(run_path, y_test, y_pred)
                    logger.save_config(run_path, config)
                    print(f"    ->Result logging/ Logging completed for {model_name} on {dataset_name} with config {config['name']}")
                    
                    # -------------------------
                    # 9. STORE IN MEMORY SUMMARY
                    # -------------------------
                    results.append({
                        "dataset": dataset_name,
                        "config": config["name"],
                        "model": model_name,
                        **metrics
                    })
                    print(f"    ->Result summary/ Stored in-memory results for {model_name} on {dataset_name} with config {config['name']}")

        # -------------------------
        # FINAL SUMMARY EXPORT
        # -------------------------
        df_results = pd.DataFrame(results)
        print(f"    -> results: {results}")
        
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        
        df_results.to_csv(RESULTS_FILE, index=False)
        print(f"    -> Final results exported to {RESULTS_FILE}")

        return df_results
    except Exception as e:
        raise RuntimeError(f"Pipeline execution failed: {e}")