# ==========================================================
# CLEANED MAIN PIPELINE
# - Dynamic target handling per dataset
# - Cleaner logs
# - Graceful dataset skipping
# - Correct PCA split usage
# - Safer evaluation flow
# ==========================================================

import os
import time
import numpy as np
import pandas as pd

from src.model_handler import MODEL_HANDLERS, BaseModelHandler

np.random.seed(42)

from experiments.ExperimentLogger import ExperimentLogger
from src.loader import load_data, standardize_target, validate_dataset
from src.preprocessing import analyze_dataset, split_data, scale_data
from src.feature_engineering import FeatureEngineer
from src.dimensionality import apply_pca
from sklearn.model_selection import train_test_split
from src.models import  get_model
from src.evaluator import evaluate


# ==========================================================
# DATASETS
# ==========================================================
DATASETS = {
    "baseline_dataset": "data/cleaned/cybersecurity_intrusion_data_cleaned.csv",
    "network_logs_dataset": "data/cleaned/Network_logs_cleaned.csv",
    "ioT_context": "data/cleaned/IoT_Intrusion_cleaned.csv"
}

# ==================================================
# SELECT RESULT FILE (MATCH ACTIVE MODEL)
# ==================================================

# 1. KNN
RESULTS_FILE = "results/summary/KNN_training_results.csv"

# 2. SVM
# RESULTS_FILE = "results/summary/SVM_training_results.csv"

# 3. MLP
# RESULTS_FILE = "results/summary/MLP_training_results.csv"

# 4. Random Forest
# RESULTS_FILE = "results/summary/RF_training_results.csv"

# 5. Logistic Regression
# RESULTS_FILE = "results/summary/LR_training_results.csv"

# 6. Decision Tree
# RESULTS_FILE = "results/summary/DT_training_results.csv"

# 7. Naive Bayes
# RESULTS_FILE = "results/summary/NB_training_results.csv"

# 8. Gradient Boosting
# RESULTS_FILE = "results/summary/GB_training_results.csv"

# 9. Voting Ensemble
# RESULTS_FILE = "results/summary/VOTE_training_results.csv"


# ==========================================================
# HELPERS
# ==========================================================
def line():
    print("=" * 70)


def section(title):
    print(f"\n🔹 {title}")

# ==========================================================
# MAIN PIPELINE
# ==========================================================
def run_pipeline(configurations, model_names, parameter_grids):
    full_training_start_time = time.perf_counter()

    logger = ExperimentLogger("results")
    results = []

    for dataset_name, dataset_path in DATASETS.items():

        line()
        print(f"📁 DATASET: {dataset_name}")
        line()

        # --------------------------------------------------
        # LOAD DATASET
        # --------------------------------------------------
        try:
            section("Loading dataset")

            df = load_data(dataset_path)
            df, label_mapping = standardize_target(df)
            df = validate_dataset(df)

            print(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        except Exception as e:
            print(f"❌ Skipped {dataset_name}: {e}")
            continue

        # --------------------------------------------------
        # TARGET SELECTION
        # --------------------------------------------------
        try:
            section("Selecting target")

            if "target_multi" in df.columns and df["target_multi"].nunique() > 2:
                target_col = "target_multi"

            elif "target_binary" in df.columns:
                target_col = "target_binary"

            elif "target" in df.columns:
                target_col = "target"

            else:
                raise ValueError("No usable target column found")

            print(f"✅ Target selected: {target_col}")
            print(f"✅ Classes: {df[target_col].nunique()}")

        except Exception as e:
            print(f"❌ Skipped {dataset_name}: {e}")
            continue

        # --------------------------------------------------
        # ANALYSIS
        # --------------------------------------------------
        section("Dataset analysis")
        analyze_dataset(df, target_col)

        # --------------------------------------------------
        # SPLIT
        # --------------------------------------------------
        try:
            section("Splitting dataset")

            X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                df=df,
                target_col=target_col,
                test_size=0.30,     
                val_size=0.00,      
                time_column=None,
                dataset_name=dataset_name,
                iot_sample_frac=0.10
            )

            print(
                f"✅ Train={len(X_train)} | "
                f"Val={len(X_val)} | "
                f"Test={len(X_test)}"
            )

        except Exception as e:
            print(f"❌ Split failed: {e}")
            continue

        # --------------------------------------------------
        # SCALE BASE FEATURES
        # --------------------------------------------------
        line()
        section("Scaling features")
        line()

        X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

        # --------------------------------------------------
        # TARGET HEALTH CHECK
        # --------------------------------------------------
        if y_train.nunique() < 2:
            print("❌ Training set has only one class")
            continue

        if X_train.isna().any().any():
            print("❌ NaN values detected in training data")
            continue

        # --------------------------------------------------
        # CONFIGURATIONS
        # --------------------------------------------------
        for config in configurations:

            line()
            print(f"⚙️ CONFIG: {config['name']}")
            line()

            X_tr = X_train.copy()
            X_val_tr = X_val.copy()
            X_te = X_test.copy()

            # ----------------------------------------------
            # FEATURE ENGINEERING
            # ----------------------------------------------
            if config["features"]:
                line()
                section("Feature engineering")
                line()

                fe = FeatureEngineer()

                X_tr = fe.fit_transform(X_tr)
                X_val_tr = fe.transform(X_val_tr)
                X_te = fe.transform(X_te)

                print(f"✅ Features after engineering: {X_tr.shape[1]}")

            # ----------------------------------------------
            # RE-SCALE
            # ----------------------------------------------
            line()
            section("Rescaling engineered features")
            line()

            X_tr, X_val_tr, X_te = scale_data(X_tr, X_val_tr, X_te)

            # ----------------------------------------------
            # PCA
            # ----------------------------------------------
            if config["pca"]:
                line()
                section("Applying PCA")
                line()

                X_tr, X_val_tr, X_te = apply_pca(X_tr, X_val_tr, X_te)

                print(f"✅ PCA output features: {X_tr.shape[1]}")

            # ----------------------------------------------
            # MODELS
            # ----------------------------------------------
            for model_name in model_names:
                
                line()
                print(f"🚀 Logger run path: {dataset_name} | {config['name']} | {model_name}")
                line()
                
                run_path = logger.create_run(
                    dataset_name,
                    config["name"],
                    model_name
                )

                line()
                print(f"🤖 MODEL: {model_name}")
                line()

                try:
                    section("Training model")
                    
                    line()
                    print(f"🤖 MODEL NAME AND HANDLER FETCH : {model_name}")
                    line()
                    
                    model = get_model(model_name)
                    params = parameter_grids.get(model_name)

                    
                    
                    line()
                    start_train = time.perf_counter()
                    print(f"⏱️ Training {model_name}... at {time.strftime('%X')}")
                    line()
                    
                    handler = MODEL_HANDLERS.get(model_name, BaseModelHandler())
                    
                    trained_model = handler.train(
                        model=model,
                        X_train=X_tr,
                        y_train=y_train,
                        params=params,
                        dataset_name=dataset_name
                    )
                    
                    line()
                    end_train = time.perf_counter() - start_train
                    print(f"✅ Training completed in {end_train:.2f} seconds at {time.strftime('%X')}")
                    line()
                    
                    y_pred = trained_model.predict(X_te)
                    
                    y_test_clean = pd.Series(y_test).reset_index(drop=True)
                    y_pred_clean = pd.Series(y_pred).reset_index(drop=True)
                    
                    logger.save_predictions(run_path, y_test_clean, y_pred_clean)

                    # decoded labels only for IoT dataset
                    if dataset_name == "ioT_context":
                        reverse_map = {v: k for k, v in label_mapping.items()}

                        decoded_y_test = y_test_clean.map(reverse_map)
                        decoded_preds = y_pred_clean.map(reverse_map)

                        logger.save_predictions(run_path, decoded_y_test, decoded_preds)
                        
                    # metrics still use numeric labels
                    line()
                    section("📅 Evaluating model")
                    line()
                    
                    unique_preds = np.unique(y_pred)
                    unique_true = np.unique(y_test)

                    print("Predicted classes:", len(unique_preds))
                    print("True classes:", len(unique_true))
                    
                    metrics = evaluate(trained_model, X_te, y_test)

                    logger.save_metrics(run_path, metrics)
                    logger.save_config(run_path, config)

                    results.append({
                        "dataset": dataset_name,
                        "target": target_col,
                        "config": config["name"],
                        "model": model_name,
                        "train_time_sec": round(end_train, 4),
                        **metrics
                    })

                    print("✅ Completed")

                except Exception as e:
                    print(f"❌ Model failed: {e}")
                    continue

    # ------------------------------------------------------
    # FINAL EXPORT
    # ------------------------------------------------------
    line()
    print("📊 Saving summary")
    line()

    df_results = pd.DataFrame(results)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    df_results.to_csv(RESULTS_FILE, index=False)

    print(f"✅ Saved: {RESULTS_FILE}")

    total_time = time.perf_counter() - full_training_start_time
    print(f"✅ Total pipeline time: {total_time:.2f} seconds")
    
    return df_results
