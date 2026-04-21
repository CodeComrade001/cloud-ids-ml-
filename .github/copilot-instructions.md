# Copilot Instructions for cloud-ids-ml

## Project Overview

This is a machine learning pipeline for intrusion detection systems (IDS). It processes three cybersecurity datasets (cybersecurity_intrusion_data, Network_logs, IoT_Intrusion) through configurable pipelines combining feature engineering and PCA, training four sklearn models (KNN, SVM, MLP, RF) with hyperparameter tuning via GridSearchCV.

## Architecture

- **Entry Point**: `main.py` calls `experiments/runner.py::run_all()`, which executes `src/pipeline.py::run_pipeline()`.
- **Pipeline Flow**: Load data → Split (80/20 stratified) → Scale (StandardScaler) → Feature engineering (if enabled) → PCA (if enabled) → Train with GridSearchCV → Evaluate → Log results.
- **Key Modules**:
  - `src/preprocessing.py`: Data splitting and scaling (StandardScaler on numeric columns).
  - `src/feature_engineering.py`: Conditional feature creation based on dataset columns (e.g., payload_norm for Payload_Size, burstiness for IAT).
  - `src/dimensionality.py`: PCA with 95% variance retention.
  - `src/models.py`: Model factory and PARAM_GRIDS for hyperparameter search.
  - `src/trainer.py`: GridSearchCV training with 3-fold CV.
  - `src/evaluator.py`: Metrics calculation (accuracy, precision, recall, f1 weighted; ROC-AUC if probabilistic).
  - `experiments/ExperimentLogger.py`: Saves results to timestamped directories under `results/{dataset}/{config}/{model}/{timestamp}/` with metrics.json, predictions.csv, config.json.

## Data Flow

- Raw CSVs in `data/raw/`, cleaned in `data/cleaned/`, processed results in `results/`.
- IoT_Intrusion dataset is large (~1M rows), causing slow training; consider subsampling for quick tests.

## Workflows

- **Run Experiments**: `python main.py` executes all dataset-config-model combinations. Results summarized in `results/summary/all_results.csv`.
- **Debugging**: Check console prints for progress; individual run logs in timestamped dirs. Use `experiments/config.py` to enable/disable models (e.g., set MODELS = ["KNN"] for quick tests).
- **Adding Models**: Extend `src/models.py::get_model()` and PARAM_GRIDS; update `experiments/config.py::MODELS`.
- **Feature Engineering**: Modify `FeatureEngineer` to add dataset-specific features; fit on train, transform on test.
- **Evaluation**: Metrics use weighted averaging for multi-class; ROC-AUC only for probabilistic models.

## Conventions

- **Configs**: Defined in `experiments/config.py` as dicts with "name", "features" (bool), "pca" (bool).
- **Error Handling**: Raise RuntimeError with descriptive messages; check for "target" column in preprocessing.
- **Dependencies**: sklearn, pandas, numpy; install via `pip install -r requirements.txt`.
- **Logging**: Always use ExperimentLogger for reproducibility; avoid overwriting results.
- **Data Handling**: Assume "target" column for labels; drop NaN targets; stratify splits.

## Examples

- To test KNN on baseline config: Set MODELS = ["KNN"], CONFIGS = [{"name": "baseline", "features": False, "pca": False}] in config.py, run main.py.
- Adding a feature: In `FeatureEngineer.transform()`, if "new_col" in X.columns: X["derived"] = X["new_col"] \* 2.
