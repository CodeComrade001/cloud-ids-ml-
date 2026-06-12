# cloud-ids-ml (Final Year Project)

This repository implements a full ML pipeline for **Cloud/Network intrusion detection** (multi-dataset), including:

- dataset loading & label normalization
- train/validation/test splitting (with IoT-specific downsampling)
- feature scaling
- optional feature engineering
- optional PCA dimensionality reduction
- model training (with grid search where applicable)
- evaluation and persistent experiment logging
- saving predictions per run and exporting a CSV summary

---

## 1. Repository Structure

```
cloud-ids-ml/
├─ main.py
├─ requirements.txt
├─ README.md
├─ data/
│  ├─ raw/
│  │  ├─ cybersecurity_intrusion_data.csv
│  │  ├─ IoT_Intrusion.csv
│  │  └─ Network_logs.csv
│  └─ cleaned/
│     ├─ cybersecurity_intrusion_data_cleaned.csv
│     ├─ IoT_Intrusion_cleaned.csv
│     └─ Network_logs_cleaned.csv
├─ experiments/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ ExperimentLogger.py
│  ├─ logger.py
│  ├─ results_merge.py
│  └─ runner.py
├─ results/
│  └─ (per-run folders)
└─ src/
   ├─ __init__.py
   ├─ pipeline.py
   ├─ loader.py
   ├─ preprocessing.py
   ├─ feature_engineering.py
   ├─ dimensionality.py
   ├─ evaluator.py
   ├─ models.py
   ├─ model_handler.py
   └─ datasets_cleaner.py
```

---

## 2. How to Run the Project

### 2.1 Install dependencies

```bash
pip install -r requirements.txt
```

### 2.2 Execute experiments

The typical entry point is `main.py`, which triggers the experiment runner.

```bash
python main.py
```

What happens `main.py` is executed:

- `experiments.runner.run_all()` is called
- it calls `src.pipeline.run_pipeline(...)`
- the pipeline loops over:
  - datasets in `src.pipeline.DATASETS`
  - preprocessing configurations in `experiments.config.CONFIGS`
  - model names in `experiments.config.MODELS`
- each run is logged into a timestamped folder under `results/`
- finally a per-run summary CSV is written (based on the active model selection in `experiments/runner.py`)

---

## 3. System Architecture (Layered Design)

This project is organized in layers so each responsibility is isolated:

1. **Experiment configuration layer** (`experiments/config.py`, `experiments/runner.py`)
2. **Core pipeline orchestration** (`src/pipeline.py`)
3. **Data handling** (`src/loader.py`, `src/preprocessing.py`)
4. **Feature transformations** (`src/feature_engineering.py`, `src/dimensionality.py`)
5. **Model creation and hyperparameters** (`src/models.py`, `src/model_handler.py`)
6. **Evaluation** (`src/evaluator.py`)
7. **Experiment logging** (`experiments/ExperimentLogger.py`)
8. **Optional dataset cleaning utilities** (`src/datasets_cleaner.py`)

---

## 4. Experiment Configuration

### 4.1 `experiments/config.py`

Defines preprocessing configurations (`CONFIGS`) and which models to run (`MODELS`).

#### Configurations

```python
CONFIGS = [
    {"name": "baseline", "features": False, "pca": False},
    {"name": "features_only", "features": True, "pca": False},
    {"name": "pca_only", "features": False, "pca": True},
    {"name": "features_pca", "features": True, "pca": True},
]
```

**Interpretation:**

- `features`: enables `FeatureEngineer`
- `pca`: enables PCA with variance retention `n_components=0.95`

#### Model selection

Currently only one model is enabled at a time for performance/resource stability:

```python
MODELS = ["KNN"]
```

All other models are commented out.

**Rationale / exception:** This repo includes multiple heavy models (SVM grid search, MLP, ensembles). Running them all at once can exceed RAM/CPU on low-end machines. The codebase is designed to run **sequential manual training**.

---

## 5. Core Pipeline Orchestration (`src/pipeline.py`)

### 5.1 Datasets used in training

`src/pipeline.py` defines:

```python
DATASETS = {
    "baseline_dataset": "data/cleaned/cybersecurity_intrusion_data_cleaned.csv",
    "network_logs_dataset": "data/cleaned/Network_logs_cleaned.csv",
    "ioT_context": "data/cleaned/IoT_Intrusion_cleaned.csv"
}
```

Each dataset is treated similarly, but the pipeline has IoT-specific logic in `split_data()` and additional safeguards in `model_handler.py`.

### 5.2 Pipeline steps

For each dataset:

1. **Load dataset** (`load_data`)
2. **Standardize target** (`standardize_target`)
3. **Validate dataset** (`validate_dataset`)
4. **Choose target column**
   - `target_multi` if it exists and has >2 unique values
   - else `target_binary`
   - else `target`
5. **Analyze dataset** (`analyze_dataset`) for shape & label distribution
6. **Split** into train/test/val (val_size is set to 0.00)
   - IoT-specific downsampling fraction is passed as `iot_sample_frac=0.10`
7. **Scale features** (`scale_data`)
8. **Loop configurations** (`baseline`, `features_only`, `pca_only`, `features_pca`)
   - optional **feature engineering**
   - rescale after engineering (important)
   - optional **PCA**
9. **Loop models**
   - create model instance (`get_model`)
   - pick hyperparameter grid (`PARAM_GRIDS` from `src/models.py`)
   - train through **handler abstraction** (`MODEL_HANDLERS[model_name]`)
   - save predictions
   - evaluate metrics
   - save metrics and config
10. **Export summary CSV** for the selected model(s)

### 5.3 Key exception/rationale inside pipeline

- **Target selection is dynamic**: datasets may produce `target_multi` vs `target_binary`.
- **Rescaling after feature engineering**: engineered features can change numeric ranges; scaling is reapplied to keep model assumptions consistent.
- **Graceful skipping**: almost every major stage is wrapped in `try/except` so a single dataset/model failure does not stop the whole experiment sweep.

---

## 6. Data Loading and Label Normalization (`src/loader.py`)

### 6.1 `load_data(path)`

- Reads CSV
- Normalizes column names: lowercase, strip spaces, replace spaces with `_`
- Renames any label-like column to `target`:
  - `intrusion` -> `target`
  - `attack_detected` -> `target`
  - `label` -> `target`
- Raises a clear error if no valid label column exists

### 6.2 `standardize_target(df)`

Creates multiple label representations:

- `target_multi`: `LabelEncoder` numeric classes
- `target_binary`: **binary mapping**
  - if dataset already has exactly 2 unique labels → binary uses the same encoding
  - otherwise:
    - strings containing `normal` or `benign` map to 0
    - everything else maps to 1
  - safety check ensures `target_binary` has at least 2 classes

Also prints the mapping for interpretability.

**Rationale / exception:**
Binary conversion is “safer” when datasets are multi-class; the code explicitly verifies binary mapping did not collapse into one class.

### 6.3 `validate_dataset(df)`

- ensures `target` exists
- ensures there are no NaNs in `target`
- ensures dataset is non-empty

---

## 7. Preprocessing & Splitting (`src/preprocessing.py`)

### 7.1 Dataset analysis (`analyze_dataset`)

Prints:

- shape
- class distribution counts and percentages
- duplicates

### 7.2 Splitting (`split_data`)

Arguments include:

- `test_size=0.30`
- `val_size=0.00` (no separate validation set)
- optional `time_column` (if present, time-based split logic applies)
- `dataset_name`
- `iot_sample_frac=0.10`

Splitting logic:

1. Remove NaNs in target
2. Build feature matrix `X` by dropping label columns:
   - `target`, `target_multi`, `target_binary` (if present)
3. If `time_column` is supplied: time-based sequential split
4. Else: random split with optional stratification
   - `safe_stratify` disables stratification if some class is too rare
5. **IoT-specific reduction**
   - if `dataset_name == "ioT_context"`:
     - it resamples `X_train`/`y_train` down to `train_size=iot_sample_frac` (10%)
     - keeps stratification where possible

**Rationale / exception:**
IoT cleaned dataset is much larger (see datasets section below), so this reduction avoids memory/time blow-ups.

### 7.3 Scaling (`scale_data`)

- Uses `StandardScaler`
- Only scales numeric columns (`int64`, `float64`)

---

## 8. Feature Engineering (`src/feature_engineering.py`)

`FeatureEngineer` conditionally adds derived features depending on which columns exist.

Examples:

- If `Payload_Size` exists:
  - `payload_norm = Payload_Size / payload_mean`
- If `Port` exists:
  - `is_high_port = 1 if Port > 1024 else 0`
- If `Duration` exists:
  - `duration_norm = Duration / duration_mean`
- If packet min/max exist:
  - `packet_range = Max - Min`
- If `IAT` exists:
  - `burstiness = 1 / (IAT + 1e-6)`
- If IoT login columns exist:
  - `login_norm`, `failure_ratio`

**Design choice:**
This approach is schema-agnostic: each dataset may have different columns, so transformations are applied only when relevant.

---

## 9. Dimensionality Reduction (`src/dimensionality.py`)

PCA:

```python
pca = PCA(n_components=0.95)
```

It retains enough components to explain 95% variance.

**Important pipeline exception:**
PCA is applied _after_ scaling and after optional feature engineering.

---

## 10. Models (`src/models.py`)

### 10.1 Model factory (`get_model(name)`)

Supported models:

- `KNN`: `KNeighborsClassifier()`
- `SVM`: `SVC(probability=True)`
- `MLP`: `MLPClassifier()`
- `RF`: `RandomForestClassifier()`
- `LR`: `LogisticRegression(saga, class_weight="balanced", max_iter=1000, random_state=42)`
- `DT`: `DecisionTreeClassifier()`
- `NB`: `GaussianNB()`
- `GB`: `GradientBoostingClassifier()`

Voting ensemble exists in comments, not currently active.

### 10.2 Hyperparameter grids (`PARAM_GRIDS`)

Example:

- KNN: `n_neighbors` and `weights`
- SVM: `C` and `kernel`
- RF: `n_estimators` and `max_depth`
- LR: small grid (`C=[1]`)
- MLP: limited grid

---

## 11. Training Specialization / Handlers (`src/model_handler.py`)

Training does not happen directly via `model.fit(...)` only.
Instead, it routes model training through handlers:

```python
MODEL_HANDLERS = {
  "KNN": BaseModelHandler(),
  "SVM": SVMHandler(),
  "MLP": MLPHandler(),
  "RF": BaseModelHandler(),
  "LR": LRHandler(),
  "DT": BaseModelHandler(),
  "NB": BaseModelHandler(),
  "GB": GBHandler(),
}
```

### 11.1 Grid search strategy (`train`)

`trainer` logic (in the repo) uses:

- `GridSearchCV` when params are provided
- reduces CV folds and jobs for `ioT_context`
- disables grid search if smallest class counts make CV impossible

### 11.2 IoT exceptions

To control runtime for the largest dataset, handlers introduce dataset-specific exceptions:

- **SVMHandler**

  - downsample train to at most 10,000 rows if too large
  - remove classes with <2 samples to prevent training failures
  - disables grid search (`params = {}`)

- **MLPHandler**

  - clone model, set early stopping
  - downsample to at most 30,000 rows
  - increases `max_iter` (dataset-dependent)

- **LRHandler**

  - sets solver, `class_weight="balanced"`, n_jobs=-1 for IoT
  - provides a tighter param grid

- **GBHandler**
  - downsample to at most 60,000 rows
  - uses conservative depth/learning rate/subsample
  - disables grid search (`params = {}`)

**Rationale / exception:**
These are deliberate performance/robustness trade-offs for the huge IoT dataset.

---

## 12. Evaluation (`src/evaluator.py`)

Metrics computed:

- accuracy
- weighted precision
- weighted recall
- weighted F1
- roc_auc (best-effort)

ROC-AUC logic:

- if model supports `predict_proba`, it uses it
- else if it supports `decision_function`, it uses that
- otherwise ROC-AUC is set to `None`

---

## 13. Experiment Logging (`experiments/ExperimentLogger.py`)

Each run creates a dedicated folder:

```
results/
  {dataset}/
    {config_name}/
      {model_name}/
        {YYYY-MM-DD_HH-MM-SS}/
```

Inside the run folder:

- `metrics.json`
- `predictions.csv` (numeric labels)
- `config.json`
- optionally a second `predictions.csv` save for IoT decoded labels (see note below)

### Important note: IoT decoded predictions

In `src/pipeline.py`:

- numeric labels are always used for metrics
- for IoT (`dataset_name == "ioT_context"`), a reverse label mapping is computed and decoded labels are saved

---

## 14. Results Export and Merging

### 14.1 Runner output CSV

`experiments/runner.py` picks one active result file.
Currently:

- `ALL_RESULTS_DIR = "results/summary/all_knn_results.csv"`

### 14.2 Merging (`experiments/results_merge.py`)

Optional helper to merge multiple CSV summaries into a single combined file.

---

## 15. Datasets (CSV Overview)

This section summarizes what was observed in the repository.

### 15.1 Raw datasets present in `data/raw/`

There are three main CSVs:

- `data/raw/cybersecurity_intrusion_data.csv` (~9.5k rows)
- `data/raw/IoT_Intrusion.csv` (~1,048,575 rows)
- `data/raw/Network_logs.csv` (~8.8k rows)

### 15.2 Cleaned datasets present in `data/cleaned/`

These are the versions actually referenced by the training pipeline:

- `data/cleaned/cybersecurity_intrusion_data_cleaned.csv`
- `data/cleaned/IoT_Intrusion_cleaned.csv`
- `data/cleaned/Network_logs_cleaned.csv`

**Why cleaned datasets are used:**
The training pipeline expects mostly numeric features and consistent label column naming (`target`). Cleaning scripts are provided for these transformations.

### 15.3 Large dataset handling (important)

Because IoT raw data is >1M rows, the system includes explicit downsampling and handler-specific training limitations (see Section 7 and Section 11).

---

## 16. Dataset Cleaning Utilities (`src/datasets_cleaner.py`)

This file contains cleaning functions for different dataset types, including:

- dropping identifier columns
- coercing features to numeric
- filling missing values (median for numeric, mode/"Unknown" for categorical)
- encoding categorical features via one-hot (`get_dummies`)
- removing duplicates
- dropping columns that violate NaN thresholds

**Important:** These cleaner functions are utilities; the training pipeline directly consumes the cleaned CSVs under `data/cleaned/`.

---

## 17. Project Decisions & Rationale Summary (What a co-worker must know)

1. **Sequential, single-model execution** (configurable) to fit low RAM/CPU constraints.
2. **Dynamic target selection** because different cleaned datasets may produce multi-class vs binary representations.
3. **Feature-engineering is schema-conditional** so it works across datasets with different column sets.
4. **Rescale after engineering** to keep scaling assumptions valid.
5. **PCA optional and variance-retaining** (`0.95`) to reduce model complexity when enabled.
6. **IoT-specific downsampling & training safeguards** to avoid training failures and excessive runtime.
7. **Graceful skips**: dataset/model training failures do not crash the whole run.

---

## 18. Expected Outputs

After running `python main.py`, expect:

- Per-run output folders under `results/`
- `results/summary/all_knn_results.csv` (or other model file if you change runner configuration)

Each run folder contains:

- `metrics.json`
- `predictions.csv`
- `config.json`

---

## 19. Troubleshooting Notes

- If training is skipped for a dataset, check the console logs printed from `src/pipeline.py`.
- If ROC-AUC is `None`, the model may not support probability estimates in this configuration.
- If IoT models fail, verify that the IoT handler-specific downsampling/class pruning logic is active.

---

## 20. Appendix: Key Files (Quick Map)

- `main.py`: entry point
- `experiments/runner.py`: selects summary CSV output and triggers `run_pipeline`
- `experiments/config.py`: chooses preprocessing configurations and active model(s)
- `src/pipeline.py`: entire training orchestration
- `src/loader.py`: CSV loading and target normalization
- `src/preprocessing.py`: split and scaling
- `src/feature_engineering.py`: optional derived feature creation
- `src/dimensionality.py`: optional PCA
- `src/models.py`: model instantiation and hyperparameter grids
- `src/model_handler.py`: specialized training strategy (especially IoT)
- `src/evaluator.py`: metrics computation
- `experiments/ExperimentLogger.py`: run folder outputs
- `experiments/results_merge.py`: merge helper for summary CSVs
