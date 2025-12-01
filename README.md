# Minute-Level Futures Direction Baseline

This repository contains a **baseline machine learning pipeline** for predicting the **next-minute price direction** of Chinese futures contracts using **minute-level data**.

The core implementation lives in:

- `baseline_notebook.ipynb`

The notebook is designed to be:

- **Reproducible** – fixed pipeline, clear configuration block.
- **Extensible** – easy to swap models, change features, or use more contracts.
- **Interpretable** – includes diagnostics such as rolling-window performance, confusion matrices, calibration curves, and coefficient analysis.

---

## 1. Problem Description

Given minute-level OHLCV data for one or more futures main contracts, we aim to predict the **direction of the next-minute close price**:

- Let \( P_t \) be the close price at minute \( t \).
- Define the label using a threshold \( \alpha > 0 \):

\[
Y_{t+1} =
\begin{cases}
1  & \text{if } \dfrac{P_{t+1}}{P_t} > 1 + \alpha, \\
-1 & \text{if } \dfrac{P_{t+1}}{P_t} < 1 - \alpha, \\
0  & \text{otherwise.}
\end{cases}
\]

This yields a **3-class classification problem**:
- `1`: Significant upward move  
- `-1`: Significant downward move  
- `0`: Flat/no significant move  

The baseline model is **multinomial logistic regression** with:
- **Standardized features** (per training fold).
- **Class weights** inversely proportional to the observed class frequencies.
- **Rolling-window cross-validation** over time.

---

## 2. Data

The project expects the raw data to be stored under:

- `2005年__20250905/`

Each CSV file represents **one futures contract** (usually a main contract) and should include at least the following columns:

- `index` – timestamp (parsed as `datetime`)
- `open`, `high`, `low`, `close`
- `volume`
- `open_interest`
- `avg` – VWAP-like average price (used to construct spread features)

The name of a file (e.g. `AG_主力合约_1m数据.csv`) is treated as the **contract name** in logs, figure titles, and output folder names.

---

## 3. Pipeline Overview

For each selected contract (CSV file), the notebook runs the following pipeline:

1. **Data loading**  
   - Read one CSV with optional:
     - row cap (`NROWS_PER_FILE`)
     - date range filter (`START_DATE`, `END_DATE`)
   - Sort by `index` and reset the index.

2. **Feature engineering**  
   The function `add_feature_columns` constructs baseline features, including:
   - Minute log return:
     - `r_t = log(close_t) − log(close_{t-1})`
   - Lagged returns:
     - `r_t_minus_1` … `r_t_minus_5`
   - Moving average:
     - `ma_5` – 5-minute moving average of `close`
   - RSI-like signal (`rsi_5`):
     - Uses 5-period rolling average of gains/losses.
   - Range features:
     - `range_1m = high − low`
     - `range_5` – 5-minute rolling high–low range
   - Volume / open interest changes:
     - `volume_change`, `oi_change`
   - Price–VWAP spread:
     - `spread_vwap = close − avg`
   - Additional features:
     - `abs_r_t` – absolute return
     - `r_times_volume` – return × volume
     - `co_move` – `close − open`

   A subset of these is selected as `BASELINE_FEATURES`.

3. **Label generation**  
   - Next-minute label computed by `generate_labels(df, alpha=0.001)`.
   - Rows with missing values (from lags, rolling windows, or shift) are dropped.

4. **Rolling-window cross-validation**  
   - Implemented by `rolling_month_windows(df, train_months=12, test_months=1)`.
   - For each fold:
     - Use 12 months for training, 1 month for testing.
     - Splits respect chronological order (no look-ahead).
     - Windows slide forward by 1 month.

5. **Model training**  
   - For each candidate regularization parameter \( C \) in `C_values`:
     - For each fold:
       - Extract training and test feature matrices via `get_feature_matrix`.
       - Standardize using `StandardScaler` **fitted on training only**.
       - Compute class weights from training labels:
         \[
           w_c \propto \frac{1}{\text{freq}(c)}
         \]
       - Fit multinomial logistic regression:
         ```python
         LogisticRegression(
             multi_class="multinomial",
             solver="lbfgs",
             max_iter=200,
             C=C,
             class_weight=class_weights,
             n_jobs=-1,
         )
         ```
       - Predict class labels and class probabilities.

6. **Model selection and metrics**  
   - For each \( C \), aggregate fold metrics:
     - Accuracy
     - Macro F1
     - Precision for label `1` and `-1`
   - Rank \( C \) by:
     1. Average macro F1 (descending)
     2. Average accuracy (descending)
   - Select **best C** and store detailed information:
     - Per-fold metrics
     - All predictions and probabilities
     - Confusion matrices per fold
     - Final model and scaler
     - Training feature matrix (for VIF)

7. **Diagnostics and interpretability**  
   - Aggregate confusion matrix across folds.
   - Compute and save:
     - Logistic coefficients per class
     - Intercepts
     - Standardization parameters (mean, scale)
   - Compute VIF (Variance Inflation Factor) on the last training fold.
   - Compute calibration curve for class `1` using binned predicted probabilities.

---

## 4. Multi-Contract Training Structure

The notebook supports **multi-contract training** as follows:

- When multiple CSV files are selected (via `MAX_FILES` or by default), the pipeline:
  - Loops over each `csv_path` in `csv_files`.
  - For each contract:
    - Loads data for that contract.
    - Builds features + labels.
    - Runs the full training + evaluation pipeline independently.
  - Outputs are organized by **contract-specific subfolders** under the run’s timestamp.

Example directory structure:

```
outputs/<RUN_TIMESTAMP>/
  figures/
    AG_主力合约_1m数据/
      price_volume_overview_<timestamp>.png
      label_distribution_<timestamp>.png
      ...
    RB_主力合约_1m数据/
      ...
  metrics/
    AG_主力合约_1m数据/
      rolling_metrics_<timestamp>.csv
      logit_coefficients_<timestamp>.csv
      ...
    RB_主力合约_1m数据/
      ...
```

This makes it easy to:
- Compare performance across different contracts.
- Inspect model behavior per underlying instrument.
- Keep all outputs organized and reproducible.

---

## 5. Configuration Knobs

All the main configuration parameters are defined near the top of the notebook.

### Paths and basic options

```python
# Root folder for CSV files
DATA_DIR = Path("2005年__20250905")

# Control how many files to load (None = all)
MAX_FILES: Optional[int] = 1

# Optional cap on rows per file (None = full file)
NROWS_PER_FILE: Optional[int] = None

# Time window filter (inclusive)
START_DATE: Optional[str] = "2005-01-01"
END_DATE: Optional[str] = "2013-12-31"
```

You can adjust:
- `MAX_FILES` to run more contracts.
- `NROWS_PER_FILE` to limit rows for faster experiments.
- `START_DATE` / `END_DATE` to control which years/months to include.

### Features

```python
BASELINE_FEATURES = [
    "r_t",
    "r_t_minus_1", "r_t_minus_2", "r_t_minus_3", "r_t_minus_4", "r_t_minus_5",
    "ma_5",
    "rsi_5",
    "range_5",
    "abs_r_t",
    "r_times_volume",
    "co_move",
]
```

You can:
- Add/remove features here (as long as they are created in `add_feature_columns`).
- Experiment with alternative technical indicators.

### Label threshold

Defined in `generate_labels(df, alpha=0.001)`:
- `alpha` controls the “no-trade / neutral zone” around flat moves.
- Larger `alpha` → more 0 labels, fewer ±1 labels.

### Model and regularization

Candidate \(C\) values are defined when calling the pipeline:

```python
candidate_C = [0.0001, 0.001, 0.01, 0.1, 1.0]
```

You can:
- Add more values or change the grid to explore different levels of regularization.

### Rolling-window settings

```python
def rolling_month_windows(df: pd.DataFrame,
                          train_months: int = 12,
                          test_months: int = 1):
    ...
```

You can:
- Increase `train_months` to use more history per fold.
- Increase `test_months` to test over longer periods per fold.

---

## 6. Outputs

For each run (invocation of the notebook), a new timestamped directory is created:

```
outputs/<RUN_TIMESTAMP>/
  figures/
  metrics/
```

Within each contract-specific subfolder, you can expect:

### Metrics (`metrics/<contract_name>/`)
- `rolling_metrics_<timestamp>.csv`  
  Per-fold performance (accuracy, macro F1, precision for ±1, time window).
- `predictions_detailed_<timestamp>.csv`  
  Each prediction: timestamp, true label, predicted label, class probabilities.
- `confusion_matrix_<timestamp>.csv`  
  Aggregate confusion matrix over all folds.
- `logit_coefficients_<timestamp>.csv`  
  Logistic regression coefficients per class and feature.
- `logit_params_<timestamp>.csv`  
  Combined table of scaler parameters (mean/scale) and coefficients.
- `vif_summary_<timestamp>.csv`  
  VIF values per feature, including high-VIF warnings.
- `calibration_curve_<timestamp>.csv`  
  Binned predicted probabilities vs observed positive rate for class 1.

### Figures (`figures/<contract_name>/`)
- `price_volume_overview_<timestamp>.png`  
  Price vs volume over time.
- `label_distribution_<timestamp>.png`  
  Class distribution of the labels.
- `feature_<feature>_by_label_<timestamp>.png`  
  Boxplots of features by label.
- `feature_correlation_heatmap_<timestamp>.png`  
  Correlation matrix of features.
- `probability_hist_<timestamp>.png`  
  Histogram of predicted probabilities for class 1.
- `rolling_performance_<timestamp>.png`  
  Rolling accuracy and macro F1 over test windows.
- `confusion_matrix_<timestamp>.png`  
  Heatmap of confusion matrix.
- `logit_coef_barplot_<timestamp>.png`  
  Top coefficients (by absolute value) for class 1.
- `calibration_plot_<timestamp>.png`  
  Calibration curve.

---

## 7. How to Run

### 7.1. Clone the repository

```bash
git clone git@github.com:NianXi1020/ESL-ML-Methods-Chinese-Future-Data.git
cd ESL-ML-Methods-Chinese-Future-Data
```

Ensure that the data folder `2005年__20250905` exists under this repository (or adjust `DATA_DIR` in the notebook accordingly).

### 7.2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate    # on macOS / Linux
# .venv\Scripts\activate     # on Windows PowerShell
```

### 7.3. Install dependencies

At minimum, you will need:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter (or jupyterlab)

You can install them via:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

(If a `requirements.txt` file exists in the repo, prefer `pip install -r requirements.txt`.)

### 7.4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open:

- `baseline_notebook.ipynb`

### 7.5. Configure and run

In the notebook:

- Adjust configuration cell: `DATA_DIR`, `MAX_FILES`, `NROWS_PER_FILE`, `START_DATE`, `END_DATE`.
- Run data loading, feature engineering, and labeling cells.
- Optionally inspect: `data_raw` and `data_labeled` (head, number of distinct months).
- Run the final pipeline cell, which calls:

```python
results = run_full_pipeline(
    data_raw=data_raw_i,
    data_labeled=data_labeled_i,
    csv_files=[csv_path],
    C_values=candidate_C,
    feature_list=BASELINE_FEATURES,
    boxplot_features=BASELINE_FEATURES[:5],
)
```

(In the multi-contract version, this call will be inside a loop over `csv_files`.)

After execution, inspect:

- On-screen summary (best C, rolling means/std, top coefficients, VIF diagnostics).
- Saved CSVs and PNGs under `outputs/<RUN_TIMESTAMP>/`.

---

## 8. Extending the Project

Potential extensions for collaborators:

### Alternative models

Replace logistic regression with:
- Linear SVM
- Gradient boosted trees
- Random forest
- Deep neural networks (e.g. LSTM, Transformer over returns)

### Richer features

- Higher-order lags.
- Volatility estimators.
- Microstructure features (bid-ask spread, order book depth if available).

### Risk / PnL simulation

- Turn predicted labels into trading signals.
- Simulate transaction costs and slippage.

### Cross-contract analysis

- Compare which contracts are more predictable.
- Analyze stability of coefficients and VIF across underlying assets.

### Bayesian or stochastic volatility models

- Use this baseline as a benchmark against more advanced time-series or SV models.

