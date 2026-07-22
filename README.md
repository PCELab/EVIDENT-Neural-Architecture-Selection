# EVIDENT: Evidence-Based Neural Architecture Selection

EVIDENT is a research implementation of evidence-based neural architecture selection under uncertainty. The example in this repository uses Bayesian temporal convolutional networks (TCNs) for multi-step blood-glucose forecasting in type 1 diabetes.

The repository contains four main stages:

`Bergman simulation → CSV preparation → Bayesian TCN architecture search → validation/testing`

The Python files are standalone experiment scripts rather than an installed package or command-line application. They execute from top to bottom and currently contain experiment-specific absolute paths that must be changed before use.

## Repository layout

```text
EVIDENT-Neural-Architecture-Selection/
├── Code/
│   ├── Bayesian_TCN_training.py
│   └── Bayesian_TCN_validation.py
├── Data/
│   └── Bergman/
│       ├── bergman_final.m
│       └── t1d_basal_bolus_meal_aligned.csv
├── LICENSE
└── README.md
```

## Requirements

The Python experiments require Python 3 and the following packages:

- PyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib

The synthetic-data generator requires MATLAB. It uses `ode45` and local functions in a script, so MATLAB R2016b or newer is recommended.

CUDA is used automatically when available; otherwise, the model scripts run on the CPU. The default epoch counts, Monte Carlo sampling, and architecture search make a full run computationally expensive.

## End-to-end workflow

### 1. Generate synthetic type 1 diabetes data

[`Data/Bergman/bergman_final.m`](Data/Bergman/bergman_final.m) simulates glucose dynamics with a five-state type 1 diabetes Bergman minimal model, a stomach/intestine meal-absorption model, and external basal-plus-bolus insulin delivery. Endogenous glucose-stimulated insulin secretion is disabled, consistent with the type 1 diabetes setting.

The model is solved at one-minute intervals. For `N` simulated days, the time vector is `0:N*1440` inclusive, producing `1440*N + 1` data rows and normally three meal events per day.

#### Choose the number of days

Open `bergman_final.m` and change:

```matlab
numDays = 7;
```

For example, use `numDays = 30;` to generate 30 days of data.

#### Meal-size and meal-time uncertainty

The script defines three nominal meals per day:

| Meal | Nominal size | Nominal time from start of day |
|---|---:|---:|
| 1 | 20 g | 0 minutes |
| 2 | 40 g | 240 minutes |
| 3 | 60 g | 540 minutes |

Each day's meal sizes and times are sampled independently:

- Meal-size uncertainty is uniform within ±30% of the nominal amount: `D = D_base * (0.7 + 0.6*rand())`. The resulting ranges are 14–26 g, 28–52 g, and 42–78 g.
- Meal-time uncertainty is an integer-valued uniform jitter of ±50 minutes: `randi([-50, 50])`.

To change these uncertainty levels, edit the multiplier in the three `D1`, `D2`, and `D3` assignments and the range in the three `randi` calls. For example, `0.8 + 0.4*rand()` gives ±20% meal-size uncertainty, and `randi([-30, 30])` gives ±30 minutes of timing uncertainty.

`rng('default')` makes the generated realization repeatable. Replace it with `rng(<integer seed>)` for a different reproducible realization or `rng('shuffle')` for a new realization each run.

> **Boundary note:** The first nominal meal is at minute 0. With a different random seed, negative jitter can move the first meal before the simulation begins and cause invalid indexing. For robust randomized runs, move the first nominal time to at least the maximum negative jitter (for example, 60 minutes when using ±50 minutes), or clip/resample meal times to the simulation interval.

#### Run the generator

From the repository root:

```bash
cd Data/Bergman
matlab -batch "bergman_final"
```

The script displays a glucose/insulin plot and overwrites `t1d_basal_bolus_meal_aligned.csv` in MATLAB's current working directory.

The generated CSV contains:

| Column | Description |
|---|---|
| `Time` | Minutes from the start of the simulation |
| `Glucose` | Simulated blood glucose concentration in mg/dL |
| `Insulin` | External basal-plus-bolus insulin delivery rate in mU/L/min |
| `MealOnset` | `1` at a meal event and `0` otherwise |
| `MealSize` | Meal mass in mg at the event row and `0` otherwise |

The included CSV contains the default seven-day realization: 10,081 data rows and 21 meal events.

### 2. Prepare the input CSV

The two Python scripts do not currently read the bundled Bergman file automatically. Both contain a hard-coded absolute path to `diabetes_data_insulin.csv`; replace the `pd.read_csv(...)` path in each script with the path to your dataset.

The required columns are:

| Script | Required columns |
|---|---|
| `Bayesian_TCN_training.py` | `Time`, `PID`, `Glucose`, `MealSize`, `Insulin` |
| `Bayesian_TCN_validation.py` | `Time`, `Glucose`, `MealSize`, `Insulin` |

`Time` must be parseable by `pandas.to_datetime`. `PID` identifies a subject and prevents training windows from crossing subject boundaries. The bundled Bergman CSV uses numeric minutes and does not include `PID`, so it needs a small conversion before it can be used by the training script. For a single synthetic subject:

```python
from pathlib import Path
import pandas as pd

source = Path("Data/Bergman/t1d_basal_bolus_meal_aligned.csv")
target = Path("Data/Bergman/diabetes_data_insulin.csv")

df = pd.read_csv(source)
df["Time"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(df["Time"], unit="min")
df["PID"] = "bergman_001"
df.to_csv(target, index=False)
```

The Python scripts do not normalize their inputs. The Bergman generator writes `MealSize` in milligrams; if the rest of an experiment represents meal size in grams, divide this column by 1,000 before training and use the same unit in every dataset.

### 3. Train and select a Bayesian TCN architecture

[`Code/Bayesian_TCN_training.py`](Code/Bayesian_TCN_training.py) trains and ranks Bayesian encoder-decoder TCNs. Its main functionality is:

1. Read raw `Glucose`, `MealSize`, and `Insulin` measurements.
2. Build subject-specific, stride-one sequences with 1,440 input rows. The future target is glucose only.
3. Infer the forecast length from the smallest positive gap between meal episodes, falling back to 60 rows when fewer than two meals exist (or one row when no positive gap exists). Forecast windows containing a meal are excluded.
4. Preserve window order and use the first 80% for training and the remaining 20% for testing.
5. Train six predefined encoder/decoder depth and kernel-size candidates with a variational ELBO objective.
6. Optimize the observation-noise variance and weight-prior variance using the evidence objective, then rebuild and retrain each candidate with the calibrated values.
7. Draw Monte Carlo posterior-weight samples to calculate prediction means and uncertainty bands.
8. Report RMSE, R², relative L2 error, log predictive likelihood, and log evidence. The candidate with the largest log evidence is reported as the selected architecture.

The Bayesian convolutional layers learn Gaussian posterior means and scales for their weights. The encoder uses residual dilated temporal blocks and average pooling; the decoder uses residual Bayesian convolutional blocks and linear upsampling to produce a multi-step glucose forecast.

After setting the dataset path, run from the repository root:

```bash
python Code/Bayesian_TCN_training.py
```

The default search trains each candidate for 800 epochs, performs 1,000 evidence-hyperparameter steps, and retrains it for 600 epochs. These values and the candidate list are defined near the end of the script.

Important outputs are written relative to the directory from which the command is run:

- `posterior_params__<architecture>.npz`: posterior parameters for the retrained model.
- `evidence_npz/evidence_constants__<architecture>.npz`: constants used in the evidence calculation.
- `train_uncertainty__<architecture>.csv` and `.png`: aggregated training predictions and uncertainty bands.
- `mc_predictions_windows_*.csv`: actual values, Monte Carlo samples, predictive mean, variance, standard deviation, and ±2σ bounds for test windows.
- `combined_windows_*.png`: test predictions and uncertainty bands.
- `all_conv_sigmas.txt`: posterior convolutional weight scales appended during training.

### 4. Validate/test the selected architecture

[`Code/Bayesian_TCN_validation.py`](Code/Bayesian_TCN_validation.py) is the testing/evaluation stage despite its `validation` filename. It is not checkpoint-only inference: it calibrates and retrains the configured architecture before evaluating it on held-out data.

Its main functionality is:

1. Read `[Glucose, MealSize, Insulin]` and create chronological sliding windows with 1,400 input rows and 60 future glucose rows.
2. Split the original time series chronologically: 60% for training and 40% for testing, while retaining enough pre-split context for the first test window.
3. Build the configured encoder `[2, 4, 8, 16, 32]`, decoder `[32, 16, 8, 4, 2]`, kernel-size-7 Bayesian TCN.
4. Optionally load the matching `posterior_params__*.npz` produced by the training stage and use that posterior as an informative prior. If the file is absent, the script continues with its scalar prior.
5. Fit the model, optimize observation-noise and prior variances, transfer the learned posterior into a second model's prior, and retrain.
6. Evaluate every 60th test window using 50 posterior-weight samples per window.
7. Report combined RMSE, R², relative L2 error, log evidence, and last-window log posterior predictive likelihood.

Before running, update both the validation dataset path and `seed_npz_default`. Point `seed_npz_default` to the matching posterior file from the training stage if warm-started validation is desired.

```bash
python Code/Bayesian_TCN_validation.py
```

The validation stage writes a posterior `.npz`, evidence constants, a Monte Carlo prediction CSV, a combined prediction/uncertainty PNG, and appends convolutional scales to `all_conv_sigmas.txt`.

## Current research-code conventions

- The training and validation scripts use different experimental protocols: training uses 1,440-row meal-safe, per-subject windows with an 80/20 window split; validation uses 1,400-row general sliding windows with a 60/40 chronological row split.
- Both Python scripts execute immediately when run or imported; they do not expose command-line arguments or a `main` guard.
- Python random seeds are not fixed, so posterior sampling and optimization results vary between runs.
- Model inputs are used in their raw units without normalization.
- Output paths are relative to the current working directory. Run from the repository root to keep artifacts together.
- `all_conv_sigmas.txt` is append-only; delete or rename it before a clean experiment if results from separate runs should not be mixed.

## License

See [`LICENSE`](LICENSE).
