# EVIDENT: Evidence-Based Neural Architecture Selection

EVIDENT is a research implementation of evidence-based neural architecture selection under uncertainty. The example in this repository uses Bayesian temporal convolutional networks (TCNs) for multi-step blood-glucose forecasting in type 1 diabetes.

The repository contains four main stages:

`Bergman simulation → Data preparation → Bayesian TCN architecture search → validation/testing`

The Python files are standalone experiment scripts rather than an installed package or command-line application.

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

CUDA is used automatically when available; otherwise, the model scripts run on the CPU. 

## End-to-end workflow

### 1. Generate synthetic type 1 diabetes data

[`Data/Bergman/bergman_final.m`](Data/Bergman/bergman_final.m) simulates glucose dynamics with a five-state type 1 diabetes Bergman minimal model, a stomach/intestine meal-absorption model, and external basal-plus-bolus insulin delivery. Endogenous glucose-stimulated insulin secretion is disabled, consistent with the type 1 diabetes setting. The model is solved at one-minute intervals.

#### Choose the number of simulation days

Open `bergman_final.m` and change:

```matlab
numDays = 7;
```


#### Meal-size and meal-time uncertainty

The script defines three nominal meals per day:

| Meal | Nominal size | Nominal time from start of day |
|---|---:|---:|
| 1 | 20 g | 0 minutes |
| 2 | 40 g | 240 minutes |
| 3 | 60 g | 540 minutes |

Each day's meal sizes and times are sampled independently:

- Meal-size uncertainty is uniform within ±30% of the nominal amount.
- Meal-time uncertainty is an integer-valued uniform jitter of ±50 minutes.

To change these uncertainty levels, edit the multiplier in the three `D1`, `D2`, and `D3` assignments and the range in the three `randi` calls. 


#### Run the generator

From the repository root:

```bash
cd Data/Bergman
matlab -batch "bergman_final"
```

The generated CSV contains:

| Column | Description |
|---|---|
| `Time` | Minutes from the start of the simulation |
| `Glucose` | Simulated blood glucose concentration in mg/dL |
| `Insulin` | External basal-plus-bolus insulin delivery rate in mU/L/min |
| `MealSize` | Meal mass in mg at the event row and `0` otherwise |

The included CSV contains the default seven-day realization: 10,081 data rows and 21 meal events.

The required columns are:

| Script | Required columns |
|---|---|
| `Bayesian_TCN_training.py` | `Time`, `PID`, `Glucose`, `MealSize`, `Insulin` |
| `Bayesian_TCN_validation.py` | `Time`, `Glucose`, `MealSize`, `Insulin` |


### 2. Train and select a Bayesian TCN architecture

[`Code/Bayesian_TCN_training.py`](Code/Bayesian_TCN_training.py) trains and ranks Bayesian encoder-decoder TCNs. Its main functionality is:

1. Read raw `Glucose`, `MealSize`, and `Insulin` measurements.
2. Build subject-specific, stride-one sequences with 1,440 input rows. The future target is glucose only.
3. Infer the forecast length from the smallest positive gap between meal episodes, falling back to 60 rows when fewer than two meals exist (or one row when no positive gap exists). Forecast windows containing a meal are excluded.
4. Preserve window order and use the first 80% for training and the remaining 20% for testing.
5. Train six predefined encoder/decoder depth and kernel-size candidates.
6. Optimize the observation-noise variance and weight-prior variance using the evidence objective, then rebuild and retrain each candidate with the calibrated values.
7. Draw Monte Carlo posterior-weight samples to calculate prediction means and uncertainty bands.
8. Report RMSE, R², relative L2 error, log predictive likelihood, and log evidence. The candidate with the largest log evidence is reported as the selected architecture.


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


## License

See [`LICENSE`](LICENSE).
