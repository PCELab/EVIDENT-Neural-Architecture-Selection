# EVIDENT: Evidence-Based Neural Architecture Selection

EVIDENT is a research implementation of evidence-based neural architecture selection under uncertainty. Candidate architectures are trained variationally and ranked by their marginal log evidence rather than by validation error alone. The main example uses Bayesian temporal convolutional networks (TCNs) for multi-step blood-glucose forecasting in type 1 diabetes; a Bayesian LSTM on ECG data is included as a second application.

The Python files are standalone experiment scripts rather than an installed package.

## Repository layout

```text
EVIDENT-Neural-Architecture-Selection/
├── codes/
│   ├── Bergman_glucose_model/
│   │   ├── bergman_final.m                       # synthetic T1D glucose simulator (MATLAB)
│   │   └── t1d_basal_bolus_meal_aligned.csv      # included 7-day realization
│   ├── Bayesian_TCN/
│   │   ├── Bayesian_TCN_population_training.py   # evidence-based architecture search
│   │   └── Bayesian_TCN_validation.py            # subject-specific adaptation + testing
│   ├── Optuna_search/
│   │   └── optuna_tcn_ohio.py                    # Optuna-driven evidence search (OhioT1DM)
│   └── Bayesian_LSTM/
│       └── bayes_lstm_ecg.py                     # Bayesian LSTM ECG forecasting
├── papers/            # reference literature, grouped by topic
├── reports/           # manuscript sources, figures, and submission material
├── slides/            # proposal-defense slides (LaTeX sources and figures)
├── LICENSE
└── README.md
```

## Requirements

The Python experiments require Python 3 and:

- PyTorch (`torch.func` is used by the LSTM script, so PyTorch 2.0 or newer)
- NumPy
- pandas
- scikit-learn
- Optuna (only for `optuna_tcn_ohio.py`)
- Matplotlib (only for `bayes_lstm_ecg.py`)

CUDA is used automatically when available; otherwise the scripts run on the CPU.

The synthetic-data generator requires MATLAB. It uses `ode45` and local functions in a script, so MATLAB R2016b or newer is recommended.

## Data

| Script | Default data file | Required columns | Included in repo |
|---|---|---|---|
| `bergman_final.m` | — (generates data) | — | output CSV included |
| `Bayesian_TCN_population_training.py` | `diabetes_data_insulin.csv` (UVA/Padova population) | `PID`, `Glucose`, `MealSize`, `Insulin` | no |
| `Bayesian_TCN_validation.py` | `diabetes_data_insulin.csv` | `PID`, `Glucose`, `MealSize`, `Insulin` | no |
| `optuna_tcn_ohio.py` | `population_ohio.csv` (OhioT1DM) | `PID`, `Time`, `Glucose`, `MealSize`, `Insulin` | no |
| `bayes_lstm_ecg.py` | `patients_basic/ecg_patient_06.csv` | `ecg_mV`, `ecg_clean_mV`, `time_s`, `t_since_R_s`, `rr_last_s`, `resp_belt_au` | no |

The clinical/population datasets are not redistributed here; point each script at your own copy with its `--data` / `--csv` flag. The Bergman CSV has no `PID` column, so it is a single-subject substitute and needs a `PID` added before use with the population scripts.

## End-to-end workflow

### 1. Generate synthetic type 1 diabetes data

[`codes/Bergman_glucose_model/bergman_final.m`](codes/Bergman_glucose_model/bergman_final.m) simulates glucose dynamics with a type 1 diabetes Bergman minimal model, a stomach/intestine meal-absorption model, and external basal-plus-bolus insulin delivery. Endogenous glucose-stimulated insulin secretion is disabled, consistent with the type 1 diabetes setting. The system is solved with `ode45` (absolute and relative tolerances `1e-8`) and sampled at one-minute intervals.

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

To change these uncertainty levels, edit the multiplier in the three `D1`, `D2`, and `D3` assignments and the range in the three `randi` calls. The random stream is reset with `rng('default')`, so the default run is reproducible.

#### Run the generator

From the repository root:

```bash
cd codes/Bergman_glucose_model
matlab -batch "bergman_final"
```

This writes `t1d_basal_bolus_meal_aligned.csv`:

| Column | Description |
|---|---|
| `Time` | Minutes from the start of the simulation |
| `Glucose` | Simulated blood glucose concentration in mg/dL |
| `Insulin` | Plasma insulin concentration in mU/L |
| `MealOnset` | `1` on the minute a meal begins, `0` otherwise |
| `MealSize` | Meal mass in mg at the event row and `0` otherwise |

The included CSV is the default seven-day realization: 10,081 data rows and 21 meal events.

### 2. Population training and evidence-based architecture search

[`codes/Bayesian_TCN/Bayesian_TCN_population_training.py`](codes/Bayesian_TCN/Bayesian_TCN_population_training.py) trains and ranks Bayesian encoder-decoder TCNs (Bayes-by-backprop `Conv1d` layers) across a population of subjects.

```bash
python codes/Bayesian_TCN/Bayesian_TCN_population_training.py \
    --data diabetes_data_insulin.csv --fold 1 --level 1 --out evident_population
```

What it does:

1. Reads `[Glucose, MealSize, Insulin]` per subject and splits subjects patient-wise using the fixed folds `1: (9, 10)`, `2: (6, 7)`, `3: (1, 2)`, `4: (3, 4)`, `5: (5, 8)`. The listed patients are held out; the rest are used for training.
2. Builds stride-one sequences with 1,440 input rows. The target is future glucose only.
3. Infers the forecast length from the smallest positive gap between meal episodes, falling back to 60 rows when fewer than two meal episodes exist (or one row when no positive gap exists). Windows whose forecast horizon contains a meal are discarded ("meal-safe" windowing).
4. Sweeps a capacity level of candidate architectures (`--level 1/2/3`, 10/9/10 candidates), each defined by an encoder channel list, a mirrored decoder, a kernel size, and a dilation base. Individual candidates can be selected with `--archs L2A4 L2A7`.
5. Trains each candidate in two stages: an ELBO stage, then optimization of the observation-noise variance and the weight-prior variance against the negative log evidence, then a retraining stage with the calibrated values.
6. Draws 50 Monte Carlo posterior-weight samples for prediction means and uncertainty bands.
7. Reports RMSE, R², relative L2 error, log posterior predictive likelihood, and log evidence.

Per-architecture outputs land in `<out>/fold<F>/level<L>/<ARCH_ID>/`:

| File | Contents |
|---|---|
| `summary.json` | architecture, parameter count, calibrated variances, metrics, fold membership |
| `posterior_params__<tag>.npz` | variational posterior (`mu`, `rho`) for every layer |
| `evidence_constants__<tag>.npz` | terms entering the log-evidence computation |
| `mc_predictions__<tag>.csv` | per-window Monte Carlo mean, variance, and ±2σ bounds |

`<out>/fold<F>/level<L>/level_summary.csv` ranks the level by log evidence, and the top row is reported as the selected architecture.

### 3. Subject-specific adaptation and testing

[`codes/Bayesian_TCN/Bayesian_TCN_validation.py`](codes/Bayesian_TCN/Bayesian_TCN_validation.py) adapts the population-selected architecture to each held-out subject. It is not checkpoint-only inference: the population posterior is used as an informative prior and the model is recalibrated and retrained per subject.

```bash
python codes/Bayesian_TCN/Bayesian_TCN_validation.py \
    --data diabetes_data_insulin.csv --fold 1 --level 1 \
    --population evident_population --out evident_validation
```

What it does:

1. Reads `level_summary.csv` for the given fold and level and picks the architecture with the highest log evidence, unless one is named with `--arch`. It aborts if the fold or the held-out patient list disagrees with the population run.
2. Re-creates chronological sliding windows per held-out subject with the population input length (1,440 rows) and a fixed 60-row forecast horizon. No meal-safe filtering is applied here.
3. Splits each subject's series chronologically at 60%, keeping enough pre-split context for the first test window.
4. Warm-starts from `posterior_params__*.npz` of the population run, then runs the same two-stage fit and variance calibration.
5. Writes the same per-subject artifacts as the training stage under `<out>/fold<F>/level<L>/<ARCH_ID>/patient_<PID>/`, plus a `validation_summary.csv` aggregating RMSE, R², relative L2 error, log predictive likelihood, and log evidence across held-out subjects.

### 4. Optuna baseline search on OhioT1DM

[`codes/Optuna_search/optuna_tcn_ohio.py`](codes/Optuna_search/optuna_tcn_ohio.py) runs the same evidence objective inside an Optuna study, over architectures grouped into parameter-count categories `C1`–`C8` (from under 2,500 up to 300,000 weights). Architectures that cannot realize the configured window lengths are skipped before the study starts.

```bash
python codes/Optuna_search/optuna_tcn_ohio.py \
    --data population_ohio.csv --category C4 --sampler grid
```

- Windows: 144 input samples, 12-minute forecast horizon, 80/20 per-patient chronological split, training stride 4 and test stride 6.
- `--sampler grid` evaluates every feasible architecture in the category once; `--sampler tpe --n-trials N` samples instead. `--storage sqlite:///study.db` persists and resumes the study.
- Each trial maximizes the marginal log evidence (with the Occam log-determinant term) and also records test RMSE.
- Outputs: `tcn_<category>_glucose60_evidence_results.csv`, updated after every trial, and checkpoints in `tcn_<category>_glucose60_evidence_models/`.

### 5. Bayesian LSTM on ECG

[`codes/Bayesian_LSTM/bayes_lstm_ecg.py`](codes/Bayesian_LSTM/bayes_lstm_ecg.py) applies the same variational machinery to a Bayesian LSTM that forecasts 0.25 s of ECG (64 samples at 256 Hz) from the preceding 1 s of ECG plus beat-phase, RR-interval, and respiration-belt channels.

```bash
python codes/Bayesian_LSTM/bayes_lstm_ecg.py --csv patients_basic/ecg_patient_06.csv
```

It splits the record 80/20 chronologically, draws 100 posterior samples at test time, and separates epistemic from aleatoric uncertainty. It reports RMSE against both the noisy measurement and the clean reference signal, compares against persistence and last-beat baselines, and checks 50%/68%/95% interval coverage. Outputs are `bayes_lstm_ecg.pt`, `bayes_lstm_test.npz`, `bayes_lstm_test_predictions.csv`, `bayes_lstm_forecast.png`, and `bayes_lstm_rmse_vs_lead.png`.

## Supporting material

- `papers/` — reference literature grouped by topic: `tcn`, `enc_decoder`, `architecture_search`, `Machine_learning_models`, `Blood_glucose_prediction`, `ECG_data`.
- `reports/manuscript_evident/` — LaTeX sources, references, and figures for the manuscript.
- `reports/neuralnetworks_submit_6_26/` — cover letter and highlights for the journal submission.
- `slides/Azharul_proposal_defense/` — proposal-defense slides with their LaTeX template and figures.

## License

See [`LICENSE`](LICENSE).
