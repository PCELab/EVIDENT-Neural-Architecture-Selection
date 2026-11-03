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

## Data

| Data | Included in repo |
|---|---|
|  Bergman in silico patient data | `bergman_final.m` code included for reproducibility  |
|  UVA/Padova software data | no |
|  OhioT1DM patients data | no |
|  Synthetic ECG data | Paper included for reproducibility |


## End-to-end workflow

### 1. Generate synthetic type 1 diabetes data

[`codes/Bergman_glucose_model/bergman_final.m`](codes/Bergman_glucose_model/bergman_final.m) simulates glucose dynamics with a type 1 diabetes Bergman minimal model, a stomach/intestine meal-absorption model, and external basal-plus-bolus insulin delivery. Endogenous glucose-stimulated insulin secretion is disabled, consistent with the type 1 diabetes setting. The system is solved with `ode45` (absolute and relative tolerances `1e-8`) and sampled at one-minute intervals.

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

For each patient-wise fold, the script trains every candidate architecture in the selected capacity level on the patients' 1440-row windows, calibrates the observation-noise and weight-prior variances against the negative log evidence, and ranks the candidates of that level by their marginal log evidence.


### 3. Subject-specific adaptation and testing

[`codes/Bayesian_TCN/Bayesian_TCN_validation.py`](codes/Bayesian_TCN/Bayesian_TCN_validation.py) adapts the population-selected architecture to each held-out subject. It is not checkpoint-only inference: the population posterior is used as an informative prior and the model is recalibrated and retrained per subject.

For each held-out subject of the fold, the script takes the architecture with the highest log evidence from the population run's `level_summary.csv`, warm-starts it from that run's posterior as an informative prior, adapts it on the subject's first four days of data, and validates 60-minute-ahead forecasts on the remaining day.

### 4. Optuna baseline search on OhioT1DM

[`codes/Optuna_search/optuna_tcn_ohio.py`](codes/Optuna_search/optuna_tcn_ohio.py) runs the same evidence objective inside an Optuna study, over architectures grouped into parameter-count categories `C1`–`C8`. 

For a chosen capacity category, the script searches every feasible architecture in that category with Optuna, fitting each on an 80/20 per-patient chronological split of 144-sample windows with a 12-minute horizon, and that maximizes the marginal log evidence.

### 5. Bayesian LSTM on ECG

[`codes/Bayesian_LSTM/bayes_lstm_ecg.py`](codes/Bayesian_LSTM/bayes_lstm_ecg.py) applies the same variational machinery to a Bayesian LSTM that forecasts 0.25 s of ECG (64 samples at 256 Hz) from the preceding 1 s of ECG plus beat-phase, RR-interval, and respiration-belt channels.

It splits the record 80/20 chronologically, draws 100 posterior samples at test time, and separates epistemic from aleatoric uncertainty. It reports RMSE against both the noisy measurement and the clean reference signal, compares against persistence and last-beat baselines, and checks 50%/68%/95% interval coverage.

## Supporting material

- `papers/` — reference literature grouped by topic: `tcn`, `enc_decoder`, `architecture_search`, `Machine_learning_models`, `Blood_glucose_prediction`, `ECG_data`.
- `reports/manuscript_evident/` — LaTeX sources, references, and figures for the manuscript.
- `reports/neuralnetworks_submit_6_26/` — cover letter and highlights for the journal submission.
- `slides/Azharul_proposal_defense/` — proposal-defense slides with their LaTeX template and figures.

## License

See [`LICENSE`](LICENSE).
