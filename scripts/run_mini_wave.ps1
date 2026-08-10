Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

python -m src.run_benchmark `
  --datasets_subset phoneme spambase letter `
  --methods logistic_regression random_forest xgboost lightgbm catboost mlp label_spreading label_propagation self_training_lr self_training_xgboost self_training_lightgbm self_training_catboost rpl_lr rpl_lite_xgboost `
  --label_budgets 50 100 250 500 `
  --seeds 0 1 2 `
  --output results/raw/mini_wave.csv `
  --resume
