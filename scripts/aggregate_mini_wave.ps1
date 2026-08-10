Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

python -m src.aggregate_results `
  --input results/raw/mini_wave.csv `
  --output_dir results/aggregated/mini_wave
