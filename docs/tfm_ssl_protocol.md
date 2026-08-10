# TFM-SSL Protocol

## Inductive primary track

Allowed during fit / adaptation / PL / graph / retrieval memory:

- labeled-train features + labels
- unlabeled-train features (**no labels**)
- validation features+labels **only** for calibration, early stopping, risk control, guarded fallback

Forbidden:

- test features in adaptation / PL / graph / memory before final predict
- unlabeled true labels in training
- fitting `LabelEncoder` on unlabeled true labels (fixed: labeled∪val only)

## Dual views

- **Raw:** pandas DataFrames with original dtypes → TabPFN-3 / TabICLv2
- **Processed:** shared impute+scale+one-hot → classical / neural / geometric methods

## Transductive exploratory group

Methods in `transductive_exploratory` must **never** enter primary inductive rankings:

- `tabpfn3_distpfn_transductive` (currently precise unsupported status)
- `d2r2_transductive` (instance-wise query prototypes)

## Result writing

Cluster runs write atomic shards:

`results/raw_shards/<wave>/<run_id>.json`

Combine only after validation via `scripts/cluster/collect_wave.py`. Historical CSVs under `results/raw/low_class_wave_paper_methods.csv` are immutable.
