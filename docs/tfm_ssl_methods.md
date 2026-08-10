# TFM-SSL Methods

Registry of foundation-model and modern SSL methods added to `ssl_tabular_benchmark`.

## Frozen TFMs (`tfm_frozen`)

| Method | View | Notes |
|---|---|---|
| `tabpfn3` | raw | Official TabPFN-3 default classifier; specialized binary/multiclass checkpoints are ablations only |
| `tabiclv2` | raw | Official TabICLv2 (`tabicl-classifier-v2-20260212.ckpt`); KV cache logged |

## TFM SSL core (`tfm_ssl_core`)

| Method | Fidelity | Summary |
|---|---|---|
| `tabpfn3_pl_one_shot` / `tabiclv2_pl_one_shot` | novel_experimental | Calibrated class-balanced one-shot PL → rebuild context once |
| `tabpfn3_loop_risk` / `tabiclv2_loop_risk` | novel_experimental | Hard-label loops 0–3 with val BA+log-loss guard; **not** soft LoopTabFM |
| `tabpfn3_cast` / `tabiclv2_cast` | novel_experimental | Shared CAST density adjuster on TFM teachers |
| `tfm_consensus_context_tabiclv2` | novel_experimental | Agreement of TabPFN-3+TabICLv2 → TabICL context |
| `tabpfn3_teacher_catboost` / `tabiclv2_teacher_catboost` | novel_experimental | TFM teacher → weighted CatBoost student + val fallback |
| `tfm_consensus_catboost` | novel_experimental | Consensus PL → CatBoost |

## Label-shift

| Method | Protocol | Status |
|---|---|---|
| `tabpfn3_unlabeled_prior_adjustment` | inductive | Novel prior adjustment from unlabeled train pool only |
| `tabpfn3_distpfn_transductive` | transductive | **Unsupported** until faithful DistPFN code is available |

## Geometric / Laplacian (`geometric_ssl_ablation`)

Shared components; ablations disable loss terms rather than duplicating code:

- `laplacian_linear`, `laplacian_mlp`
- `prototype_alignment_ssl`
- `retrieval_attention_ssl`
- `geometric_attention_{supervised,laplacian,prototype,retrieval,ssl}`

## TFM adapters (`tfm_geometric`)

Frozen TFM embeddings/probabilities + trainable geometric heads. TabICL hidden embeddings only if API probe succeeds; otherwise separately named `tabiclv2_predfeat_*` methods.

## Modern non-TFM (`modern_ssl`)

| Method | Fidelity | Upstream |
|---|---|---|
| `cast_catboost` / `cast_lightgbm` | novel_experimental | Shared CAST core |
| `stunt` | official adapter | `jaehyun513/STUNT` (pinned commit in module) |
| `seba` | official or faithful_reimplementation | `kacper3615/SeBA` when available |
| `d2r2_c` | paper_core inductive | Mean-support prototypes; no query/test prototype updates |

See `docs/tfm_ssl_fidelity.md` for deviations.
