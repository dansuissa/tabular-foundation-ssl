# TFM-SSL Fidelity Notes

Do not claim paper fidelity unless the implementation matches.

| Method | Claimed fidelity | Deviations / notes |
|---|---|---|
| `tabpfn3` / `tabiclv2` | official | Requires licensed/package installs; `allow_auto_download=False` on compute jobs |
| `*_loop_risk` | novel_experimental | Hard-label loops only; **not** a soft-label LoopTabFM reproduction |
| `*_cast` / `cast_*` | novel_experimental | CAST-style class-conditional density confidence adjustment adapted to teachers/trees |
| `stunt` | official adapter | Uses official STUNT algorithm with benchmark fixed splits; see module `UPSTREAM_COMMIT` and `training_meta` diffs |
| `seba` | official or faithful_reimplementation | Prefers `kacper3615/SeBA`; otherwise Separated-at-Birth NN alignment reimplementation — **not** generic contrastive SSL |
| `d2r2_c` | paper_core (inductive) | Mean support prototypes only; no query-set prototype refinement |
| `d2r2_transductive` | exploratory | Separately named; excluded from inductive ranking |
| `tabpfn3_distpfn_transductive` | unsupported | Refuses silent heuristic substitution (`unsupported_faithful_distpfn_unavailable`) |
| `tabpfn3_unlabeled_prior_adjustment` | novel_experimental | Inductive unlabeled-pool prior adjustment; not DistPFN |
| Geometric / Laplacian / retrieval / prototype | novel_experimental | Research implementations with leakage-safe train-pool graphs/memory |
| TFM adapters | novel_experimental | Identity guard falls back to frozen TFM on val regression; TabICL embedding adapters only if API probe passes |

Always read `training_meta.method_fidelity` and `training_meta.reference_family` in result shards.
