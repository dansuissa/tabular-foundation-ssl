# Final method mapping and shard validity

Audit date: 2026-07-28.

| Existing method name | Intended final name | Current implementation status | Scientific fidelity / required fixes | Prior shards valid for final method? |
|---|---|---|---|---|
| `tabpfn3` | `tabpfn3` | Complete Phase-A grid | Official TabPFN 8.1.0 V3 checkpoint, raw/native view | Yes: 120 successful Phase-A shards |
| `tabiclv2` | `tabiclv2` | Complete Phase-A grid | Official pinned V2 checkpoint, raw/native view | Yes: 120 successful Phase-A shards |
| `tabpfn3_loop_risk` | `tabpfn3_self_training` | Final alias registered | Shared hard-label engine, up to 3 rounds, validation guard/fallback; final logging and GPU smoke still required | No old shard is silently renamed; only new-hash final-name runs count |
| `tabiclv2_loop_risk` | `tabiclv2_self_training` | Final alias registered | Same shared engine and gates as TabPFN-3 | No old shard is silently renamed; only new-hash final-name runs count |
| `tabpfn3_pl_one_shot` | internal ablation | Implemented | One-shot hard pseudo-label context, not the requested iterative final method | No |
| `tabiclv2_pl_one_shot` | internal ablation | Implemented | One-shot hard pseudo-label context, not the requested iterative final method | No |
| `laplacian_mlp` | `laplacian_ssl` | Final alias registered | Explicit sparse graph-regularised neural classifier; final-name smoke/diagnostic coverage required | Existing smoke remains an implementation sanity check only; no final result reuse |
| `retrieval_attention_ssl` | `unlabeled_attention_ssl` | Final alias registered | Labeled-plus-unlabeled training memory by default; final diagnostics/gates required | No final-name result exists |
| `prototype_alignment_ssl` | `embedding_alignment_ssl` | Final alias registered | Class-conditional reliable pseudo-label/prototype alignment; diagnostics need extension before full wave | No final-name result exists |
| `geometric_attention_supervised` | supervised encoder ablation | Implemented | Valid supervised control | Prior diagnostic result is an ablation only |
| `geometric_attention_laplacian` | supervised + Laplacian ablation | Implemented | Valid component ablation | Prior diagnostic result is an ablation only |
| `geometric_attention_retrieval` | supervised + unlabelled attention ablation | Implemented, but old preset used labeled-only memory | Must run labeled-plus-unlabeled variant for the requested ablation | No |
| `geometric_attention_prototype` | supervised + embedding alignment ablation | Implemented | Component ablation; final logging/gates required | Prior diagnostic result is an ablation only |
| `geometric_attention_ssl` | `geometric_attention_ssl` | Modular implementation present | Final preset now genuinely uses labeled-plus-unlabeled attention. Previous labeled-only mitigation did not satisfy the requested method. The known collapse must be re-diagnosed/gated. | No prior labeled-only shard is valid for the final combined method |

The final registered set is exactly:

`tabpfn3`, `tabiclv2`, `tabpfn3_self_training`,
`tabiclv2_self_training`, `laplacian_ssl`,
`unlabeled_attention_ssl`, `embedding_alignment_ssl`, and
`geometric_attention_ssl`.

Other pre-existing experimental methods remain in the source tree for backward
compatibility but are excluded from the focused Phase-B/C configuration.
