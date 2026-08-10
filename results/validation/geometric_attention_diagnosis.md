# Geometric Attention Diagnosis

Scope: `segment` @ budget=50, seed=0. **Does not block Phase A.**

## Sanity tests

| Test | Result |
|---|---|
| Synthetic multiclass overfit (regs off) | train_acc=0.90, exceeds chance ✓ |
| Segment supervised-only (all regs off, 100 epochs) | train_acc=0.95, test BA≈0.81 ✓ |
| Probabilities finite / sum≈1 | ✓ |
| Class mapping (7 classes) | ✓ |

Conclusion: supervised CE path and label mapping are **not** broken.

## Ablation comparison (identical split)

| method / case | test BA | notes |
|---|---:|---|
| `geometric_attention_supervised` | ~0.67 | healthy |
| `geometric_attention_laplacian` | ~0.68 | healthy |
| `geometric_attention_prototype` | ~0.68 | healthy |
| `geometric_attention_retrieval` (post-fix) | **0.587** | recovered |
| `laplacian_mlp` | ~0.69 | healthy control |
| `geometric_attention_ssl` + `labeled_plus_unlabeled` | **0.145** | chance |
| `geometric_attention_ssl` + `labeled_only` (production) | **0.571** | recovered |
| retrieval-only + unlabeled memory (no regs) | 0.548 | OK |
| retrieval + PL + labeled memory | 0.628 | OK |

Full component ablation: `results/validation/geometric_attention_ssl_ablation.json`.

## Root causes (ordered)

### Bug 1 — frozen retrieval embeddings (fixed)
`class_emb` / `type_emb` were evaluated inside `torch.no_grad()` while building memory tokens → embeddings never trained → retrieval collapsed to chance.

### Bug 2 — prototype tokens on retrieval-only (fixed)
Retrieval ablation always attended to `proto_tokens` even when `use_prototype=False`.

### Bug 3 — device default (fixed)
Default `device: cpu` ignored CUDA; fit now prefers CUDA when present.

### Bug 4 — unlabeled memory × SSL regularizers (mitigated)
With retrieval repaired, **full SSL still collapsed iff `attention_memory=labeled_plus_unlabeled`**.
Unlabeled memory alone (regs off) is fine (~0.55). Adding Laplacian / PL / consistency / prototype terms with an unlabeled bank collapses to chance (~1/7).

**Production mitigation:** `geometric_attention_ssl` defaults to `attention_memory=labeled_only`. Unlabeled rows still enter PL / Laplacian / consistency losses; they are not injected into the retrieval bank under the small-budget smoke regime.

Also hardened `_topk_neighbors` to ignore out-of-range exclude indices (labeled-only bank + unlabeled query offsets).

## Phase C gate

| Criterion | Status |
|---|---|
| `geometric_attention_retrieval` BA ≥ 0.40 on segment@50 | **PASS** (0.587) |
| `geometric_attention_ssl` production BA ≥ 0.40 on segment@50 | **PASS** (0.571) |
| Unlabeled-memory + full regularizers | **experimental / unstable** — do not enable in Phase C waves |

Phase C large waves remain **conditionally unblocked** for the production `labeled_only` SSL default, but must **not** enable `attention_memory=labeled_plus_unlabeled` until a stronger criterion passes.

Artifacts:
- `results/validation/geometric_attention_retest.json`
- `results/validation/geometric_attention_ssl_ablation.json`
