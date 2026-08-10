# TFM-SSL Resource Plan

Derived from measured profile runs (not guesses). Safety margin ≈ 2.5× RAM, 3× wall time.

## Measured cells

| dataset | method | budget | status | total_s | peak_rss_mb | peak_vram_mb | suggested_profile | mem | time |
|---|---|---:|---|---:|---:|---:|---|---|---|
| phoneme | logistic_regression | 50 | success | 0.152 | 549.3 | 0.0 | cpu_light | 16G | 1:00:00 |
| phoneme | sslae | 50 | success | 3.323 | 686.4 | 0.0 | gpu_representation | 64G | 1:00:00 |
| phoneme | tabiclv2 | 50 | success | 1.953 | 1300.9 | 355.0 | gpu_tfm_small | 64G | 1:00:00 |
| segment | laplacian_mlp | 50 | success | 0.324 | 1350.6 | 8.1 | gpu_representation | 64G | 1:00:00 |
| segment | geometric_attention_ssl | 50 | success | 0.431 | 1389.6 | 8.1 | gpu_representation | 64G | 1:00:00 |
| adult | tabiclv2 | 100 | success | 2.722 | 1579.0 | 3649.3 | gpu_tfm_large | 96G | 4:00:00 |
| jannis | tabiclv2 | 100 | success | 8.377 | 1745.8 | 19354.0 | gpu_tfm_large | 96G | 4:00:00 |
| jannis | stunt | 50 | success | 20.664 | 1791.6 | 614.3 | gpu_tfm_large | 96G | 4:00:00 |

## Profile definitions

| profile | partition | gres | default cpus | role |
|---|---|---|---:|---|
| cpu_light | cpu192G-48h | — | 4 | LR / tiny classical |
| cpu_tree_ssl | cpu192G-48h | — | 8 | GBDT + self-training / CAST trees |
| gpu_tfm_small | p_uriofir | gpu:1 | 8 | TabPFN/TabICL on small/medium data |
| gpu_tfm_large | p_uriofir | gpu:1 | 8 | adult/jannis-scale TFM |
| gpu_representation | p_uriofir | gpu:1 | 8 | STUNT/SeBA/Laplacian/geometric |
| gpu_diffusion | p_uriofir | gpu:1 | 8 | reserved for heavy diffusion-like methods |

## Concurrency

`p_uriofir` has 2×A100. Default array concurrency: **2** for GPU waves.
CPU waves may use higher concurrency on `cpu192G-48h`.

## Notes

- Pre-cache models/datasets before array submission.
- Do not submit multiple jobs that race the same first-time checkpoint download.
- Update this file when new profile measurements supersede prior ones.

