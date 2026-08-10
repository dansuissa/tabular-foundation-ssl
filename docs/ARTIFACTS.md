# External research artifacts

The Git repository contains the complete source, canonical CSVs, derived
results, plots, reports, manifests, and validation evidence. High-volume cluster
execution records are packaged separately so they do not make ordinary clones
noisy or dependent on site-specific paths.

## Full cluster artifact archive

| Field | Value |
|---|---|
| Filename | `tabular-foundation-ssl-full-cluster-artifacts.tar.gz` |
| SHA-256 | `5e402aef4413d6393378ab180fea5688c889e693fb03415a459744c58834dcab` |
| Archive members | 3,334 |
| Compressed size | approximately 840 KiB |
| Source footprint | approximately 88 MiB |

The archive contains:

- every atomic JSON result shard and task map under `results/raw_shards/`;
- Slurm standard output/error logs for final waves, gates, repairs, and smoke
  tests under `results/logs/`;
- the shared `results/combined/` directory as it existed at packaging time.

It covers the frozen TFM screen, Phase B diagnostics, both halves of the final
Phase C grid, attention/combined repair and final gates, and the earlier smoke
matrix. It does **not** include Python environments, OpenML caches, licensed
model checkpoints, or secrets.

Verify and extract on Linux/macOS:

```bash
sha256sum tabular-foundation-ssl-full-cluster-artifacts.tar.gz
tar -tzf tabular-foundation-ssl-full-cluster-artifacts.tar.gz | less
tar -xzf tabular-foundation-ssl-full-cluster-artifacts.tar.gz
```

Verify on Windows PowerShell:

```powershell
Get-FileHash .\tabular-foundation-ssl-full-cluster-artifacts.tar.gz -Algorithm SHA256
tar -tzf .\tabular-foundation-ssl-full-cluster-artifacts.tar.gz
```

The archive should be uploaded as a private GitHub release asset after the
initial repository push. The adjacent `.sha256` file allows verification before
extraction.
