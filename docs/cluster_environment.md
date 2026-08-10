# Cluster Environment Report (Bar-Ilan / DSI)

**Reconnaissance date:** 2026-07-22  
**Auditor login user:** `suissad4` (uid 41458)  
**Command session hostname:** `dsiuriofir01.local.biu.ac.il`  

Only facts verified by commands on this session are stated as confirmed. Uncertainties are listed explicitly.

---

## 1. Verified roles of `dsihead` and `dsiuriofir01`

| Host | Verified facts | Role interpretation |
|---|---|---|
| **dsiuriofir01** | Current shell host; Rocky Linux 9.8; 128× Xeon Gold 6338; 502 GiB RAM; **2× NVIDIA A100 80 GB PCIe**; listed in Slurm as `NodeName=dsiuriofir01`; partitions `A100-4h`, `p_uriofir`; `CPUAlloc=0` while interactive; `Gres=gpu:a100:2` | **Slurm compute / GPU node** owned by Uri Ofir partition family. It is **not** merely a login-only host. |
| **dsihead** | Resolves via DNS: `192.168.2.1 dsihead.local.biu.ac.il`. SSH from this session: `Permission denied (publickey,password,keyboard-interactive)`. **Not** present as a Slurm node (`scontrol show node dsihead` → not found). | Almost certainly the **cluster login / head entry host** users SSH to first. Full interactive verification of login-node policy was **not** possible from this session without credentials/keys. |

**Important operational note:** This Cursor session is already attached to `dsiuriofir01` **outside an active Slurm allocation** (`CPUAlloc=0`, no `SLURM_JOB_ID`). Heavy training or large downloads should still be submitted with `sbatch`/`salloc` on an appropriate partition rather than assuming interactive GPU node use is policy-compliant.

---

## 2. Scheduler

**Type: Slurm** (verified).

| Tool | Path / status |
|---|---|
| `sbatch`, `srun`, `squeue`, `sinfo` | Present under `/usr/bin` |
| `scontrol` | Works; cluster version slurmd reports **24.11.5** |
| `SLURM_CONF_SERVER` | `slurmctl1:6817` |
| PBS (`qsub`/`qstat`) | Not found |
| LSF (`bsub`/`bjobs`) | Not found |

Accounts for `suissad4` (`sacctmgr`):

- `ug_dsi`
- `ug_lindenbaum` (default)
- `ug_uri_ofir`

---

## 3. Partitions / queues relevant to this project

Verified with `sinfo` / `scontrol show partition`:

| Partition | Timelimit | Nodes (A100-related) | Notes |
|---|---|---|---|
| **`p_uriofir`** | **UNLIMITED** | `dsiuriofir01` only (2×A100 80 GB) | `AllowAccounts=ug_uri_ofir`. Preferred home partition for this user/lab node. |
| **`A100-4h`** | **4:00:00** | `dsiasaf01` (4×A100), `dsiuriofir01` (2×A100), `hpc2a100-01` (2×A100) | `AllowAccounts=ALL`. Shared A100 pool. |
| `generic` | 4:00:00 | several GPU nodes | Default partition (`*`) |
| `cpu192G-48h`, `cpu512G-48h`, `cpu1T-24h` | 24–48 h | CPU nodes | Good for classical sklearn / aggregation |
| H200 / L4 / B200 partitions | 4–12 h | other accelerators | Available cluster-wide; **not required** for phase-0 TabPFN/TabICL plan |

**Recommended default for TFM GPU jobs:**  
`-A ug_uri_ofir -p p_uriofir --gres=gpu:1` (or `gpu:a100:1`).

**Recommended for classical CPU waves:**  
`-A ug_uri_ofir` or `ug_dsi` on a CPU partition (`cpu192G-48h`) to avoid occupying A100s.

CPU/memory limits: partitions show `DefMemPerNode=UNLIMITED`, `MaxMemPerNode=UNLIMITED`. Practical limit is node RAM (~489 GB reported by Slurm on `dsiuriofir01`). Always request explicit `--cpus-per-task` and `--mem` to avoid oversubscription.

---

## 4. GPU / CUDA

On `dsiuriofir01` (`nvidia-smi`):

- **2× NVIDIA A100 80GB PCIe**, persistence mode on
- Driver **610.43.02**, CUDA UMD **13.3** (driver capability)
- Compute capability **8.0**
- Toolkit on node: `/usr/local/cuda` → **CUDA 12.3.2** (`nvcc` V12.3.107)
- No processes using GPU at audit time

**Implication for PyTorch wheels:** install a CUDA 12.x build compatible with driver 610 (e.g. cu121/cu124 family as pinned during bootstrap validation). Do not assume the toolkit major equals the pip wheel tag until `torch.cuda.is_available()` is tested inside the env.

---

## 5. CPU / memory / OS

- **OS:** Rocky Linux 9.8 (Blue Onyx), kernel 5.14.0-687.el9
- **CPU:** 2 sockets × 32 cores × 2 threads = **128 CPUs**, Xeon Gold 6338 @ 2.00 GHz, 2 NUMA nodes
- **RAM:** ~502 GiB host / Slurm RealMemory 489147 MB
- **Local root disk:** ~1.8 TB (`/`), ~1.3 TB free — usable for `/tmp` scratch during jobs

---

## 6. Storage locations and quotas

| Path | Type | Verified access | Role |
|---|---|---|---|
| `/home/eng/suissad4` | NFS (`psa...:/ifs/a/home`) | RW; **~200 G** visible quota; ~1.1 G used | Code, docs, small artifacts |
| `/private/ofirlin-lab/suissad4/` | NFS lab share | **RW verified** (`touch` succeeded); dirs created | **Recommended** envs, caches, shard results, large logs |
| `/private/shared` | NFS | Readable listing | Shared datasets/models (others’ content); not assumed writable |
| `/tmp` | Local root FS | Large free space | Job-local scratch; not durable |

**Recommendations (phase-0):**

```text
SSL_ROOT=/home/eng/suissad4/projects/ssl-foundation-models/ssl_tabular_benchmark
LAB_ROOT=/private/ofirlin-lab/suissad4
ENVS=$LAB_ROOT/envs
CACHES=$LAB_ROOT/caches
RESULTS_LAB=$LAB_ROOT/results
```

Do **not** put multi-GB checkpoints or conda/micromamba packages in `$HOME` if avoidable.

---

## 7. Software stack availability

| Component | Status |
|---|---|
| System Python | **3.9.25** only (`python3.10+` **not** on PATH) |
| System packages | No torch / sklearn / xgboost in bare system Python |
| Conda | `/usr/bin/conda` **4.14.0**; **base env is `/usr` read-only**; user envs default to `~/.conda/envs` |
| mamba / micromamba / uv | **Not installed** for this user |
| Environment modules | Lmod present; only MPI modules + lmod/settarg visible in `module avail` |
| Singularity CE | **4.5.0-1.el9** available |
| Internet on this node | **Yes** — HTTPS to `pypi.org` and `huggingface.co` succeeded; ICMP to 8.8.8.8 failed (ICMP may be filtered; not conclusive) |

**Environment decision implication:** bootstrap a **user-space micromamba** (or miniforge) under `/private/ofirlin-lab/suissad4/envs/` creating Python **3.11** environments. Do not install into the read-only system conda base. Do not rely on system Python 3.9 for TabPFN-3 / TabICLv2 (both require ≥3.10).

---

## 8. Auth / tokens (presence only — values never printed)

| Secret | Status |
|---|---|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | **Unset** |
| `TABPFN_TOKEN` | **Unset** |
| `~/.cache/huggingface/token` | **Absent** |
| PriorLabs browser cache | **Absent** |

**Blocker for TabPFN-3 weight download:** license acceptance + `TABPFN_TOKEN` (or interactive PriorLabs login) required for headless jobs. TabICL checkpoints via Hugging Face may need `HF_TOKEN` depending on gate status—warm-cache script must fail clearly if missing.

---

## 9. Network / offline compute assumption

- This GPU node **can** reach PyPI and Hugging Face today.
- Other compute nodes were **not** probed. Assume some worker nodes may lack egress.
- Design: warm caches **once** on an internet-capable node under `$LAB_ROOT/caches`, then point jobs at those caches via env vars; jobs must not stampede downloads.

---

## 10. Exact job-submission pattern (Slurm)

GPU single run (template):

```bash
sbatch \
  -A ug_uri_ofir \
  -p p_uriofir \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=12:00:00 \
  --job-name=ssl-tfm \
  --output=/private/ofirlin-lab/suissad4/results/logs/%x-%j.out \
  --error=/private/ofirlin-lab/suissad4/results/logs/%x-%j.err \
  scripts/cluster/run_single.sh <args>
```

Array wave (template):

```bash
sbatch --array=0-N%16 ... scripts/cluster/run_single.sh --task-id ${SLURM_ARRAY_TASK_ID}
```

Inside jobs always export:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

Never launch heavy work on `dsihead` (login). Prefer `sbatch` even when already on `dsiuriofir01`.

---

## 11. Uncertainties (not verified)

1. Full interactive behavior / MOTD / fair-use policy text on **dsihead** (SSH denied).
2. Whether interactive GPU use without allocation is tolerated long-term.
3. Internet egress from **all** A100-4h nodes (`dsiasaf01`, `hpc2a100-01`).
4. Lab-wide quota soft/hard limits on `/private/ofirlin-lab` beyond successful write probe.
5. Whether `p_uriofir` has unspoken per-user GPU concurrency caps (QoS not fully enumerated beyond partition AllowAccounts).
6. Exact PriorLabs / HF gated-model requirements until first authenticated warm-cache run.
7. Project is **not** a git repo yet — commit hashing deferred.

---

## 12. Recommended directories (created during reconnaissance)

```text
/private/ofirlin-lab/suissad4/envs/
/private/ofirlin-lab/suissad4/caches/{hf,torch,tabpfn,tabicl,openml,pip,uv,tmp}
/private/ofirlin-lab/suissad4/results/{raw_shards,logs,combined}
/private/ofirlin-lab/suissad4/tmp/
```
