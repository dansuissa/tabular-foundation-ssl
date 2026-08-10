# Security and restricted artifacts

This private repository must not contain access tokens, model-portal
credentials, SSH keys, downloaded checkpoints, or private dataset exports.

- Store TabPFN and Hugging Face credentials in the lab's restricted secret
  storage, never in YAML, shell scripts, notebooks, logs, or result payloads.
- Keep model checkpoints in a private cache and identify them in results by
  filename and SHA-256 only.
- Treat raw scheduler logs as restricted research artifacts; review them for
  secrets before creating a release asset.
- Run `python scripts/verify_repository.py` before every push. CI repeats its
  high-risk credential-pattern scan.

If a secret is committed, revoke or rotate it first. Removing it from the latest
commit is not sufficient because it remains in Git history. Notify the
repository owner privately rather than opening a public issue.
