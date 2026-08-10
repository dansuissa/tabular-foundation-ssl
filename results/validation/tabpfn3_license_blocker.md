# TabPFN-3 License / Warm Status

## Secure token status
- Lab path `/private/ofirlin-lab/suissad4/secrets/TABPFN_TOKEN` exists, nonempty, mode `0600`, directory mode `0700`.
- Token value is never printed.
- PriorLabs API: `tabpfn-3-license-v1.0` → **accepted=True**.

## Verified TabPFN-3 identity
- Package: `tabpfn==8.1.0`
- Checkpoint: `tabpfn-v3-classifier-v3_default.ckpt`
- Path: `/private/ofirlin-lab/suissad4/caches/tabpfn/tabpfn-v3-classifier-v3_default.ckpt`
- SHA-256: `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`
- Bytes: `212804803`
- Pointer: `/private/ofirlin-lab/suissad4/caches/tabpfn3_default_path.txt`
- Identity JSON: `/private/ofirlin-lab/suissad4/caches/tabpfn3_identity.json`
- Minimal GPU inference: OK
- Note: do **not** pass `model_path=ModelVersion.V3` (treated as a filename). Use `"auto"` or the local `.ckpt` path.

## Smokes
- Smoke 3 `tabpfn3` phoneme@50: **success** (job `18408127`)
- Smoke 6 `tabpfn3_loop_risk` phoneme@50: **success** (job `18408128`)
