# Previously Missing Method Capabilities

Explicit table for the four capabilities that were missing from `METHOD_CAPABILITIES` during the first validation pass. They are **now registered**. Phase A requires only `tabpfn3` and `tabiclv2`.

| exact method / capability | registry name | reason unavailable (originally) | missing dependency / API / impl | blocks Phase A | blocks Phase B | blocks Phase C | can fix now | registry returns unsupported (not false success) |
|---|---|---|---|---|---|---|---|---|
| Inductive D2R2-C alias | `d2r2c_inductive` | Was absent from capability map | none — alias of inductive `d2r2_c` | No | No | No (needed for C only if selected) | Yes — registered | Builds real method; not a silent fake |
| Transductive D2R2 | `d2r2_transductive` | Transductive IP path not enabled for inductive ranking | Faithful D2R2 IP / `allow_transductive` | No | No | No (exploratory only) | Partial — raises unless explicitly allowed | Yes — `unsupported_d2r2_ip_not_enabled` |
| TabICLv2 predfeat Laplacian adapter | `tabiclv2_predfeat_laplacian_adapter` | Was unregistered | ssl-tfm + warmed TabICLv2 + geometric stack | No | No | Yes if selected in C | Yes — registered builder | Builds via `build_tfm_adapter`; not false success |
| TabICLv2 predfeat geometric | `tabiclv2_predfeat_geometric` | Was unregistered | ssl-tfm + warmed TabICLv2 + geometric stack | No | No | Yes if selected in C | Yes — registered builder | Builds via `build_tfm_adapter`; not false success |

None of these block Phase A.
