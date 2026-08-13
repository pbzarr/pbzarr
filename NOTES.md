# 08-12-26
- import bed default logic:

| sources | value fields | result | required flags |
|---|---|---|---|
| 1 | 1 (BED4) | scalar `(position,)` track | `--track` (headerless) |
| 1 | 2+ | one 2D `(position, n_fields)` track; field names are the column labels | `--track` and `--column-dim` |
| 2+ | 1 | one 2D `(position, n_sources)` track; source labels are the column labels | `--track` (headerless) and `--column-dim` |
| 2+ | 2+ | one 2D `(position, n_sources)` track *per field*, named after the field | `--column-dim` |

- pipeline fns = fixed (readers x fields) -> (tracks x columns) mappings:

| fn | in | out |
|---|---|---|
| `run_pipeline<T>` | N readers, 1 value | 1 track, N cols (typed/binary: d4, bigwig, stack) |
| `run_multi_pipeline` | 1 reader, K fields | K scalar tracks |
| `run_matrix_pipeline` | N readers, K fields | K 2D tracks, N cols |
| `run_wide_pipeline` | 1 reader, K fields | 1 2D track, K cols |

- TODO: dedupe worker-pool scaffolding (channel/scope/State/fork/drain)
