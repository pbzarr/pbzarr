# 08-12-26
- import bed default logic:

| sources | value fields | result | required flags |
|---|---|---|---|
| 1 | 1 (BED4) | scalar `(position,)` track | `--track` (headerless) |
| 1 | 2+ | one 2D `(position, n_fields)` track; field names are the column labels | `--track` and `--column-dim` |
| 2+ | 1 | one 2D `(position, n_sources)` track; source labels are the column labels | `--track` (headerless) and `--column-dim` |
| 2+ | 2+ | one 2D `(position, n_sources)` track *per field*, named after the field | `--column-dim` |

