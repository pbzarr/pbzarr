"""Stack scalar PBZ tracks into a new destination collection."""

import pbzarr


pbzarr.stack(
    [("sample-a.pbz", "A"), ("sample-b.pbz", "B")],
    "cohort.pbz",
    tracks=["depth"],
    column_dim="sample",
)

cohort = pbzarr.open("cohort.pbz")
try:
    print(cohort["depth"].to_dataset()["values"])
finally:
    cohort.close()
