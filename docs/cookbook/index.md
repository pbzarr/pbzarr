# Cookbook

Each recipe solves one task. Each recipe has the same parts: the goal, what you need before you start, the steps, and the result.

The recipes use a store `cohort.pbz` with a `depth` track over `(position, sample)`. Use your own store and names in their place.

| Recipe | Task |
|---|---|
| [Build a cohort store](build-a-cohort-store.md) | Import many files as the columns of one track, and add samples later. |
| [Depth in regions](depth-in-regions.md) | Get the mean, median, or another statistic for each region and sample. |
| [Callable mask](callable-mask.md) | Mark the bases with usable depth in each sample, and count them. |
| [Compare samples](compare-samples.md) | Compare each sample with the cohort, or two samples with each other. |
| [Export to bedGraph](export-bedgraph.md) | Write a track or a region as text, from the command line or from Python. |
| [Save a derived track](derived-track.md) | Compute a whole-genome result and save it in the store. |
