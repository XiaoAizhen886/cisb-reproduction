# Archived results and rechecks

`results/main/` is the fixed main-analysis snapshot used by the manuscript. Other directories under `results/` contain per-sample SST-2 and AG News behavior, canonical, residual, coordinate, and audit records.

- `archive/`: earlier machine-readable records used by the aggregator. Superseded historical script copies were removed from the release; result records and adapter hashes were not changed.
- `recheck/`: outputs regenerated with the portable scripts during packaging.
- `../results_fx41/`: strong-coordinate and cross-configuration surgery inputs.
- `../mistral_results/`: minimal Mistral replication inputs.

Raw records may contain absolute paths from the original execution environment. These strings are source metadata only. Portable scripts resolve new inputs and outputs relative to the repository or through documented environment variables.

The raw archived files should not be overwritten. New aggregation, bootstrap, and figure outputs belong in `results/recheck/`, another explicitly selected output directory, or `runs/`.

Adapter weights are not included. Archived adapter SHA-256 values support provenance checks but do not replace the weight files required for GPU evaluation.
