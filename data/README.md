# Data and coordinates

## Public tabular data

- `sst2_validation.parquet`: the 872-example SST-2 validation split used by the unified behavior protocol and the canonical detection pool.
- `agnews_test.parquet`: the public AG News test split used by the multiclass experiments.
- `sst2_pool.json`: 100 clean calibration examples and 200 disjoint clean/triggered test pairs, generated with split seed 2024.
- `sst2_cache_meta.json`: provenance for the canonical pool.

The parquet files are convenience copies of public dataset splits. Their original source locations and SHA-256 hashes are recorded in `source_manifest.json`; those source locations are provenance, not runtime paths.

## Circuit files

`circuit_library_cisb.json` and `circuit_library_cisb_agnews.json` contain target-class coordinates. `circuit_random*.json` contain sampled controls. `circuit_non_target.json` and the Mistral coordinate files support the corresponding counterfactual and cross-architecture checks. JSON files record model and discovery metadata where available.

No base-model weights, LoRA weights, Hugging Face credentials, or server credentials belong in this directory.

## Optional inputs

Paraphrase-style evaluations require separately generated mapping files. Set `KBS_PARAPHRASE_MAP_DIR` to their directory. The main CISB results do not require those optional maps.
