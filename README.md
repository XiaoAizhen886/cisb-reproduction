# CISB Reproduction Package

Repository: <https://github.com/XiaoAizhen886/cisb-reproduction>

This repository provides the reproducible open-source code for the manuscript **“CISB: A Circuit-Informed Framework for Target-Class Knowledge-Guided Test LoRA Adapter Generation and Multi-View Activation Analysis.”** It also includes coordinate configurations, small public-data splits, archived per-sample results, paper figures, and offline analysis scripts.

CISB is a framework with three connected components: a **generation method** for target-class knowledge-guided test LoRA adapters, a **test suite** that organizes the resulting test cases and controls, and an **evaluation protocol** for behavioral and multi-view activation analysis. Here, *circuit-informed* refers to candidate attention-head circuit coordinates selected through logit ablation and related coordinate analyses; it does not claim exhaustive circuit discovery.

The package distinguishes **reanalysis of archived results** from **full GPU training**. The former runs without model weights or a GPU. The latter requires the original base models, trained LoRA adapters for evaluation-only stages, and a suitable CUDA environment.

## Repository layout

- `code/`: coordinate discovery, CISB training, evaluation, spectral surgery, and statistical analysis.
- `baseline_code/`: external-baseline training and evaluation utilities.
- `configs/`: experiment matrix and protocol metadata.
- `data/`: circuit coordinates, the canonical SST-2 pool, and small public SST-2/AG News parquet splits.
- `results/`: archived SST-2 and AG News records, including the fixed main analysis in `results/main/`.
- `results_fx41/`, `results_fx41_analysis/`: strong-coordinate and cross-configuration surgery records.
- `mistral_results/`, `mistral_analysis/`: minimal Mistral replication records and analysis.
- `scripts/`: offline rechecks, figure generation, and GPU orchestration entry points.
- `figures/paper/`: figures used by the manuscript; `figures/recheck/` contains regenerated figures.
- `docs/`: configuration naming and release notes.

Local paths embedded in archived JSON or adapter metadata have been normalized to `archive://source/...` provenance identifiers; they are not runtime configuration. Superseded historical script copies are excluded from the archive; maintained code is under `code/`.

## Offline recheck

Use Python 3.10 or later. Create an isolated environment and install only the analysis dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-analysis.txt
```

From the repository root, run:

```bash
python code/analyze_results.py --out results/recheck/summary.local.json
python code/contrast_stats.py --out results/recheck/contrast_stats.local.json
python scripts/analyze_fx41.py
python scripts/analyze_mistral_fx42.py
python scripts/analyze_coordinates.py
python scripts/make_main_figures.py
python scripts/make_fx41_figures.py
python scripts/verify_recheck.py
```

The analyzer refuses to overwrite an explicitly specified existing output. Use a new `.local.json` name for repeated runs. The coordinate bootstrap is deterministic but comparatively slow.

The following packaging-time recheck is retained as historical provenance; rerun the commands above to verify the current revision. It reproduced:

- 66 aggregated adapters with zero input issues;
- all 15 planned main comparisons in 6 complete Benjamini–Hochberg families;
- 20 strong-control comparisons and the Mistral control analysis;
- independent-coordinate statistics and both figure groups.

`scripts/verify_recheck.py` compares the numerical outputs with the archived analyses. It does not treat regenerated paths or timestamps as numerical evidence.

## Full GPU reproduction

Install `requirements.txt` in a CUDA-compatible environment. Package versions are intentionally not invented in this release; consult archived environment metadata where available and record the versions used for a new run.

The default model identifiers are `Qwen/Qwen2.5-7B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3`. Local model directories can be supplied through environment variables:

```bash
export KBS_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export KBS_MISTRAL_MODEL_PATH=/path/to/Mistral-7B-Instruct-v0.3
export KBS_ADAPTER_BASE="$PWD/adapters"
export KBS_SST2_PARQUET="$PWD/data/sst2_validation.parquet"
export KBS_AGNEWS_PARQUET="$PWD/data/agnews_test.parquet"
```

Typical entry points are:

```bash
bash scripts/run_phase.sh prepare
bash scripts/run_phase.sh p0
bash scripts/run_phase.sh ag
python code/run_remote_fx41.py
python code/run_mistral_fx42.py
```

These commands can launch long training and inference jobs. Inspect their resource settings, adapter reuse checks, and GPU assignments before running them. New orchestration outputs go under `runs/`; archived result directories are not intended as training output locations.

To evaluate an existing adapter without modifying its directory, use `code/cisb_evaluate_output.py` and provide separate `--output_json` and `--records_json` paths. The training recipes and configuration-name mapping are documented in `configs/experiment_matrix.json` and `docs/configuration_mapping_zh.md`.

## Data and adapter weights

SST-2 and AG News are public datasets. The included parquet files support exact reconstruction of the evaluation pools; see `data/README.md` and `data/source_manifest.json`.

**Base-model weights and trained LoRA weights are not redistributed.** Some archived manifests retain adapter hashes for provenance, but a hash is not a model artifact. Full evaluation therefore requires training the adapters or obtaining them separately from the authors.

## Scope and research use

This code is for controlled research on adapter safety testing and activation geometry. Run poisoning and backdoor experiments only in authorized local or controlled environments. The manuscript reports both positive and negative findings; the archived negative results are retained rather than removed.

## Release status

The repository URL is <https://github.com/XiaoAizhen886/cisb-reproduction>. The repository's original code, documentation, analyses, and figures are released under the MIT License; see `LICENSE`. Third-party datasets, base models, and dependencies remain subject to their respective licenses and terms.

No release tag or archival DOI is asserted here. After pushing the final revision, the authors should record the exact commit and any release tag, then optionally archive that release and add its DOI. The manuscript, archived results, and released repository revision should be kept consistent.
