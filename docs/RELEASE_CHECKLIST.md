# Release checklist

## Packaging-time verification (historical)

The following results were recorded during packaging. They are historical provenance, not a substitute for validating the current revision.

- All 38 maintained Python files passed syntax compilation; both shell entry points passed `bash -n`.
- Offline aggregation contained 66 adapters and zero input issues.
- Main statistics completed 15/15 comparisons in 6/6 families and matched the archived numerical analysis.
- Strong-control, Mistral, independent-coordinate, and derived figure data matched their archived analyses.
- Six PDF figures were regenerated successfully.
- No model weights, private keys, server passwords, or internal SSH connection scripts are included.
- Redundant tokenizer caches and superseded historical script copies were removed. Historical source locations are stored as `archive://source/...` provenance identifiers.

## Revalidate the current revision

Run these checks from the repository root and record their actual results:

```bash
python scripts/verify_recheck.py
python -m compileall -q code baseline_code scripts
while IFS= read -r f; do bash -n "$f" || exit 1; done < <(git ls-files '*.sh')
git diff --check
git status --short
```

### Local validation on 2026-11-21

- `python scripts/verify_recheck.py` passed: 15/15 main comparisons in 6/6 families, strong controls, Mistral, independent-coordinate statistics, derived figure data, and all six regenerated PDFs matched the archive.
- `python -m compileall -q code baseline_code scripts` passed.
- `bash -n` passed for every tracked shell script.
- A release-hygiene pass normalized 8,926 local absolute paths in 47 tracked archive files to `archive://source/...` identifiers. A field-by-field JSON comparison confirmed that numerical values, hashes, and all non-path fields were unchanged.
- `python scripts/verify_recheck.py` passed again after path normalization.
- `git diff --check` passed.
- Final release cleanup removed 11 redundant files: two duplicate main-summary copies, two duplicate contrast-statistics copies, one duplicate sensitivity-statistics copy, and six identical tokenizer `merges.txt` files. Canonical files under `results/main/` and `results/recheck/` were retained.
- A final path-hygiene pass normalized 2,349 residual server-absolute paths in 202 tracked files to `archive://source/...` identifiers. The offline recheck was not rerun after this final path-only and duplicate-file cleanup.
- These checks used the current local analysis environment. They are not a clean-environment test, and no full GPU training or GPU inference was run.

## Not performed during packaging

- No full GPU training or GPU inference was rerun.
- No claim is made that a new software environment reproduces bitwise-identical model outputs.
- No GitHub repository was created or uploaded during packaging.

The configured repository URL is <https://github.com/XiaoAizhen886/cisb-reproduction>. No release tag or archival DOI is asserted in this checklist.

## Before submission and public release

1. Confirm author names, the repository URL, and citation information. After pushing the final revision, record the exact commit and any release tag; optionally archive the release and add its DOI.
2. Keep the MIT `LICENSE` and its copyright notice in redistributions.
3. Review the licenses and terms for SST-2, AG News, Qwen, Mistral, and all dependencies.
4. Decide whether trained adapters will be distributed separately. Do not commit them accidentally; the current package does not redistribute base-model or trained LoRA weights.
5. Review the full diff and raw result metadata, then run `git status --ignored` before committing.
6. Test the offline commands in a clean environment and record exact package versions for any new GPU run.
7. Confirm that the submitted PDF, archived analyses, and released repository revision describe the same experiments and numerical results.
