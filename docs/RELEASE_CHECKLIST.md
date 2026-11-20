# Release checklist

## Verified in this package

- All 38 maintained Python files pass syntax compilation; both shell entry points pass `bash -n`.
- Offline aggregation contains 66 adapters and zero input issues.
- Main statistics complete 15/15 comparisons in 6/6 families and match the archived numerical analysis.
- Strong-control, Mistral, independent-coordinate, and derived figure data match their archived analyses.
- Six PDF figures were regenerated successfully.
- No model weights, private keys, server passwords, or internal SSH connection scripts are included.
- Redundant tokenizer caches and superseded historical script copies were removed. Remaining absolute paths in raw result metadata are provenance only.

Run the numerical check from the repository root:

```bash
python scripts/verify_recheck.py
```

## Not performed during packaging

- No full GPU training or GPU inference was rerun.
- No claim is made that a new software environment reproduces bitwise-identical model outputs.
- No GitHub repository was created or uploaded.

## Before public release

1. Add author names, the final repository URL, and citation information.
2. Select and add an appropriate `LICENSE`; none is chosen by this package.
3. Review the licenses and terms for SST-2, AG News, Qwen, Mistral, and all dependencies.
4. Decide whether trained adapters will be distributed separately. Do not commit them accidentally.
5. Review the full diff and raw result metadata, then run `git status --ignored` before committing.
6. Test the offline commands in a clean environment and record exact package versions for any new GPU run.
