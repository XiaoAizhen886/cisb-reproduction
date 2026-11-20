#!/usr/bin/env python3
"""Create model-specific canonical PPL cache for Mistral."""
import argparse
import json
from pathlib import Path
import numpy as np
import kbs_common as C

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
pool_path = Path(C.OUT_BASE) / 'cache' / 'sst2_pool.json'
output = Path(C.OUT_BASE) / 'cache' / 'sst2_base_ppl.npy'
pool = json.loads(pool_path.read_text())
texts = pool['cal_clean'] + pool['test_clean'] + pool['test_trig']
if output.exists():
    print(f'[cache] reuse {output}')
else:
    model, tokenizer = C.load_base(args.gpu)
    values = C.batch_ppl(model, tokenizer, texts)
    if len(values) != len(texts) or not np.isfinite(values).all():
        raise RuntimeError('Invalid Mistral PPL cache')
    with output.open('xb') as handle:
        np.save(handle, values)
    metadata = {'model_path': C.MODEL_PATH, 'pool': str(pool_path),
                'n_texts': len(texts), 'source': 'clean base model; no adapter'}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'[cache] wrote {output} with {len(values)} rows')
