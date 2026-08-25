import shutil
from pathlib import Path
import numpy as np
import run_10y_china_behavior_daily as m

_old_score = m.BehaviorRidge.score
_old_top_idx = m.top_idx

def _safe_score(self, *args, **kwargs):
    try:
        return _old_score(self, *args, **kwargs)
    except ValueError as e:
        if '0 sample(s)' in str(e):
            return np.empty(0, dtype=float)
        raise

def _safe_top_idx(score, mask, k):
    score_arr = np.asarray(score)
    if score_arr.size == 0:
        return np.empty(0, dtype=int)
    return _old_top_idx(score, mask, k)

m.BehaviorRidge.score = _safe_score
m.top_idx = _safe_top_idx
m.main()

# Normalize result directory for the workflow artifact without altering any calculation.
src = Path(getattr(m, 'OUT', ''))
dst = Path('results_10y_china_behavior')
print('actual result directory:', src, 'exists=', src.exists(), flush=True)
if src.exists() and src.resolve() != dst.resolve():
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
