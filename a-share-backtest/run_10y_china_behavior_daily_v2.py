import numpy as np
import run_10y_china_behavior_daily as m

_old_score = m.BehaviorRidge.score
_old_top_idx = m.top_idx

def _safe_score(self, *args, **kwargs):
    try:
        return _old_score(self, *args, **kwargs)
    except ValueError as e:
        if '0 sample(s)' in str(e):
            # Empty filtered universe means no model trade today.
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
