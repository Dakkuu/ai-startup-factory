import numpy as np
import run_10y_china_behavior_daily as m

_old_score = m.BehaviorRidge.score

def _safe_score(self, *args, **kwargs):
    try:
        return _old_score(self, *args, **kwargs)
    except ValueError as e:
        if '0 sample(s)' in str(e):
            return np.empty(0, dtype=float)
        raise

m.BehaviorRidge.score = _safe_score
m.main()
