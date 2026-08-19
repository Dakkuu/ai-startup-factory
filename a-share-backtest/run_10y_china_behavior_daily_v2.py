import numpy as np
import run_10y_china_behavior_daily as m

_old_score = m.BehaviorRidge.score

def _safe_score(self, *args, **kwargs):
    try:
        return _old_score(self, *args, **kwargs)
    except ValueError as e:
        if '0 sample(s)' in str(e):
            # BehaviorRidge is expected to return a full cross-sectional score vector.
            # If its filtered candidate set is empty, represent "no prediction/no trade"
            # by a full-length NaN vector; never relax the filter or substitute another stock.
            for a in reversed(args):
                if isinstance(a, np.ndarray) and a.ndim == 1 and a.size > 0:
                    return np.full(a.shape, np.nan, dtype=float)
            for v in m.__dict__.values():
                if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] > 1000:
                    return np.full(v.shape[1], np.nan, dtype=float)
            raise RuntimeError('Could not infer cross-sectional universe length for empty Ridge candidate set') from e
        raise

m.BehaviorRidge.score = _safe_score
m.main()
