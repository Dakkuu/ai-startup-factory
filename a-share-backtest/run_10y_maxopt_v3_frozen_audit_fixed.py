from __future__ import annotations
import numpy as np
import run_10y_maxopt_v3_frozen_audit as audit


def noisy_q_fixed(q, sigma, rng):
    x = q.copy()
    m = np.isfinite(x.rank_test.to_numpy(float))
    v = x.rank_test.to_numpy(dtype=float, copy=True)
    v[m] = np.clip(v[m] + rng.normal(0, float(sigma), int(m.sum())), 0, 1)
    x['rank_test'] = v
    x.loc[m, 'rank_test'] = x.loc[m].groupby('signal_date').rank_test.rank(pct=True, method='average')
    return x


audit.noisy_q = noisy_q_fixed

if __name__ == '__main__':
    audit.main()
