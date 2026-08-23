from __future__ import annotations
import numpy as np
import run_10y_maxopt_v3_frozen_audit as audit


def noisy_q_copy(q, sigma, rng):
    x=q.copy()
    base=x.rank_test.to_numpy(dtype=float, copy=True)
    m=np.isfinite(base)
    base[m]=np.clip(base[m]+rng.normal(0,float(sigma),int(m.sum())),0,1)
    x['rank_test']=base
    x.loc[m,'rank_test']=x.loc[m].groupby('signal_date').rank_test.rank(pct=True,method='average')
    return x


def random_q_copy(q, rng):
    x=q.copy()
    base=x.rank_test.to_numpy(dtype=float, copy=True)
    m=np.isfinite(base)
    v=np.full(len(x),np.nan)
    v[m]=rng.random(int(m.sum()))
    x['rank_test']=v
    return x


audit.noisy_q=noisy_q_copy
audit.random_q=random_q_copy

if __name__=='__main__':
    audit.main()
