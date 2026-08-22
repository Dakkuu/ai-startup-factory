from __future__ import annotations
import numpy as np
import pandas as pd
import run_10y_max_audit as m

# Pure compatibility patch for pandas 3: make the NumPy view writable.
# Audit design, strategy, random seeds and pre-registered gates are unchanged.
def noise_audit_fixed(q,cal,members,bm):
    rng=np.random.default_rng(m.SEED); rows=[]; q0=m.minimal(q)
    finite=np.isfinite(q0.rank_test.to_numpy(float))
    for sig in m.NOISE_SIGMAS:
        for k in range(m.N_NOISE):
            x=q0.copy(); vals=x.rank_test.to_numpy(float,copy=True)
            vals[finite]=np.clip(vals[finite]+rng.normal(0,sig,finite.sum()),0,1); x['rank_test']=vals
            x.loc[finite,'rank_test']=x.loc[finite].groupby('signal_date').rank_test.rank(pct=True,method='average')
            st,_,_,_=m.run_panel(m.subset_phase(x,60,0),cal,members,bm)
            rows.append({**st,'noise_sigma':sig,'seed':k})
    z=pd.DataFrame(rows); z.to_csv(m.OUT/'rank_noise.csv',index=False); return z

m.noise_audit=noise_audit_fixed

if __name__=='__main__':
    m.main()
