# V1.1 patch wrapper: fixes breakout lookback, sector-UNKNOWN leakage, and turnover fallback.
import numpy as np
import pandas as pd
from factor_engine import compute_factors as _legacy_compute

EPS=1e-12

def compute_factors(bars: pd.DataFrame) -> pd.DataFrame:
    b=bars.copy().sort_values(['ts_code','trade_date']).reset_index(drop=True)
    has_turn=('turnover_rate' in b and b['turnover_rate'].notna().mean()>=0.20)
    if not has_turn:
        if 'volume' not in b: raise ValueError('volume required')
        b['turnover_rate']=np.log1p(pd.to_numeric(b['volume'],errors='coerce').clip(lower=0))
    out=_legacy_compute(b)
    g=b.groupby('ts_code',group_keys=False)
    prev_hh=g['high'].transform(lambda s:s.shift(1).rolling(20,min_periods=20).max())
    prev_ll=g['low'].transform(lambda s:s.shift(1).rolling(20,min_periods=20).min())
    out['F39']=b['close']/(prev_hh+EPS)-1
    out['F40']=b['close']/(prev_ll+EPS)-1
    ind=b.get('industry',pd.Series('UNKNOWN',index=b.index)).fillna('UNKNOWN').astype(str)
    bad=ind.str.upper().isin(['UNKNOWN','UNK','NAN','NONE',''])
    if bad.all():
        for c in ['F53','F54','F55','F56']: out[c]=np.nan
    else:
        for c in ['F53','F54','F55','F56']: out.loc[bad,c]=np.nan
    out.attrs['turnover_basis']='turnover_rate' if has_turn else 'log1p(volume)_rolling_z_proxy'
    out.attrs['sector_features_available']=bool((~bad).any())
    return out
