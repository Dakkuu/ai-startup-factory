from __future__ import annotations
import sys
import requests
import pandas as pd

_orig_get=requests.get
def _safe_get(*a,**k):
    k.setdefault('timeout',20)
    return _orig_get(*a,**k)
requests.get=_safe_get

import run_geff_fundamental_fastpit_v1 as fp

def _qs(y0=2015,y1=2026):
    return [f'{y}{md}' for y in range(2015,2027) for md in ('0331','0630','0930','1231') if f'{y}{md}'<='20260630']
fp.quarter_ends=_qs

def ann_mult(report_date):
    m=pd.to_datetime(report_date).dt.month
    return m.map({3:4.0,6:2.0,9:4.0/3.0,12:1.0}).astype(float)

mode=sys.argv[1] if len(sys.argv)>1 else '3stmt'
if mode=='3stmt':
    import run_geff_fundamental_3stmt_pit_v1 as m
    old=m.prepare_events
    def corrected(B,I,C,cal):
        E=old(B,I,C,cal)
        if len(E):
            mult=ann_mult(E.report_date)
            for c in ['roa_proxy','asset_turnover','cfo_assets','accrual_quality']:
                E[c]=E[c]*mult
        return E
    m.prepare_events=corrected
    m.OUT=m.Path('results_geff_fundamental_3stmt_period_v2'); m.OUT.mkdir(exist_ok=True)
    m.main()
elif mode=='value':
    import run_geff_fundamental_value_pit_v1 as m
    old=m.attach
    def corrected(p,E,cal):
        A=old(p,E,cal)
        if len(A):
            mult=ann_mult(A.report_date)
            A['earnings_yield']=A['earnings_yield']*mult
            A['cashflow_yield']=A['cashflow_yield']*mult
        return A
    m.attach=corrected
    m.OUT=m.Path('results_geff_fundamental_value_period_v2'); m.OUT.mkdir(exist_ok=True)
    m.main()
else:
    raise SystemExit(mode)
