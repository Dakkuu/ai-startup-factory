from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_geff_phase0_identity'); OUT.mkdir(exist_ok=True)
HORIZONS=(30,35,40,45,50,55,60,65,70,75,90,105)
STYLES={
    'mom': {'iv':.25,'down':.15,'rmom':.35,'tstat':.25},
    'def': {'iv':.35,'down':.25,'rmom':.25,'tstat':.15},
}

def subset(q,h):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
    step=max(1,round(h/5)); chosen=set(dates[::step])
    cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def eqret(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index()
    s=s[~s.index.duplicated(keep='last')]
    return s.pct_change().dropna()

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False)
    p=strict.attach_gap_flags(p,cal,'board')
    metrics=[]; returns={}
    for style,w in STYLES.items():
        sp={'name':f'geff55_{style}','kind':'gate','g':{'ef':.55},'w':w}
        q=mega.make_rank(p,sp)
        for h in HORIZONS:
            print('RUN',style,h,flush=True)
            st,eq,tr,tm=ma.run_panel(subset(q,h),cal,members,bm,n=10,entry=.10,keep=.30)
            name=f'geff55_{style}|h{h}|n10|e0.1|k0.3'
            st.update(candidate=name,style=style,H=h)
            metrics.append(st); returns[name]=eqret(eq).rename(name)
        del q
    pd.DataFrame(metrics).to_csv(OUT/'phase0_metrics.csv',index=False)
    R=pd.concat(returns.values(),axis=1,join='inner').sort_index(); R.index.name='trade_date'
    R.to_csv(OUT/'phase0_daily_returns.csv')
    meta={'source_commit':'91a9247473f0ea9e68bbd73560c3e6cf63127ef6','market_factor':market_code,'universe_audit':ua,'horizons':list(HORIZONS)}
    (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
    print(pd.DataFrame(metrics)[['candidate','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True)

if __name__=='__main__': main()
