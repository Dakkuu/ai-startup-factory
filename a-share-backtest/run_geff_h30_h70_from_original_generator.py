from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_geff_h30_h70_original_generator'); OUT.mkdir(exist_ok=True)
GATE=.55
HORIZONS=tuple(range(30,71,5))
N=10
ENTRY=.10
KEEP=.30
STYLES={
    'mom': {'iv':.25,'down':.15,'rmom':.35,'tstat':.25},
    'def': {'iv':.35,'down':.25,'rmom':.25,'tstat':.15},
}

def spec(style,w):
    return {'name':f'geff55_{style}','kind':'gate','g':{'ef':GATE},'w':w}

def subset(q,h,phase=0):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
    step=max(1,round(h/5))
    chosen=set(dates[phase::step])
    cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(chosen)][cols].copy()
    z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def run(q,h,phase,cal,members,bm):
    return ma.run_panel(subset(q,h,phase),cal,members,bm,n=N,entry=ENTRY,keep=KEEP)

def eqret(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index()
    s=s[~s.index.duplicated(keep='last')]
    return s.pct_change().dropna()

def main():
    # This is the exact contemporaneous module chain used by run_10y_geff55_statistics_v2.py.
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False)
    p=strict.attach_gap_flags(p,cal,'board')

    metrics=[]
    daily_by_style={k:{} for k in STYLES}
    holdings_rows=[]
    ranking_diag=[]

    for style,w in STYLES.items():
        sp=spec(style,w)
        print('BUILD_RANK',style,flush=True)
        q=mega.make_rank(p,sp)
        q[['signal_date','trade_date','code','rank_test']].to_csv(OUT/f'score_rank_{style}.csv.gz',index=False,compression='gzip')
        for h in HORIZONS:
            phase_count=max(1,round(h/5))
            for ph in range(phase_count):
                print('RUN',style,'H',h,'PH',ph,'/',phase_count,flush=True)
                st,eq,tr,tm=run(q,h,ph,cal,members,bm)
                name=f'{style}_h{h}_p{ph}'
                st.update(style=style,H=h,phase=ph,phase_count=phase_count,candidate=f'geff55_{style}|h{h}|n10|e0.1|k0.3')
                metrics.append(st)
                daily_by_style[style][name]=eqret(eq).rename(name)
                if len(tm):
                    x=tm.copy(); x['style']=style; x['H']=h; x['phase']=ph
                    holdings_rows.append(x)
        del q

    m=pd.DataFrame(metrics)
    m.to_csv(OUT/'phase_metrics.csv',index=False)
    for style,dct in daily_by_style.items():
        R=pd.concat(dct.values(),axis=1,join='outer').sort_index()
        R.index.name='trade_date'
        R.to_csv(OUT/f'daily_{style}.csv.gz',compression='gzip')

    if holdings_rows:
        pd.concat(holdings_rows,ignore_index=True).to_csv(OUT/'timing_all_phases.csv.gz',index=False,compression='gzip')

    # Per-H robustness summary; phase0 is preserved separately from all-phase statistics.
    rows=[]
    for (style,h),g in m.groupby(['style','H']):
        r={'style':style,'H':int(h),'phase_count':len(g)}
        for c in ['cagr','max_drawdown','sharpe','sortino','total_return']:
            if c in g:
                r[f'{c}_phase0']=float(g.loc[g.phase.eq(0),c].iloc[0])
                r[f'{c}_min']=float(g[c].min())
                r[f'{c}_median']=float(g[c].median())
                r[f'{c}_mean']=float(g[c].mean())
                r[f'{c}_max']=float(g[c].max())
                r[f'{c}_std']=float(g[c].std(ddof=0))
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/'horizon_phase_summary.csv',index=False)

    meta={
        'status':'ORIGINAL_GENERATOR_EXTENSION_PENDING_IDENTITY_REPLAY_CHECK',
        'source_branch':'geff55-external-audit-v2',
        'source_commit':'91a9247473f0ea9e68bbd73560c3e6cf63127ef6',
        'source_generator':'run_10y_geff55_statistics_v2.py + contemporaneous imports',
        'market_factor_reported_by_build_panel':market_code,
        'gate':GATE,'N':N,'entry':ENTRY,'keep':KEEP,
        'horizons':list(HORIZONS),
        'styles':STYLES,
        'phase_rule':'weekly signal grid; step=round(H/5); phase=0..step-1',
        'universe_audit':ua,
    }
    (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
    print('SUMMARY')
    print(pd.read_csv(OUT/'horizon_phase_summary.csv').to_string(index=False),flush=True)

if __name__=='__main__':
    main()
