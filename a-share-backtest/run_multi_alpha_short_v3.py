from __future__ import annotations
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_era_backtest as base
import run_10y_geff55_strict_audit_v2 as strict
import run_multi_alpha_system_v1 as v1
from run_multi_alpha_shard_common_v2 import run_candidate_v2,normalize_equity
import run_10y_hard_executor_v3 as hv3
hv3.patch()

FAMILIES=('pullback60','pullback120','pullback_lowiv','quiet_pullback','market_relative_pullback')
HOLD=(10,20); NS=(15,20); ENTRY=.10; KEEP=.30

def add_features(p,cal,bm):
    x=p.copy()
    for c in ['ret5','ret60','ret120','relvol20_120','mkt_ret5','excess5']: x[c]=np.nan
    m5=(bm/bm.shift(5)-1)
    groups=x.groupby('code').groups
    warm=pd.Timestamp('2015-01-01')
    for i,(code,idxs) in enumerate(groups.items(),1):
        idxs=np.asarray(list(idxs)); c=base.qb.read_bin(code,'close',cal).loc[warm:v1.END]; vol=base.qb.read_bin(code,'volume',cal).loc[warm:v1.END]
        if c.empty or vol.empty: continue
        r5=c/c.shift(5)-1; r60=c/c.shift(60)-1; r120=c/c.shift(120)-1
        rv=vol.rolling(20,min_periods=16).mean()/vol.rolling(120,min_periods=90).mean().replace(0,np.nan)
        ds=pd.DatetimeIndex(x.loc[idxs,'signal_date']); mr=m5.reindex(ds).to_numpy(float); sr=r5.reindex(ds).to_numpy(float)
        x.loc[idxs,'ret5']=sr; x.loc[idxs,'ret60']=r60.reindex(ds).to_numpy(float); x.loc[idxs,'ret120']=r120.reindex(ds).to_numpy(float); x.loc[idxs,'relvol20_120']=rv.reindex(ds).to_numpy(float); x.loc[idxs,'mkt_ret5']=mr; x.loc[idxs,'excess5']=sr-mr
        if i%1000==0: print('FEATURES',i,'/',len(groups),flush=True)
    x['liq_pct_v3']=x.groupby('signal_date').liq20.rank(pct=True,ascending=False,method='average')
    return x

def rp(q,m,col,asc): return q.loc[m].groupby('signal_date')[col].rank(pct=True,method='average',ascending=asc)

def make_q(p,fam):
    q=p.copy(); q['rank_test']=np.nan; liq=q.liq_pct_v3<=.70
    if fam=='pullback60':
        m=liq & (q.ret60>.05) & np.isfinite(q.ret5); a=rp(q,m,'ret5',True); b=rp(q,m,'ret60',False); raw=.70*a+.30*b
    elif fam=='pullback120':
        m=liq & (q.ret120>.05) & np.isfinite(q.ret5); a=rp(q,m,'ret5',True); b=rp(q,m,'ret120',False); raw=.70*a+.30*b
    elif fam=='pullback_lowiv':
        m=liq & (q.ret60>0) & np.isfinite(q.ret5) & np.isfinite(q.ivol60); a=rp(q,m,'ret5',True); b=rp(q,m,'ret60',False); c=rp(q,m,'ivol60',True); raw=.60*a+.20*b+.20*c
    elif fam=='quiet_pullback':
        m=liq & (q.ret60>0) & np.isfinite(q.ret5) & np.isfinite(q.relvol20_120); a=rp(q,m,'ret5',True); b=rp(q,m,'ret60',False); c=rp(q,m,'relvol20_120',True); raw=.60*a+.20*b+.20*c
    elif fam=='market_relative_pullback':
        m=liq & (q.ret60>0) & np.isfinite(q.excess5); a=rp(q,m,'excess5',True); b=rp(q,m,'ret60',False); raw=.70*a+.30*b
    else: raise ValueError(fam)
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True); return q

def main():
    fam=str(sys.argv[1]);
    if fam not in FAMILIES: raise ValueError(fam)
    OUT=Path(f'results_multi_alpha_short_v3_{fam}'); OUT.mkdir(exist_ok=True)
    p,cal,members,ua,mc,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board'); p=add_features(p,cal,bm); q=make_q(p,fam)
    rows=[]; cache={}
    for h in HOLD:
      for n in NS:
        print('RUN',fam,h,n,flush=True); eq,st=run_candidate_v2(q,h,n,ENTRY,KEEP,cal,members,bm,1.0,False); key=f'{fam}|h{h}|n{n}|e{ENTRY}|k{KEEP}'; rows.append({**st,'family':fam,'H':h,'N':n,'entry':ENTRY,'keep':KEEP,'key':key}); cache[key]=(eq,h,n)
    d=pd.DataFrame(rows); ok=d[(d.train_cagr>0)&(d.train_mdd>-0.45)].copy(); passed=bool(len(ok)); sel=ok if passed else d; win=sel.sort_values(['train_calmar','train_sharpe','trade_count_mean'],ascending=[False,False,True]).iloc[0]; key=str(win.key); eq,h,n=cache[key]
    d.to_csv(OUT/f'short_v3_{fam}_grid.csv',index=False); normalize_equity(eq).to_csv(OUT/f'short_v3_{fam}_equity_cost1.csv',index=False)
    stress=[{**win.to_dict(),'cost_mult':1.0}]
    for cm in (2.0,4.0):
      ee,st=run_candidate_v2(q,h,n,ENTRY,KEEP,cal,members,bm,cm,False); normalize_equity(ee).to_csv(OUT/f'short_v3_{fam}_equity_cost{int(cm)}.csv',index=False); stress.append({**st,'family':fam,'H':h,'N':n,'entry':ENTRY,'keep':KEEP,'key':key,'cost_mult':cm})
    pd.DataFrame(stress).to_csv(OUT/f'short_v3_{fam}_winner_metrics.csv',index=False); (OUT/f'short_v3_{fam}_meta.json').write_text(json.dumps({'family':fam,'winner_key':key,'train_gate_passed':passed,'post_diagnostic':True,'market_factor':mc,'universe_audit':ua},indent=2,default=str)); print('WINNER',key,'GATE',passed,flush=True); print(pd.DataFrame(stress).to_string(index=False),flush=True)
if __name__=='__main__': main()
