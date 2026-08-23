from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_geff55_statistics_v2'); OUT.mkdir(exist_ok=True)
ORIG={'iv':.30,'down':.20,'rmom':.30,'tstat':.20}
WEIGHTS=[
 ('orig',ORIG),
 ('def',{'iv':.35,'down':.25,'rmom':.25,'tstat':.15}),
 ('mom',{'iv':.25,'down':.15,'rmom':.35,'tstat':.25}),
]
GATES=(.50,.55,.60); HOLDS=(75,90,105); NS=(10,15,20); BUFFERS=((.05,.20),(.10,.30))

def spec(g,nm,w): return {'name':f'geff{int(g*100)}_{nm}','kind':'gate','g':{'ef':g},'w':w}

def subset(q,h):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,round(h/5)); chosen=set(dates[::step])
    cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')

def run(q,h,n,e,k,cal,members,bm): return ma.run_panel(subset(q,h),cal,members,bm,n=n,entry=e,keep=k)

def eqret(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]
    return s.pct_change().dropna()

def bootstrap(r,block,reps=2500):
    z=ma.moving_block_bootstrap(r,reps,block); rows=[]
    for c in ('total','cagr','sharpe'): rows.append({'block':block,'metric':c,'p2_5':z[c].quantile(.025),'median':z[c].median(),'p97_5':z[c].quantile(.975)})
    return rows

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    rows=[]; rets={}; frozen_eq=None
    for g in GATES:
      for nm,w in WEIGHTS:
        sp=spec(g,nm,w); print('SIGNAL',sp['name'],flush=True); q=mega.make_rank(p,sp)
        for h in HOLDS:
          for n in NS:
            for e,k in BUFFERS:
                name=f'{sp["name"]}|h{h}|n{n}|e{e}|k{k}'; st,eq,tr,tm=run(q,h,n,e,k,cal,members,bm); st.update(candidate=name,gate=g,weights=nm,hold=h,n_hold_test=n,entry=e,keep=k); rows.append(st); rets[name]=eqret(eq).rename(name)
                if g==.55 and nm=='orig' and h==90 and n==15 and e==.10 and k==.30: frozen_eq=eq.copy(); frozen_name=name
        del q
    df=pd.DataFrame(rows); df.to_csv(OUT/'candidate_family.csv',index=False)
    R=pd.concat(rets.values(),axis=1,join='inner').dropna(); R.to_csv(OUT/'candidate_daily_returns.csv')
    pbo=ma.pbo_test(R,nblocks=10); pd.DataFrame([pbo]).to_csv(OUT/'pbo_local_family.csv',index=False)
    rc=ma.reality_check(R,bm,reps=1200,block=20); pd.DataFrame([rc]).to_csv(OUT/'reality_check_local_family.csv',index=False)
    sharpes=df.sharpe.to_numpy(float); fr=eqret(frozen_eq)
    dsr=[]
    for ntr in (len(df),3000,10000): dsr.append(ma.dsr_one(fr,sharpes,ntr))
    pd.DataFrame(dsr).to_csv(OUT/'deflated_sharpe.csv',index=False)
    boots=[]
    for b in (20,60,120): boots.extend(bootstrap(fr,b))
    pd.DataFrame(boots).to_csv(OUT/'bootstrap_ci.csv',index=False)
    # Frozen strategy rank within local family is descriptive only, never used for selection.
    obs=df[df.candidate==frozen_name].iloc[0]; ranks={'frozen_candidate':frozen_name,'family_size':len(df),'return_percentile':float((df.total_return<=obs.total_return).mean()),'sharpe_percentile':float((df.sharpe<=obs.sharpe).mean()),'pbo':pbo['pbo'],'reality_check_p':rc['bootstrap_p'],'dsr3000':float(pd.DataFrame(dsr).loc[pd.DataFrame(dsr).n_trials==3000,'dsr_prob'].iloc[0]),'dsr10000':float(pd.DataFrame(dsr).loc[pd.DataFrame(dsr).n_trials==10000,'dsr_prob'].iloc[0])}
    pd.DataFrame([ranks]).to_csv(OUT/'verdict.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'audit':'GEff55 local-family multiple-testing under strict board-limit proxy','family_specs':len(GATES)*len(WEIGHTS),'construction_per_spec':len(HOLDS)*len(NS)*len(BUFFERS),'candidate_count':len(df),'selection':'NO selection performed; frozen GEff55 audited against family','global_trials_note':'DSR also evaluated at 3000 and 10000 to reflect prior researcher degrees of freedom'}]).to_csv(OUT/'audit.csv',index=False)
    print('VERDICT',pd.DataFrame([ranks]).to_string(index=False),flush=True); print('DSR',pd.DataFrame(dsr).to_string(index=False),flush=True); print('PBO',pd.DataFrame([pbo]).to_string(index=False),flush=True); print('RC',pd.DataFrame([rc]).to_string(index=False),flush=True)

if __name__=='__main__': main()
