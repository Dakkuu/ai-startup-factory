from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_alpha2f_v2 as sim
import run_10y_max_audit as ma

OUT=Path('results_long_refine_v1'); OUT.mkdir(exist_ok=True)
WEIGHTS=(.42,.46,.50,.54,.58)
HOLDS=(100,110,120,130,140)
NS=(8,10,12,15,20)
BUFFERS=((.05,.20),(.10,.30))


def annual(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]
    out=[]
    for y,g in s.groupby(s.index.year):
        b=s[s.index<pd.Timestamp(f'{y}-01-01')]; st=float(b.iloc[-1]) if len(b) else float(g.iloc[0])
        out.append({'year':int(y),'return':float(g.iloc[-1]/st-1)})
    return pd.DataFrame(out)


def train(q,h,n,e,k,cal,members,bm):
    st,eq,tr,tm=ma.run_q(q,h,0,cal,members,bm,n=n,entry=e,keep=k,start=mo.START,end=mo.TRAIN_END)
    st['half1_return']=mo.period_return(eq,mo.START,mo.HALF1_END); st['half2_return']=mo.period_return(eq,mo.HALF2_START,mo.TRAIN_END)
    st['min_half_return']=min(st['half1_return'],st['half2_return']) if np.isfinite(st['half1_return']) and np.isfinite(st['half2_return']) else np.nan
    return st


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False)
    rows=[]
    for w in WEIGHTS:
        print('WEIGHT',w,flush=True); q=mo.rerank(p,mo.baseline_spec(w))
        for h in HOLDS:
          for n in NS:
           for e,k in BUFFERS:
            st=train(q,h,n,e,k,cal,members,bm); st.update(ivol_weight=w,hold=h,n_hold=n,entry_pct=e,keep_pct=k,key=f'w{w}|h{h}|n{n}|e{e}|k{k}'); rows.append(st)
        del q
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'train_grid.csv',index=False)
    z=grid[(grid.half1_return>0)&(grid.half2_return>0)&(grid.max_drawdown>-0.40)].copy()
    if len(z)==0:z=grid.copy()
    growth=z.sort_values(['total_return','min_half_return'],ascending=[False,False]).iloc[0]
    robust=z.sort_values(['min_half_return','total_return'],ascending=[False,False]).iloc[0]
    riskadj=z.assign(score=z.total_return/(1+z.max_drawdown.abs())).sort_values(['score','min_half_return'],ascending=[False,False]).iloc[0]
    fins=pd.DataFrame([growth,robust,riskadj]).drop_duplicates('key'); fins.to_csv(OUT/'train_selected.csv',index=False)
    full=[]; costs=[]; phases=[]; anns=[]; tails=[]
    for r in fins.itertuples(index=False):
        q=mo.rerank(p,mo.baseline_spec(float(r.ivol_weight)))
        st,eq,tr,tm=mo.full_run(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm); st.update(key=r.key,ivol_weight=float(r.ivol_weight),hold=int(r.hold),n_hold=int(r.n_hold),entry_pct=float(r.entry_pct),keep_pct=float(r.keep_pct),selection='2016-2021 only'); full.append(st)
        a=annual(eq); a['key']=r.key; anns.append(a)
        rr=sim.robustness(eq,tr); rr['key']=r.key; tails.append(rr)
        for cm in (2.,4.,8.):
            x,_,_,_=mo.full_run(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm,cost=cm); x.update(key=r.key,cost_mult_test=cm); costs.append(x)
        step=max(1,round(int(r.hold)/5))
        for ph in range(step):
            x,_,_,_=mo.full_run(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm,phase=ph); x.update(key=r.key,phase=ph); phases.append(x)
    pd.DataFrame(full).to_csv(OUT/'full_finalists.csv',index=False); pd.DataFrame(costs).to_csv(OUT/'costs.csv',index=False); pd.DataFrame(phases).to_csv(OUT/'phases.csv',index=False); pd.DataFrame(tails).to_csv(OUT/'tails.csv',index=False)
    if anns: pd.concat(anns,ignore_index=True).to_csv(OUT/'annual.csv',index=False)
    ps=[]
    for key,g in pd.DataFrame(phases).groupby('key'):
        ps.append({'key':key,'phase_count':len(g),'min_return':g.total_return.min(),'median_return':g.total_return.median(),'max_return':g.total_return.max(),'all_positive':int((g.total_return>0).all()),'min_pseudo_oos':g.pseudo_oos_2022_2026_return.min(),'median_pseudo_oos':g.pseudo_oos_2022_2026_return.median()})
    pd.DataFrame(ps).to_csv(OUT/'phase_summary.csv',index=False)
    audit={**ua,'market_factor':market_code,'points':len(grid),'selection':'2016-2021 only; 2022-2026 pseudo-OOS only','signal_universe':'T-only signal-pure','volume_unit_shares':100,'timing_violations':0}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    print('=== TRAIN TOP ==='); print(grid.sort_values('total_return',ascending=False).head(30).to_string(index=False),flush=True)
    print('=== FULL ==='); print(pd.DataFrame(full).to_string(index=False),flush=True)
    print('=== PHASE SUMMARY ==='); print(pd.DataFrame(ps).to_string(index=False),flush=True)

if __name__=='__main__': main()
