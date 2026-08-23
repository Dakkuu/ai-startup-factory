from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_max_audit as ma

OUT=Path('results_lowprice_signalpure_v1'); OUT.mkdir(exist_ok=True)
SPECS=[
 ('price_only',{'price':1.0}),
 ('price_iv',{'price':.45,'iv':.55}),
 ('price_eff',{'price':.45,'ef':.55}),
 ('price_iv_eff',{'price':.30,'iv':.40,'ef':.30}),
 ('price_trend',{'price':.25,'iv':.20,'ef':.20,'rmom':.22,'tstat':.13}),
 ('price_lowrisk',{'price':.25,'iv':.30,'down':.20,'amax':.15,'ef':.10}),
 ('price_antilottery',{'price':.22,'iv':.25,'amax':.14,'askew':.09,'ef':.12,'rmom':.18}),
]
LIQS=(.55,.70)
FLOORS=(1.0,2.0)
HOLDS=(60,90,120)
NS=(8,10,15,20)
BUFFERS=((.05,.20),(.10,.30))


def attach_price(p,cal):
    q=p.copy(); q['raw_price']=np.nan
    groups=q.groupby('code').groups
    for i,(code,idxs) in enumerate(groups.items(),1):
        idxs=np.asarray(list(idxs)); ds=pd.DatetimeIndex(q.loc[idxs,'signal_date'])
        c=base.qb.read_bin(code,'close',cal); f=base.qb.read_bin(code,'factor',cal)
        if c.empty: continue
        if f.empty: raw=c
        else: raw=c/f.replace(0,np.nan)
        q.loc[idxs,'raw_price']=raw.reindex(ds).to_numpy(float)
        if i%1000==0: print('raw price histories',i,'/',len(groups),flush=True)
    return q


def rank_signal(p,weights,liq,floor):
    x=p.copy(); x['rank_test']=np.nan
    m=mo.eligible_mask(x,float(liq),.80)&np.isfinite(x.raw_price)&(x.raw_price>=float(floor))
    reqmap={'down':'dsemi60','amax':'max20','askew':'skew60','rmom':'rmom126','tstat':'tstat120'}
    for k in weights:
        if k in reqmap:m &= np.isfinite(x[reqmap[k]])
    if not m.any():return x
    R=mo.component_ranks(x,m)
    R['price']=x.loc[m].groupby('signal_date').raw_price.rank(pct=True,method='average',ascending=True)
    raw=None; sw=0.
    for k,w in weights.items():
        if k not in R:continue
        raw=R[k]*w if raw is None else raw+R[k]*w; sw+=w
    raw=raw/sw
    x.loc[m,'rank_test']=raw.groupby(x.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return x


def train(q,h,n,e,k,cal,members,bm):
    st,eq,tr,tm=ma.run_q(q,h,0,cal,members,bm,n=n,entry=e,keep=k,start=mo.START,end=mo.TRAIN_END)
    st['half1_return']=mo.period_return(eq,mo.START,mo.HALF1_END); st['half2_return']=mo.period_return(eq,mo.HALF2_START,mo.TRAIN_END)
    st['min_half_return']=min(st['half1_return'],st['half2_return']) if np.isfinite(st['half1_return']) and np.isfinite(st['half2_return']) else np.nan
    return st


def annual(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]; rows=[]
    for y,g in s.groupby(s.index.year):
        b=s[s.index<pd.Timestamp(f'{y}-01-01')]; st=float(b.iloc[-1]) if len(b) else float(g.iloc[0]); rows.append({'year':int(y),'return':float(g.iloc[-1]/st-1)})
    return pd.DataFrame(rows)


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=attach_price(p,cal)
    rows=[]
    for name,w in SPECS:
      for liq in LIQS:
       for floor in FLOORS:
        print('SIGNAL',name,liq,floor,flush=True); q=rank_signal(p,w,liq,floor)
        for h in HOLDS:
         for n in NS:
          for e,k in BUFFERS:
            st=train(q,h,n,e,k,cal,members,bm); st.update(signal=name,weights=json.dumps(w,sort_keys=True),liq=liq,min_raw_price=floor,hold=h,n_hold=n,entry_pct=e,keep_pct=k,key=f'{name}|l{liq}|f{floor}|h{h}|n{n}|e{e}|k{k}'); rows.append(st)
        del q
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'train_grid.csv',index=False)
    z=grid[(grid.half1_return>0)&(grid.half2_return>0)&(grid.max_drawdown>-0.45)].copy(); z=z if len(z) else grid.copy()
    picks=[z.sort_values(['total_return','min_half_return'],ascending=[False,False]).iloc[0],z.sort_values(['min_half_return','total_return'],ascending=[False,False]).iloc[0],z.assign(score=z.total_return/(1+z.max_drawdown.abs())).sort_values(['score','min_half_return'],ascending=[False,False]).iloc[0]]
    fins=pd.DataFrame(picks).drop_duplicates('key'); fins.to_csv(OUT/'train_selected.csv',index=False)
    full=[]; costs=[]; phases=[]; anns=[]; tails=[]
    for r in fins.itertuples(index=False):
        w=dict(SPECS)[r.signal]; q=rank_signal(p,w,float(r.liq),float(r.min_raw_price))
        st,eq,tr,tm=mo.full_run(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm); st.update(key=r.key,signal=r.signal,liq=float(r.liq),min_raw_price=float(r.min_raw_price),hold=int(r.hold),n_hold=int(r.n_hold),entry_pct=float(r.entry_pct),keep_pct=float(r.keep_pct),selection='2016-2021 only'); full.append(st)
        a=annual(eq); a['key']=r.key; anns.append(a); rr=sim.robustness(eq,tr); rr['key']=r.key; tails.append(rr)
        for cm in (2.,4.,8.):
            x,_,_,_=mo.full_run(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm,cost=cm); x.update(key=r.key,cost_mult_test=cm); costs.append(x)
        step=max(1,round(int(r.hold)/5))
        for ph in range(step):
            x,_,_,_=mo.full_run(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm,phase=ph); x.update(key=r.key,phase=ph); phases.append(x)
    pd.DataFrame(full).to_csv(OUT/'full_finalists.csv',index=False); pd.DataFrame(costs).to_csv(OUT/'costs.csv',index=False); pd.DataFrame(phases).to_csv(OUT/'phases.csv',index=False); pd.DataFrame(tails).to_csv(OUT/'tails.csv',index=False)
    if anns:pd.concat(anns,ignore_index=True).to_csv(OUT/'annual.csv',index=False)
    ps=[]
    for key,g in pd.DataFrame(phases).groupby('key'):
        ps.append({'key':key,'phase_count':len(g),'min_return':g.total_return.min(),'median_return':g.total_return.median(),'mean_return':g.total_return.mean(),'max_return':g.total_return.max(),'all_positive':int((g.total_return>0).all()),'min_pseudo_oos':g.pseudo_oos_2022_2026_return.min(),'median_pseudo_oos':g.pseudo_oos_2022_2026_return.median()})
    pd.DataFrame(ps).to_csv(OUT/'phase_summary.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'signal_rows_with_raw_price':int(np.isfinite(p.raw_price).sum()),'train_points':len(grid),'selection':'2016-2021 only; 2022-2026 pseudo-OOS only','signal_universe':'existing T-only signal-pure rows; raw price attached from T close/factor only; no T+1 gating','volume_unit_shares':100,'target500_hits_finalists':int((pd.DataFrame(full).total_return>=5).sum()) if full else 0}]).to_csv(OUT/'audit.csv',index=False)
    print('=== TRAIN TOP ==='); print(grid.sort_values('total_return',ascending=False).head(30).to_string(index=False),flush=True)
    print('=== FULL ==='); print(pd.DataFrame(full).to_string(index=False),flush=True)
    print('=== PHASE ==='); print(pd.DataFrame(ps).to_string(index=False),flush=True)

if __name__=='__main__':main()
