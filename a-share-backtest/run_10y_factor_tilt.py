from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq

OUT=Path('results_factor_tilt'); OUT.mkdir(exist_ok=True)
WEIGHTS=(0.50,2/3,0.75,0.85,1.0)
GATES=(False,True)
HOLD_GRID=(60,120)
N_HOLD=20

def rerank(p,w,gate):
    q=p.copy(); q['rank_test']=np.nan
    m=(q.liq_rank_pct<=sim.LIQ_KEEP_PCT)&np.isfinite(q.ivol60)&np.isfinite(q.eff120)
    if gate: m=m&(q.eff120>0)
    a=q.loc[m].groupby('signal_date').ivol60.rank(pct=True,method='average',ascending=True)
    if w>=0.999999:
        raw=a
    else:
        b=q.loc[m].groupby('signal_date').eff120.rank(pct=True,method='average',ascending=False)
        raw=w*a+(1-w)*b
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q

def subset(q,hold):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=hold//5; chosen=set(dates[::step])
    z=q[q.signal_date.isin(chosen)][['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy()
    z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')

def run(q,hold,cal,members,bm,cost=1.0):
    old=sim.N_HOLD; sim.N_HOLD=N_HOLD
    try:
        z=subset(q,hold); eq,tr,tm,to=sim.simulate(z,'ivol',cal,members,cost,daily_mtm=True)
        st=sim.perf(eq,tr,to,bm); st['hold_days']=hold; st['n_hold']=N_HOLD; st['cost_mult']=cost
        st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31')
        st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
        return st,eq,tr,tm
    finally: sim.N_HOLD=old

def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal)
    bm=market_close.loc[sim.START:sim.END].dropna(); br=float(bm.iloc[-1]/bm.iloc[0]-1)
    rows=[]; annual=[]; cache={}
    for gate in GATES:
        for w in WEIGHTS:
            q=rerank(p,w,gate)
            for h in HOLD_GRID:
                print('RUN gate',gate,'w',w,'h',h,flush=True)
                st,eq,tr,tm=run(q,h,cal,members,bm); st.update({'gate_positive':gate,'ivol_weight':w,'benchmark_return':br,'excess':st['total_return']-br})
                rows.append(st); cache[(gate,w,h)]=(q,eq,tr,tm)
                a=sim.annual_returns(eq)
                for _,r in a.iterrows(): annual.append({'gate_positive':gate,'ivol_weight':w,'hold_days':h,'year':int(r.year),'return':float(r['return'])})
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'grid.csv',index=False); pd.DataFrame(annual).to_csv(OUT/'annual_all.csv',index=False)
    wrow=grid.sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False]).iloc[0]
    key=(bool(wrow.gate_positive),float(wrow.ivol_weight),int(wrow.hold_days)); q,eq,tr,tm=cache[key]
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=run(q,key[2],cal,members,bm,cm); st.update({'gate_positive':key[0],'ivol_weight':key[1]}); costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'winner_cost.csv',index=False)
    rob=sim.robustness(eq,tr); rob.update({'gate_positive':key[0],'ivol_weight':key[1],'hold_days':key[2]}); pd.DataFrame([rob]).to_csv(OUT/'winner_robust.csv',index=False)
    # weight-neighborhood summary across both holds and gate choices
    stab=grid.groupby(['gate_positive','ivol_weight']).agg(median_return=('total_return','median'),min_return=('total_return','min'),max_return=('total_return','max'),median_train=('train_2016_2021_return','median'),median_pseudo_oos=('pseudo_oos_2022_2026_return','median'),min_pseudo_oos=('pseudo_oos_2022_2026_return','min'),median_mdd=('max_drawdown','median')).reset_index(); stab.to_csv(OUT/'stability.csv',index=False)
    alltm=pd.concat([x[3] for x in cache.values() if len(x[3])],ignore_index=True); bad=int((pd.to_datetime(alltm.signal_date)>=pd.to_datetime(alltm.trade_date)).sum()) if len(alltm) else 0
    audit={**ua,'market_factor':market_code,'weights':'|'.join(f'{x:.4f}' for x in WEIGHTS),'gates':'none|eff120>0','hold_grid':'60|120','n_hold':N_HOLD,'selection':'train 2016-2021 only; 2022-2026 pseudo-OOS because iterative research','timing_violations':bad}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad: raise RuntimeError('timing violation')
    print('=== GRID ==='); print(grid.sort_values('total_return',ascending=False).to_string(index=False),flush=True)
    print('=== STABILITY ==='); print(stab.to_string(index=False),flush=True)
    print('=== ANNUAL ==='); print(pd.DataFrame(annual).to_string(index=False),flush=True)
    print('=== COST ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== ROBUST ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
if __name__=='__main__': main()
