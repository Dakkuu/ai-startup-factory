from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf

OUT=Path('results_skewfilter_hard'); OUT.mkdir(exist_ok=True)
CONFIGS=[
 ('anchor',None,None,2/3,120),
 ('skew',40,.8,.6,60),('skew',40,.9,.7,60),
 ('skew',60,.7,2/3,60),('skew',60,.8,.6,60),('skew',60,.8,2/3,60),('skew',60,.8,2/3,120),('skew',60,.9,.6,60),
 ('skew',80,.8,.6,60),('skew',80,.9,.7,120),
]
N_HOLD=20

def rerank_anchor(p,wiv=2/3):
    q=p.copy(); q['rank_test']=np.nan
    m=(q.liq_rank_pct<=sim.LIQ_KEEP_PCT)&np.isfinite(q.ivol60)&np.isfinite(q.eff120)
    iv=q.loc[m].groupby('signal_date').ivol60.rank(pct=True,method='average',ascending=True)
    ef=q.loc[m].groupby('signal_date').eff120.rank(pct=True,method='average',ascending=False)
    raw=wiv*iv+(1-wiv)*ef
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q

def subset(q,hold):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,round(hold/5)); chosen=set(dates[::step])
    z=q[q.signal_date.isin(chosen)][['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def choose_det(g,current):
    col='ivol60_pct'
    x=g[np.isfinite(g[col])].sort_values([col,'liq20','code'],ascending=[True,False,True]).copy()
    keep=[c for c in x.loc[x[col]<=sim.KEEP_PCT,'code'].tolist() if c in current]
    keep=keep[:N_HOLD]
    if len(keep)<N_HOLD:
        entrants=[c for c in x.loc[x[col]<=sim.ENTRY_PCT,'code'].tolist() if c not in current and c not in keep]
        keep.extend(entrants[:N_HOLD-len(keep)])
    return keep[:N_HOLD]

def hard_simulate(panel,cal,members,cost_mult=1.0):
    by={d:g.set_index('code',drop=False) for d,g in panel.groupby('signal_date')}; dates=sorted(by)
    cash=sim.INITIAL_CASH; pos={}; equity=[]; trades=[]; timing=[]; turnover=0.0
    member_end=members.groupby('code').end.max().to_dict(); close_cache={}
    def close_series(code):
        if code not in close_cache: close_cache[code]=base.qb.read_bin(code,'close',cal).loc[sim.START:sim.END]
        return close_cache[code]
    trade_cal=cal[(cal>=sim.START)&(cal<=sim.END)]; slip=sim.SLIPPAGE*cost_mult
    for j,d in enumerate(dates):
        g=by[d]; td=pd.Timestamp(g.trade_date.iloc[0]); target=choose_det(g.reset_index(drop=True),set(pos)); tgt=set(target)
        for c,pp in list(pos.items()):
            if c in g.index and np.isfinite(g.loc[c].exec_open): pp.last_price=float(g.loc[c].exec_open)
            elif pd.Timestamp(member_end.get(c,sim.END))<td:
                old=pos.pop(c); trades.append({'variant':'hard','code':c,'entry_date':old.entry_date,'exit_date':td,'net_pnl':-old.entry_cost,'net_return':-1.0,'exit_reason':'membership_end_writeoff'})
        nav_open=cash+sum(pp.units*pp.last_price for pp in pos.values())
        for c in sorted(list(pos)):
            if c in tgt or c not in g.index: continue
            r=g.loc[c]; locked=(np.isfinite(r.exec_open) and np.isfinite(r.exec_high) and np.isfinite(r.exec_low) and abs(float(r.exec_high)-float(r.exec_low))<1e-12 and abs(float(r.exec_open)-float(r.exec_high))<1e-12)
            if locked: continue
            px=float(r.exec_open)*(1-slip); gross=pos[c].units*px; cost=sim.fee(gross,'sell',td,cost_mult); old=pos.pop(c); cash+=gross-cost; turnover+=gross
            trades.append({'variant':'hard','code':c,'entry_date':old.entry_date,'exit_date':td,'net_pnl':gross-cost-old.entry_cost,'net_return':(gross-cost)/old.entry_cost-1,'exit_reason':'rank_exit'})
            timing.append({'variant':'hard','signal_date':pd.Timestamp(d),'trade_date':td,'side':'sell','code':c})
        per=nav_open*.99/N_HOLD
        for c in target:
            if len(pos)>=N_HOLD: break
            if c in pos or c not in g.index: continue
            r=g.loc[c]; locked=(np.isfinite(r.exec_open) and np.isfinite(r.exec_high) and np.isfinite(r.exec_low) and abs(float(r.exec_high)-float(r.exec_low))<1e-12 and abs(float(r.exec_open)-float(r.exec_high))<1e-12)
            if locked: continue
            factor=float(r.exec_factor) if np.isfinite(r.exec_factor) and r.exec_factor>0 else 1.0; adjpx=float(r.exec_open)*(1+slip); rawpx=adjpx/factor
            if rawpx<=0: continue
            maxraw=max(0,int(abs(float(r.exec_volume))*factor*sim.VOLUME_PARTICIPATION//100)*100)
            shares=int(min(per,cash*.98)//(rawpx*100))*100
            if maxraw>0: shares=min(shares,maxraw)
            if shares<=0: continue
            units=shares/factor; gross=units*adjpx; cost=sim.fee(gross,'buy',td,cost_mult); total=gross+cost
            if total>cash: continue
            cash-=total; pos[c]=sim.Pos(units,total,td,float(r.exec_open)); turnover+=gross
            timing.append({'variant':'hard','signal_date':pd.Timestamp(d),'trade_date':td,'side':'buy','code':c})
        if len(pos)>N_HOLD: raise RuntimeError(f'position cap violation {len(pos)}')
        next_td=pd.Timestamp(by[dates[j+1]].trade_date.iloc[0]) if j+1<len(dates) else sim.END+pd.Timedelta(days=1)
        seg=trade_cal[(trade_cal>=td)&(trade_cal<next_td)]
        for day in seg:
            for c,pp in pos.items():
                px=close_series(c).get(day,np.nan)
                if np.isfinite(px) and px>0: pp.last_price=float(px)
            nav=cash+sum(pp.units*pp.last_price for pp in pos.values()); equity.append({'variant':'hard','signal_date':pd.Timestamp(d),'trade_date':pd.Timestamp(day),'equity':nav,'cash':cash,'positions':len(pos)})
    e=pd.DataFrame(equity).drop_duplicates('trade_date',keep='last').sort_values('trade_date'); t=pd.DataFrame(trades); tm=pd.DataFrame(timing)
    if len(tm) and (pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).any(): raise RuntimeError('timing violation')
    return e,t,tm,turnover

def run(q,hold,cal,members,bm,cost=1.0):
    z=subset(q,hold); eq,tr,tm,to=hard_simulate(z,cal,members,cost); st=sim.perf(eq,tr,to,bm)
    st.update({'hold_days':hold,'n_hold':N_HOLD,'cost_mult':cost,'positions_max':int(eq.positions.max()),'positions_median':float(eq.positions.median())})
    st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31'); st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
    return st,eq,tr,tm

def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal); p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close)
    bm=market_close.loc[sim.START:sim.END].dropna(); rows=[]; cache={}
    for name,sw,keep,wiv,h in CONFIGS:
        q=rerank_anchor(p,wiv) if name=='anchor' else sf.rerank(p,sw,keep,wiv)
        print('HARD RUN',name,sw,keep,wiv,h,flush=True); st,eq,tr,tm=run(q,h,cal,members,bm); st.update({'name':name,'skew_window':sw,'skew_keep_pct':keep,'ivol_weight':wiv}); rows.append(st); cache[(name,sw,keep,wiv,h)]=(q,eq,tr,tm)
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'grid.csv',index=False)
    # pre-existing central skew anchor hard stress
    key=('skew',60,.8,2/3,120); q,eq,tr,tm=cache[key]
    costs=[]
    for cm in (2.,4.,8.):
        x,_,_,_=run(q,120,cal,members,bm,cm); x.update({'name':'skew','skew_window':60,'skew_keep_pct':.8,'ivol_weight':2/3}); costs.append(x)
    pd.DataFrame(costs).to_csv(OUT/'anchor_cost.csv',index=False); ann=sim.annual_returns(eq); ann.to_csv(OUT/'anchor_annual.csv',index=False); rob=sim.robustness(eq,tr); pd.DataFrame([rob]).to_csv(OUT/'anchor_robust.csv',index=False)
    bad=sum(int((pd.to_datetime(x[3].signal_date)>=pd.to_datetime(x[3].trade_date)).sum()) for x in cache.values() if len(x[3]))
    audit={**ua,'market_factor':market_code,'execution':'deterministic ranked keep; trapped unsold positions consume slots; hard cap N=20','configs':len(CONFIGS),'timing_violations':bad,'all_positions_le_20':int((grid.positions_max<=20).all())}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad or not (grid.positions_max<=20).all(): raise RuntimeError('hard audit failed')
    print('=== HARD GRID ==='); print(grid.sort_values('total_return',ascending=False).to_string(index=False),flush=True)
    print('=== COST ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== ANNUAL ==='); print(ann.to_string(index=False),flush=True)
    print('=== ROBUST ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
if __name__=='__main__': main()
