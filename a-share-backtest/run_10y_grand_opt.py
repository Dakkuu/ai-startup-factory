from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_skewfilter_hard as hard

OUT=Path('results_grand_opt'); OUT.mkdir(exist_ok=True)
SIGNALS=('anchor','tstat','tri_tstat','tri_rmom','tri_pos','tri_dd','tri_volshock','anchor_rmompos','anchor_mompos')
POOLS=('all80','liq60','liq70','liq90','mature365','mature730','ex_star_bj','quality_pool')
NS=(10,15,20)
HOLDS=(50,60,70)
BUFFERS=((.05,.20),(.10,.30),(.15,.40))


def add_grand_fields(panel,cal,members):
    p=panel.copy()
    for c in ('tstat120','posratio120','dd120','mom120raw'): p[c]=np.nan
    groups=p.groupby('code').groups
    for i,(code,idx) in enumerate(groups.items(),1):
        c=base.qb.read_bin(code,'close',cal).loc[sim.WARM:sim.END]
        if c.empty: continue
        r=c.pct_change(fill_method=None); y=np.log(c.where(c>0)); tt=pd.Series(np.arange(len(y),dtype=float),index=y.index)
        corr=y.rolling(120,min_periods=100).corr(tt).clip(-.999999,.999999)
        tstat=corr*np.sqrt(118.0/(1-corr*corr))
        pos=(r>0).where(r.notna()).astype(float).rolling(120,min_periods=100).mean()
        dd=c/c.rolling(120,min_periods=100).max()-1; mom=c/c.shift(120)-1
        ds=pd.DatetimeIndex(p.loc[idx,'signal_date'])
        p.loc[idx,'tstat120']=tstat.reindex(ds).to_numpy(float); p.loc[idx,'posratio120']=pos.reindex(ds).to_numpy(float)
        p.loc[idx,'dd120']=dd.reindex(ds).to_numpy(float); p.loc[idx,'mom120raw']=mom.reindex(ds).to_numpy(float)
        if i%1000==0: print('grand fields',i,'/',len(groups),flush=True)
    first=members.groupby('code').start.min(); p['first_start']=p.code.map(first)
    p['age_days']=(pd.to_datetime(p.signal_date)-pd.to_datetime(p.first_start)).dt.days
    def board(code):
        s=str(code)
        if s.startswith('BJ'): return 'BJ'
        if s.startswith('SH688'): return 'STAR'
        if s.startswith('SZ300') or s.startswith('SZ301'): return 'CHINEXT'
        return 'MAIN'
    p['board']=p.code.map(board); p['volshock']=p.ivol40/p.ivol80.replace(0,np.nan)
    return p


def pool_mask(q,name):
    m=np.isfinite(q.ivol60)&np.isfinite(q.eff120)&np.isfinite(q.skew40)
    liq={'all80':.80,'liq60':.60,'liq70':.70,'liq90':.90,'mature365':.80,'mature730':.80,'ex_star_bj':.80,'quality_pool':.70}[name]
    m=m&(q.liq_rank_pct<=liq)
    if name=='mature365': m=m&(q.age_days>=365)
    if name=='mature730': m=m&(q.age_days>=730)
    if name=='ex_star_bj': m=m&(~q.board.isin(['STAR','BJ']))
    if name=='quality_pool': m=m&(q.age_days>=365)&(~q.board.isin(['STAR','BJ']))
    mm=q.loc[m]; sp=mm.groupby('signal_date').skew40.rank(pct=True,method='average',ascending=True)
    ok=pd.Series(False,index=q.index); ok.loc[sp.index]=sp<=.80
    return m&ok


def pct_rank(q,m,col,ascending): return q.loc[m].groupby('signal_date')[col].rank(pct=True,method='average',ascending=ascending)


def rerank(p,signal,pool):
    q=p.copy(); q['rank_test']=np.nan; m=pool_mask(q,pool)
    needed=[]
    if signal in ('tstat','tri_tstat'): needed=['tstat120']
    elif signal in ('tri_rmom','anchor_rmompos'): needed=['rmom126']
    elif signal=='tri_pos': needed=['posratio120']
    elif signal=='tri_dd': needed=['dd120']
    elif signal=='tri_volshock': needed=['volshock']
    elif signal=='anchor_mompos': needed=['mom120raw']
    for c in needed: m=m&np.isfinite(q[c])
    if signal=='anchor_rmompos': m=m&(q.rmom126>0)
    if signal=='anchor_mompos': m=m&(q.mom120raw>0)
    if not m.any(): return q
    iv=pct_rank(q,m,'ivol60',True); ef=pct_rank(q,m,'eff120',False)
    if signal in ('anchor','anchor_rmompos','anchor_mompos'): raw=.60*iv+.40*ef
    elif signal=='tstat': raw=.60*iv+.40*pct_rank(q,m,'tstat120',False)
    elif signal=='tri_tstat': raw=.50*iv+.25*ef+.25*pct_rank(q,m,'tstat120',False)
    elif signal=='tri_rmom': raw=.55*iv+.30*ef+.15*pct_rank(q,m,'rmom126',False)
    elif signal=='tri_pos': raw=.55*iv+.30*ef+.15*pct_rank(q,m,'posratio120',False)
    elif signal=='tri_dd': raw=.55*iv+.30*ef+.15*pct_rank(q,m,'dd120',False)
    elif signal=='tri_volshock': raw=.55*iv+.30*ef+.15*pct_rank(q,m,'volshock',True)
    else: raise ValueError(signal)
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q


def subset(q,hold):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,round(hold/5)); chosen=set(dates[::step])
    cols=['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']
    z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')


def fast_simulate(panel,cal,members,cost_mult=1.0):
    # Exact signal/target/execution mechanics, sparse MTM only at rebalance dates + exact final close.
    by={d:g.set_index('code',drop=False) for d,g in panel.groupby('signal_date')}; dates=sorted(by)
    cash=sim.INITIAL_CASH; pos={}; equity=[]; trades=[]; timing=[]; turnover=0.0
    member_end=members.groupby('code').end.max().to_dict(); slip=sim.SLIPPAGE*cost_mult
    for d in dates:
        g=by[d]; td=pd.Timestamp(g.trade_date.iloc[0]); target=hard.choose_det(g.reset_index(drop=True),set(pos)); tgt=set(target)
        for c,pp in list(pos.items()):
            if c in g.index and np.isfinite(g.loc[c].exec_open): pp.last_price=float(g.loc[c].exec_open)
            elif pd.Timestamp(member_end.get(c,sim.END))<td:
                old=pos.pop(c); trades.append({'variant':'fast','code':c,'entry_date':old.entry_date,'exit_date':td,'net_pnl':-old.entry_cost,'net_return':-1.0,'exit_reason':'membership_end_writeoff'})
        nav_open=cash+sum(pp.units*pp.last_price for pp in pos.values())
        for c in sorted(list(pos)):
            if c in tgt or c not in g.index: continue
            r=g.loc[c]; locked=(np.isfinite(r.exec_open) and np.isfinite(r.exec_high) and np.isfinite(r.exec_low) and abs(float(r.exec_high)-float(r.exec_low))<1e-12 and abs(float(r.exec_open)-float(r.exec_high))<1e-12)
            if locked: continue
            px=float(r.exec_open)*(1-slip); gross=pos[c].units*px; cost=sim.fee(gross,'sell',td,cost_mult); old=pos.pop(c); cash+=gross-cost; turnover+=gross
            trades.append({'variant':'fast','code':c,'entry_date':old.entry_date,'exit_date':td,'net_pnl':gross-cost-old.entry_cost,'net_return':(gross-cost)/old.entry_cost-1,'exit_reason':'rank_exit'}); timing.append({'variant':'fast','signal_date':pd.Timestamp(d),'trade_date':td,'side':'sell','code':c})
        per=nav_open*.99/hard.N_HOLD
        for c in target:
            if len(pos)>=hard.N_HOLD: break
            if c in pos or c not in g.index: continue
            r=g.loc[c]; locked=(np.isfinite(r.exec_open) and np.isfinite(r.exec_high) and np.isfinite(r.exec_low) and abs(float(r.exec_high)-float(r.exec_low))<1e-12 and abs(float(r.exec_open)-float(r.exec_high))<1e-12)
            if locked: continue
            factor=float(r.exec_factor) if np.isfinite(r.exec_factor) and r.exec_factor>0 else 1.0; adjpx=float(r.exec_open)*(1+slip); rawpx=adjpx/factor
            if rawpx<=0: continue
            maxraw=max(0,int(abs(float(r.exec_volume))*factor*sim.VOLUME_PARTICIPATION//100)*100); shares=int(min(per,cash*.98)//(rawpx*100))*100
            if maxraw>0: shares=min(shares,maxraw)
            if shares<=0: continue
            units=shares/factor; gross=units*adjpx; cost=sim.fee(gross,'buy',td,cost_mult); total=gross+cost
            if total>cash: continue
            cash-=total; pos[c]=sim.Pos(units,total,td,float(r.exec_open)); turnover+=gross; timing.append({'variant':'fast','signal_date':pd.Timestamp(d),'trade_date':td,'side':'buy','code':c})
        if len(pos)>hard.N_HOLD: raise RuntimeError('fast cap violation')
        nav=cash+sum(pp.units*pp.last_price for pp in pos.values()); equity.append({'variant':'fast','signal_date':pd.Timestamp(d),'trade_date':td,'equity':nav,'cash':cash,'positions':len(pos)})
    # exact final close valuation
    if pos:
        for c,pp in pos.items():
            s=base.qb.read_bin(c,'close',cal).loc[:sim.END].dropna()
            if len(s): pp.last_price=float(s.iloc[-1])
    nav=cash+sum(pp.units*pp.last_price for pp in pos.values()); equity.append({'variant':'fast','signal_date':pd.NaT,'trade_date':sim.END,'equity':nav,'cash':cash,'positions':len(pos)})
    e=pd.DataFrame(equity).drop_duplicates('trade_date',keep='last').sort_values('trade_date'); t=pd.DataFrame(trades); tm=pd.DataFrame(timing)
    if len(tm) and (pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).any(): raise RuntimeError('fast timing violation')
    return e,t,tm,turnover


def period_metrics(eq,a,b):
    z=eq[(pd.to_datetime(eq.trade_date)>=pd.Timestamp(a))&(pd.to_datetime(eq.trade_date)<=pd.Timestamp(b))].copy()
    if len(z)<5: return {'return':np.nan,'cagr':np.nan,'mdd':np.nan,'sharpe':np.nan}
    s=z.set_index(pd.to_datetime(z.trade_date)).equity.astype(float); total=float(s.iloc[-1]/s.iloc[0]-1); yrs=max((s.index[-1]-s.index[0]).days/365.25,1e-9)
    cagr=float((s.iloc[-1]/s.iloc[0])**(1/yrs)-1); dd=s/s.cummax()-1; r=s.pct_change().dropna(); sh=float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else np.nan
    return {'return':total,'cagr':cagr,'mdd':float(dd.min()),'sharpe':sh}


def run(q,hold,n,entry,keep,cal,members,bm,cost=1.0,fast=False):
    oldn=hard.N_HOLD; olde=sim.ENTRY_PCT; oldk=sim.KEEP_PCT; hard.N_HOLD=n; sim.ENTRY_PCT=entry; sim.KEEP_PCT=keep
    try:
        z=subset(q,hold); eq,tr,tm,to=(fast_simulate(z,cal,members,cost) if fast else hard.hard_simulate(z,cal,members,cost)); st=sim.perf(eq,tr,to,bm if not fast else None)
        st.update({'hold_days':hold,'n_hold':n,'entry_pct':entry,'keep_pct':keep,'cost_mult':cost,'positions_max':int(eq.positions.max()),'positions_median':float(eq.positions.median()),'discovery_fast':int(fast)})
        a=period_metrics(eq,'2016-07-29','2021-12-31'); b=period_metrics(eq,'2022-01-01','2026-07-29')
        for k,v in a.items(): st['train_'+k]=v
        for k,v in b.items(): st['validation_'+k]=v
        return st,eq,tr,tm
    finally: hard.N_HOLD=oldn; sim.ENTRY_PCT=olde; sim.KEEP_PCT=oldk


def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=add_grand_fields(p,cal,members)
    keepcols=['signal_date','trade_date','code','liq20','liq_rank_pct','exec_open','exec_high','exec_low','exec_volume','exec_factor','ivol40','ivol60','ivol80','eff120','skew40','rmom126','tstat120','posratio120','dd120','mom120raw','volshock','age_days','board']
    p=p[keepcols].copy(); bm=market_close.loc[sim.START:sim.END].dropna(); phase1=[]

    # A. broad signal/pool discovery; sparse MTM, exact execution/terminal wealth.
    for sig in SIGNALS:
        for pool in POOLS:
            q=rerank(p,sig,pool); print('PHASE1',sig,pool,flush=True); st,_,_,_=run(q,60,20,.10,.30,cal,members,bm,fast=True); st.update({'signal':sig,'pool':pool}); phase1.append(st); del q
    p1=pd.DataFrame(phase1); p1.to_csv(OUT/'phase1_signal_pool.csv',index=False)
    eligible=p1[(p1.train_return>0)&(p1.train_mdd>-0.35)&(p1.positions_max<=p1.n_hold)].sort_values(['train_cagr','train_mdd'],ascending=[False,False])
    top_pairs=[(str(r.signal),str(r.pool)) for _,r in eligible.head(2).iterrows()]
    if len(top_pairs)<2: raise RuntimeError('insufficient phase1 candidates')
    print('TOP PAIRS TRAIN ONLY',top_pairs,flush=True)

    # B. construction discovery only on the two training winners.
    phase2=[]
    for sig,pool in top_pairs:
        q=rerank(p,sig,pool)
        for n in NS:
            for h in HOLDS:
                for entry,keep in BUFFERS:
                    print('PHASE2',sig,pool,n,h,entry,keep,flush=True); st,_,_,_=run(q,h,n,entry,keep,cal,members,bm,fast=True); st.update({'signal':sig,'pool':pool}); phase2.append(st)
        del q
    p2=pd.DataFrame(phase2); p2.to_csv(OUT/'phase2_construction.csv',index=False)
    ok=p2[(p2.train_return>0)&(p2.train_mdd>-0.35)&(p2.positions_max<=p2.n_hold)].sort_values(['train_cagr','train_mdd'],ascending=[False,False])

    # C. exact daily MTM audit for top 12 training candidates only.
    exact=[]; exact_cache={}
    top12=ok.head(12).copy()
    for sig,pool in top12[['signal','pool']].drop_duplicates().itertuples(index=False,name=None):
        q=rerank(p,str(sig),str(pool)); sub=top12[(top12.signal==sig)&(top12.pool==pool)]
        for _,r in sub.iterrows():
            key=(str(sig),str(pool),int(r.n_hold),int(r.hold_days),float(r.entry_pct),float(r.keep_pct)); print('EXACT',key,flush=True)
            st,eq,tr,tm=run(q,key[3],key[2],key[4],key[5],cal,members,bm,fast=False); st.update({'signal':key[0],'pool':key[1]}); exact.append(st); exact_cache[key]=(eq,tr,tm)
        del q
    ex=pd.DataFrame(exact).sort_values(['train_cagr','train_mdd'],ascending=[False,False]); ex.to_csv(OUT/'exact_top12.csv',index=False)
    best=ex.iloc[0]; key=(str(best.signal),str(best.pool),int(best.n_hold),int(best.hold_days),float(best.entry_pct),float(best.keep_pct)); eq,tr,tm=exact_cache[key]
    pd.DataFrame([best]).to_csv(OUT/'winner_summary.csv',index=False)

    # D. exact local parameter neighborhood around winner: one coordinate at a time.
    q=rerank(p,key[0],key[1]); local_keys=set()
    for n in NS: local_keys.add((n,key[3],key[4],key[5]))
    for h in HOLDS: local_keys.add((key[2],h,key[4],key[5]))
    for e,k in BUFFERS: local_keys.add((key[2],key[3],e,k))
    local=[]
    for n,h,e,k in sorted(local_keys):
        print('LOCAL EXACT',n,h,e,k,flush=True); st,_,_,_=run(q,h,n,e,k,cal,members,bm,fast=False); st.update({'signal':key[0],'pool':key[1]}); local.append(st)
    loc=pd.DataFrame(local); loc.to_csv(OUT/'winner_local_exact.csv',index=False)
    plateau={'signal':key[0],'pool':key[1],'exact_local_points':len(loc),'median_total_return':float(loc.total_return.median()),'min_total_return':float(loc.total_return.min()),'max_total_return':float(loc.total_return.max()),'median_validation_return':float(loc.validation_return.median()),'min_validation_return':float(loc.validation_return.min()),'positive_validation_points':int((loc.validation_return>0).sum()),'median_mdd':float(loc.max_drawdown.median()),'worst_mdd':float(loc.max_drawdown.min())}
    pd.DataFrame([plateau]).to_csv(OUT/'winner_plateau.csv',index=False)

    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=run(q,key[3],key[2],key[4],key[5],cal,members,bm,cm,fast=False); st.update({'signal':key[0],'pool':key[1]}); costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'winner_cost.csv',index=False)
    ann=sim.annual_returns(eq); ann.to_csv(OUT/'winner_annual.csv',index=False); rob=sim.robustness(eq,tr); pd.DataFrame([rob]).to_csv(OUT/'winner_robust.csv',index=False)
    blocks=[]
    for label,a,b in [('2016_2018','2016-07-29','2018-12-31'),('2019_2021','2019-01-01','2021-12-31'),('2022_2024','2022-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-07-29')]:
        z=period_metrics(eq,a,b); z.update({'block':label}); blocks.append(z)
    pd.DataFrame(blocks).to_csv(OUT/'winner_blocks.csv',index=False)
    bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    audit={**ua,'market_factor':market_code,'signals':'|'.join(SIGNALS),'pools':'|'.join(POOLS),'phase1_points':len(p1),'phase2_points':len(p2),'exact_finalists':len(ex),'selection':'all selection on 2016-2021; 2022-2026 reused validation, NOT untouched OOS','discovery':'exact execution + sparse MTM + exact final close','final':'daily MTM deterministic hard execution; trapped positions occupy slots','timing_violations':bad,'winner_positions_within_target':int(best.positions_max<=best.n_hold)}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad or not audit['winner_positions_within_target']: raise RuntimeError('grand audit failed')

    print('=== PHASE1 TOP TRAIN ==='); print(eligible.head(20).to_string(index=False),flush=True)
    print('=== PHASE2 TOP TRAIN ==='); print(ok.head(30).to_string(index=False),flush=True)
    print('=== EXACT FINALISTS ==='); print(ex.to_string(index=False),flush=True)
    print('=== WINNER ==='); print(pd.DataFrame([best]).to_string(index=False),flush=True)
    print('=== LOCAL ==='); print(loc.sort_values('total_return',ascending=False).to_string(index=False),flush=True)
    print('=== PLATEAU ==='); print(pd.DataFrame([plateau]).to_string(index=False),flush=True)
    print('=== BLOCKS ==='); print(pd.DataFrame(blocks).to_string(index=False),flush=True)
    print('=== ANNUAL ==='); print(ann.to_string(index=False),flush=True)
    print('=== COST ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== ROBUST ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)

if __name__=='__main__': main()
