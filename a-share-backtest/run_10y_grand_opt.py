from __future__ import annotations
from pathlib import Path
import math
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
    for c in ('tstat120','posratio120','dd120','mom120raw'):
        p[c]=np.nan
    groups=p.groupby('code').groups
    for i,(code,idx) in enumerate(groups.items(),1):
        c=base.qb.read_bin(code,'close',cal).loc[sim.WARM:sim.END]
        if c.empty: continue
        r=c.pct_change(fill_method=None)
        y=np.log(c.where(c>0))
        tt=pd.Series(np.arange(len(y),dtype=float),index=y.index)
        corr=y.rolling(120,min_periods=100).corr(tt).clip(-.999999,.999999)
        tstat=corr*np.sqrt(118.0/(1-corr*corr))
        pos=(r>0).where(r.notna()).astype(float).rolling(120,min_periods=100).mean()
        dd=c/c.rolling(120,min_periods=100).max()-1
        mom=c/c.shift(120)-1
        ds=pd.DatetimeIndex(p.loc[idx,'signal_date'])
        p.loc[idx,'tstat120']=tstat.reindex(ds).to_numpy(float)
        p.loc[idx,'posratio120']=pos.reindex(ds).to_numpy(float)
        p.loc[idx,'dd120']=dd.reindex(ds).to_numpy(float)
        p.loc[idx,'mom120raw']=mom.reindex(ds).to_numpy(float)
        if i%1000==0: print('grand fields',i,'/',len(groups),flush=True)
    first=members.groupby('code').start.min()
    p['first_start']=p.code.map(first)
    p['age_days']=(pd.to_datetime(p.signal_date)-pd.to_datetime(p.first_start)).dt.days
    def board(code):
        s=str(code)
        if s.startswith('BJ'): return 'BJ'
        if s.startswith('SH688'): return 'STAR'
        if s.startswith('SZ300') or s.startswith('SZ301'): return 'CHINEXT'
        return 'MAIN'
    p['board']=p.code.map(board)
    p['volshock']=p.ivol40/p.ivol80.replace(0,np.nan)
    return p


def pool_mask(q,name):
    m=np.isfinite(q.ivol60)&np.isfinite(q.eff120)&np.isfinite(q.skew40)
    liq={'all80':.80,'liq60':.60,'liq70':.70,'liq90':.90,'mature365':.80,'mature730':.80,'ex_star_bj':.80,'quality_pool':.70}[name]
    m=m&(q.liq_rank_pct<=liq)
    if name=='mature365': m=m&(q.age_days>=365)
    if name=='mature730': m=m&(q.age_days>=730)
    if name=='ex_star_bj': m=m&(~q.board.isin(['STAR','BJ']))
    if name=='quality_pool': m=m&(q.age_days>=365)&(~q.board.isin(['STAR','BJ']))
    # fixed anti-lottery filter: remove highest 20% residual skewness cross-sectionally
    mm=q.loc[m]
    sp=mm.groupby('signal_date').skew40.rank(pct=True,method='average',ascending=True)
    ok=pd.Series(False,index=q.index); ok.loc[sp.index]=sp<=.80
    return m&ok


def pct_rank(q,m,col,ascending):
    return q.loc[m].groupby('signal_date')[col].rank(pct=True,method='average',ascending=ascending)


def rerank(p,signal,pool):
    q=p.copy(); q['rank_test']=np.nan
    m=pool_mask(q,pool)
    needed=[]
    if signal in ('tstat','tri_tstat'): needed=['tstat120']
    elif signal=='tri_rmom' or signal=='anchor_rmompos': needed=['rmom126']
    elif signal=='tri_pos': needed=['posratio120']
    elif signal=='tri_dd': needed=['dd120']
    elif signal=='tri_volshock': needed=['volshock']
    elif signal=='anchor_mompos': needed=['mom120raw']
    for c in needed: m=m&np.isfinite(q[c])
    if signal=='anchor_rmompos': m=m&(q.rmom126>0)
    if signal=='anchor_mompos': m=m&(q.mom120raw>0)
    if not m.any(): return q
    iv=pct_rank(q,m,'ivol60',True)
    ef=pct_rank(q,m,'eff120',False)
    if signal in ('anchor','anchor_rmompos','anchor_mompos'):
        raw=.60*iv+.40*ef
    elif signal=='tstat':
        ts=pct_rank(q,m,'tstat120',False); raw=.60*iv+.40*ts
    elif signal=='tri_tstat':
        ts=pct_rank(q,m,'tstat120',False); raw=.50*iv+.25*ef+.25*ts
    elif signal=='tri_rmom':
        rr=pct_rank(q,m,'rmom126',False); raw=.55*iv+.30*ef+.15*rr
    elif signal=='tri_pos':
        pr=pct_rank(q,m,'posratio120',False); raw=.55*iv+.30*ef+.15*pr
    elif signal=='tri_dd':
        dr=pct_rank(q,m,'dd120',False); raw=.55*iv+.30*ef+.15*dr
    elif signal=='tri_volshock':
        vr=pct_rank(q,m,'volshock',True); raw=.55*iv+.30*ef+.15*vr
    else: raise ValueError(signal)
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q


def subset(q,hold):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
    step=max(1,round(hold/5)); chosen=set(dates[::step])
    cols=['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']
    z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')


def period_metrics(eq,a,b):
    z=eq[(pd.to_datetime(eq.trade_date)>=pd.Timestamp(a))&(pd.to_datetime(eq.trade_date)<=pd.Timestamp(b))].copy()
    if len(z)<20: return {'return':np.nan,'cagr':np.nan,'mdd':np.nan,'sharpe':np.nan}
    s=z.set_index(pd.to_datetime(z.trade_date)).equity.astype(float)
    total=float(s.iloc[-1]/s.iloc[0]-1)
    yrs=max((s.index[-1]-s.index[0]).days/365.25,1e-9)
    cagr=float((s.iloc[-1]/s.iloc[0])**(1/yrs)-1)
    dd=s/s.cummax()-1; r=s.pct_change().dropna()
    sh=float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else np.nan
    return {'return':total,'cagr':cagr,'mdd':float(dd.min()),'sharpe':sh}


def run(q,hold,n,entry,keep,cal,members,bm,cost=1.0):
    oldn=hard.N_HOLD; olde=sim.ENTRY_PCT; oldk=sim.KEEP_PCT
    hard.N_HOLD=n; sim.ENTRY_PCT=entry; sim.KEEP_PCT=keep
    try:
        z=subset(q,hold); eq,tr,tm,to=hard.hard_simulate(z,cal,members,cost); st=sim.perf(eq,tr,to,bm)
        st.update({'hold_days':hold,'n_hold':n,'entry_pct':entry,'keep_pct':keep,'cost_mult':cost,'positions_max':int(eq.positions.max()),'positions_median':float(eq.positions.median())})
        a=period_metrics(eq,'2016-07-29','2021-12-31'); b=period_metrics(eq,'2022-01-01','2026-07-29')
        for k,v in a.items(): st['train_'+k]=v
        for k,v in b.items(): st['validation_'+k]=v
        return st,eq,tr,tm
    finally:
        hard.N_HOLD=oldn; sim.ENTRY_PCT=olde; sim.KEEP_PCT=oldk


def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=add_grand_fields(p,cal,members)
    bm=market_close.loc[sim.START:sim.END].dropna(); phase1=[]; qcache={}
    # Phase A: signal/pool structure only. Construction frozen.
    for sig in SIGNALS:
        for pool in POOLS:
            q=rerank(p,sig,pool); qcache[(sig,pool)]=q
            print('PHASE1',sig,pool,flush=True)
            st,eq,tr,tm=run(q,60,20,.10,.30,cal,members,bm); st.update({'signal':sig,'pool':pool}); phase1.append(st)
    p1=pd.DataFrame(phase1); p1.to_csv(OUT/'phase1_signal_pool.csv',index=False)
    eligible=p1[(p1.train_return>0)&(p1.train_mdd>-0.35)&(p1.positions_max<=p1.n_hold)].copy()
    eligible=eligible.sort_values(['train_cagr','train_mdd'],ascending=[False,False])
    top_pairs=[(str(r.signal),str(r.pool)) for _,r in eligible.head(2).iterrows()]
    if len(top_pairs)<2: raise RuntimeError('insufficient phase1 candidates')
    print('TOP PAIRS TRAIN ONLY',top_pairs,flush=True)

    # Phase B: construction optimization only for the two training winners.
    phase2=[]; cache={}
    for sig,pool in top_pairs:
        q=qcache[(sig,pool)]
        for n in NS:
            for h in HOLDS:
                for entry,keep in BUFFERS:
                    print('PHASE2',sig,pool,'N',n,'H',h,'buffer',entry,keep,flush=True)
                    st,eq,tr,tm=run(q,h,n,entry,keep,cal,members,bm); st.update({'signal':sig,'pool':pool}); phase2.append(st); cache[(sig,pool,n,h,entry,keep)]=(q,eq,tr,tm)
    p2=pd.DataFrame(phase2); p2.to_csv(OUT/'phase2_construction.csv',index=False)
    ok=p2[(p2.train_return>0)&(p2.train_mdd>-0.35)&(p2.positions_max<=p2.n_hold)].sort_values(['train_cagr','train_mdd'],ascending=[False,False])
    best=ok.iloc[0]
    key=(str(best.signal),str(best.pool),int(best.n_hold),int(best.hold_days),float(best.entry_pct),float(best.keep_pct))
    q,eq,tr,tm=cache[key]

    # Neighbor/plateau summary for final signal/pool across all construction points.
    neigh=p2[(p2.signal==key[0])&(p2.pool==key[1])].copy()
    neigh.to_csv(OUT/'winner_neighborhood.csv',index=False)
    plateau={
        'signal':key[0],'pool':key[1],'points':len(neigh),
        'median_total_return':float(neigh.total_return.median()),'min_total_return':float(neigh.total_return.min()),'max_total_return':float(neigh.total_return.max()),
        'median_train_cagr':float(neigh.train_cagr.median()),'min_train_cagr':float(neigh.train_cagr.min()),
        'median_validation_return':float(neigh.validation_return.median()),'min_validation_return':float(neigh.validation_return.min()),
        'positive_validation_points':int((neigh.validation_return>0).sum()),
        'median_mdd':float(neigh.max_drawdown.median()),'worst_mdd':float(neigh.max_drawdown.min()),
    }
    pd.DataFrame([plateau]).to_csv(OUT/'winner_plateau.csv',index=False)

    # Diagnostics for training-selected winner. Validation is reported, not used to choose it.
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=run(q,key[3],key[2],key[4],key[5],cal,members,bm,cm); st.update({'signal':key[0],'pool':key[1]}); costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'winner_cost.csv',index=False)
    ann=sim.annual_returns(eq); ann.to_csv(OUT/'winner_annual.csv',index=False)
    rob=sim.robustness(eq,tr); pd.DataFrame([rob]).to_csv(OUT/'winner_robust.csv',index=False)
    pd.DataFrame([best]).to_csv(OUT/'winner_summary.csv',index=False)

    # Era blocks to expose regime dependence.
    blocks=[]
    for label,a,b in [('2016_2018','2016-07-29','2018-12-31'),('2019_2021','2019-01-01','2021-12-31'),('2022_2024','2022-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-07-29')]:
        z=period_metrics(eq,a,b); z.update({'block':label}); blocks.append(z)
    pd.DataFrame(blocks).to_csv(OUT/'winner_blocks.csv',index=False)

    allt=[]
    for x in cache.values():
        if len(x[3]): allt.append(x[3])
    alltm=pd.concat(allt,ignore_index=True) if allt else pd.DataFrame()
    bad=int((pd.to_datetime(alltm.signal_date)>=pd.to_datetime(alltm.trade_date)).sum()) if len(alltm) else 0
    audit={**ua,'market_factor':market_code,'signals':'|'.join(SIGNALS),'pools':'|'.join(POOLS),'phase1_points':len(p1),'phase2_points':len(p2),'selection':'signal/pool and construction chosen on 2016-2021 only; 2022-2026 reused validation, NOT untouched OOS','execution':'deterministic ranked keep; trapped unsold positions consume slots; hard cap','timing_violations':bad,'all_positions_within_target':int((p2.positions_max<=p2.n_hold).all())}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad or not audit['all_positions_within_target']: raise RuntimeError('grand audit failed')

    print('=== PHASE1 TOP TRAIN ==='); print(eligible.head(20).to_string(index=False),flush=True)
    print('=== PHASE2 TOP TRAIN ==='); print(ok.head(30).to_string(index=False),flush=True)
    print('=== WINNER ==='); print(pd.DataFrame([best]).to_string(index=False),flush=True)
    print('=== PLATEAU ==='); print(pd.DataFrame([plateau]).to_string(index=False),flush=True)
    print('=== BLOCKS ==='); print(pd.DataFrame(blocks).to_string(index=False),flush=True)
    print('=== ANNUAL ==='); print(ann.to_string(index=False),flush=True)
    print('=== COST ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== ROBUST ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)

if __name__=='__main__': main()
