from __future__ import annotations
from pathlib import Path
import itertools, math
import numpy as np
import pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_skewfilter_hard as hard
import run_10y_grand_opt as grand

OUT=Path('results_mega_price'); OUT.mkdir(exist_ok=True)
SIGNALS=(
    'base','lowbeta','downside','r2trend','lowmax','lowgap','lowrange','lowamtcv',
    'lowamount','rmomsoft','defensive','smooth','tailquality','sizeproxy'
)
FILTERS=('none','max80','gap80','range80','beta80','amount50')
REGIMES=('always','mkt120','mkt200','breadth45','mkt200_or_breadth45')
NS=(10,15,20)
HOLDS=(20,40,60,80)
BUFFERS=((.05,.20),(.10,.30))
TOP_EXACT=20


def add_mega_fields(p,cal,market_close):
    q=p.copy()
    cols=('beta252','downbeta252','downsemivol60','max20','max60','gapvol60','range20','amtcv60','r2trend120','amount20')
    for c in cols: q[c]=np.nan
    mr=market_close.pct_change(fill_method=None)
    mvar=mr.rolling(252,min_periods=180).var()
    groups=q.groupby('code').groups
    for j,(code,idx) in enumerate(groups.items(),1):
        c=base.qb.read_bin(code,'close',cal); o=base.qb.read_bin(code,'open',cal); h=base.qb.read_bin(code,'high',cal); l=base.qb.read_bin(code,'low',cal); v=base.qb.read_bin(code,'volume',cal)
        if c.empty: continue
        d=pd.concat([c.rename('c'),o.rename('o'),h.rename('h'),l.rename('l'),v.rename('v'),mr.rename('m')],axis=1)
        r=d.c.pct_change(fill_method=None)
        cov=r.rolling(252,min_periods=180).cov(d.m); beta=cov/mvar.reindex(cov.index).replace(0,np.nan)
        neg=(d.m<0)&r.notna()&d.m.notna(); num=(r*d.m).where(neg).rolling(252,min_periods=120).sum(); den=(d.m*d.m).where(neg).rolling(252,min_periods=120).sum(); db=num/den.replace(0,np.nan)
        dsemi=np.sqrt((r.clip(upper=0)**2).rolling(60,min_periods=45).mean())
        mx20=r.rolling(20,min_periods=15).max(); mx60=r.rolling(60,min_periods=45).max()
        gap=d.o/d.c.shift(1)-1; gv=gap.rolling(60,min_periods=45).std()
        rng=d.h/d.l-1; rg=rng.rolling(20,min_periods=15).mean()
        amt=(d.c.abs()*d.v.abs()).replace(0,np.nan); am=amt.rolling(20,min_periods=15).mean(); acv=amt.rolling(60,min_periods=45).std()/amt.rolling(60,min_periods=45).mean().replace(0,np.nan)
        y=np.log(d.c.where(d.c>0)); tt=pd.Series(np.arange(len(y),dtype=float),index=y.index); cr=y.rolling(120,min_periods=100).corr(tt); r2=cr*cr
        ds=pd.DatetimeIndex(q.loc[idx,'signal_date'])
        vals={'beta252':beta,'downbeta252':db,'downsemivol60':dsemi,'max20':mx20,'max60':mx60,'gapvol60':gv,'range20':rg,'amtcv60':acv,'r2trend120':r2,'amount20':am}
        for k,s in vals.items(): q.loc[idx,k]=s.reindex(ds).to_numpy(float)
        if j%1000==0: print('MEGA FIELDS',j,'/',len(groups),flush=True)
    return q


def base_mask(q):
    m=np.isfinite(q.ivol60)&np.isfinite(q.eff120)&np.isfinite(q.skew40)&(q.liq_rank_pct<=.70)
    sk=q.loc[m].groupby('signal_date').skew40.rank(pct=True,method='average',ascending=True)
    ok=pd.Series(False,index=q.index); ok.loc[sk.index]=sk<=.80
    return m&ok


def pct(q,m,col,ascending):
    return q.loc[m].groupby('signal_date')[col].rank(pct=True,method='average',ascending=ascending)


def apply_filter(q,m,name):
    if name=='none': return m
    mapping={'max80':('max60',True,.80),'gap80':('gapvol60',True,.80),'range80':('range20',True,.80),'beta80':('beta252',True,.80),'amount50':('amount20',True,.50)}
    c,asc,cut=mapping[name]; mm=m&np.isfinite(q[c])
    rr=pct(q,mm,c,asc); ok=pd.Series(False,index=q.index); ok.loc[rr.index]=rr<=cut
    return mm&ok


def make_rank(p,signal,flt):
    q=p.copy(); q['rank_test']=np.nan; m=apply_filter(q,base_mask(q),flt)
    need=[]
    if signal=='lowbeta': need=['beta252']
    elif signal=='downside': need=['downbeta252','downsemivol60']
    elif signal=='r2trend': need=['r2trend120']
    elif signal=='lowmax': need=['max60']
    elif signal=='lowgap': need=['gapvol60']
    elif signal=='lowrange': need=['range20']
    elif signal=='lowamtcv': need=['amtcv60']
    elif signal in ('lowamount','sizeproxy'): need=['amount20']
    elif signal=='rmomsoft': need=['rmom126']
    elif signal=='defensive': need=['beta252','downsemivol60']
    elif signal=='smooth': need=['r2trend120','gapvol60','range20']
    elif signal=='tailquality': need=['max60','downsemivol60']
    for c in need: m=m&np.isfinite(q[c])
    if not m.any(): return q
    iv=pct(q,m,'ivol60',True); ef=pct(q,m,'eff120',False)
    if signal=='base': raw=.60*iv+.40*ef
    elif signal=='lowbeta': raw=.52*iv+.33*ef+.15*pct(q,m,'beta252',True)
    elif signal=='downside': raw=.50*iv+.30*ef+.10*pct(q,m,'downbeta252',True)+.10*pct(q,m,'downsemivol60',True)
    elif signal=='r2trend': raw=.50*iv+.30*ef+.20*pct(q,m,'r2trend120',False)
    elif signal=='lowmax': raw=.52*iv+.33*ef+.15*pct(q,m,'max60',True)
    elif signal=='lowgap': raw=.52*iv+.33*ef+.15*pct(q,m,'gapvol60',True)
    elif signal=='lowrange': raw=.52*iv+.33*ef+.15*pct(q,m,'range20',True)
    elif signal=='lowamtcv': raw=.52*iv+.33*ef+.15*pct(q,m,'amtcv60',True)
    elif signal=='lowamount': raw=.50*iv+.30*ef+.20*pct(q,m,'amount20',True)
    elif signal=='rmomsoft': raw=.50*iv+.30*ef+.20*pct(q,m,'rmom126',False)
    elif signal=='defensive': raw=.44*iv+.26*ef+.15*pct(q,m,'beta252',True)+.15*pct(q,m,'downsemivol60',True)
    elif signal=='smooth': raw=.42*iv+.26*ef+.12*pct(q,m,'r2trend120',False)+.10*pct(q,m,'gapvol60',True)+.10*pct(q,m,'range20',True)
    elif signal=='tailquality': raw=.44*iv+.26*ef+.15*pct(q,m,'max60',True)+.15*pct(q,m,'downsemivol60',True)
    elif signal=='sizeproxy': raw=.45*iv+.25*ef+.30*pct(q,m,'amount20',True)
    else: raise ValueError(signal)
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q


def regime_map(p,market_close):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(p.signal_date.unique())))
    mc=market_close.sort_index(); m120=mc/mc.shift(120)-1; ma200=mc.rolling(200,min_periods=160).mean(); up200=mc>ma200
    tmp=p[['signal_date','code','mom120raw','liq_rank_pct']].copy(); z=tmp[(tmp.liq_rank_pct<=.70)&np.isfinite(tmp.mom120raw)]
    breadth=z.groupby('signal_date').mom120raw.apply(lambda x: float((x>0).mean()))
    out=pd.DataFrame(index=dates); out['mkt120']=m120.reindex(dates).to_numpy(float)>0; out['mkt200']=up200.reindex(dates).fillna(False).to_numpy(bool); out['breadth45']=breadth.reindex(dates).fillna(0).to_numpy(float)>.45; out['always']=True; out['mkt200_or_breadth45']=out.mkt200|out.breadth45
    out.to_csv(OUT/'regime_state.csv'); return out


def apply_regime(q,regime,state):
    if regime=='always': return q
    x=q.copy(); ok=state[regime].to_dict(); risk=pd.to_datetime(x.signal_date).map(ok).fillna(False).to_numpy(bool); x.loc[~risk,'rank_test']=np.nan; return x


def period(eq,a,b): return grand.period_metrics(eq,a,b)


def run_fast(q,h,n,e,k,cal,members,bm): return grand.run(q,h,n,e,k,cal,members,bm,fast=True)
def run_exact(q,h,n,e,k,cal,members,bm,cost=1.): return grand.run(q,h,n,e,k,cal,members,bm,cost=cost,fast=False)


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members); p=add_mega_fields(p,cal,market_close)
    state=regime_map(p,market_close); bm=market_close.loc[sim.START:sim.END].dropna()
    keep=['signal_date','trade_date','code','liq20','liq_rank_pct','exec_open','exec_high','exec_low','exec_volume','exec_factor','ivol60','eff120','skew40','rmom126','mom120raw','beta252','downbeta252','downsemivol60','max20','max60','gapvol60','range20','amtcv60','r2trend120','amount20']
    p=p[keep].copy()

    # Stage 1: broad structural search. Selection metrics use 2016-2021 only.
    rows=[]; structures={}
    for sig,flt,reg in itertools.product(SIGNALS,FILTERS,REGIMES):
        q=apply_regime(make_rank(p,sig,flt),reg,state); key=(sig,flt,reg); structures[key]=q
        st,_,_,_=run_fast(q,60,20,.10,.30,cal,members,bm); st.update({'signal':sig,'filter':flt,'regime':reg}); rows.append(st)
        print('S1',sig,flt,reg,'TRAIN',st.get('train_return'),'FULL',st.get('total_return'),flush=True)
    s1=pd.DataFrame(rows); s1.to_csv(OUT/'stage1.csv',index=False)
    elig=s1[(s1.train_return>0)&(s1.train_mdd>-0.40)&(s1.positions_max<=20)].copy(); elig['train_score']=elig.train_cagr-.20*elig.train_mdd.abs(); elig=elig.sort_values(['train_score','train_cagr'],ascending=False)
    topkeys=[(r.signal,r['filter'],r.regime) for _,r in elig.head(8).iterrows()]
    pd.DataFrame(topkeys,columns=['signal','filter','regime']).to_csv(OUT/'stage1_selected.csv',index=False)

    # Stage 2: coarse construction search on top 8 train structures.
    rows2=[]; configs=[]
    for key in topkeys:
        q=structures[key]
        for n,h,(e,k) in itertools.product(NS,HOLDS,BUFFERS):
            st,_,_,_=run_fast(q,h,n,e,k,cal,members,bm); st.update({'signal':key[0],'filter':key[1],'regime':key[2]}); rows2.append(st)
            configs.append((st,key,n,h,e,k))
    s2=pd.DataFrame(rows2); s2.to_csv(OUT/'stage2.csv',index=False)
    ok=s2[(s2.train_return>0)&(s2.train_mdd>-0.40)&(s2.positions_max<=s2.n_hold)].copy(); ok['train_score']=ok.train_cagr-.20*ok.train_mdd.abs(); ok=ok.sort_values(['train_score','train_cagr'],ascending=False)

    # Stage 3: exact daily MTM on top train-only candidates and also full-sample exploratory max for transparency.
    exact=[]; cache={}
    dedup=[]
    for _,r in ok.head(TOP_EXACT).iterrows():
        cfg=(str(r.signal),str(r['filter']),str(r.regime),int(r.n_hold),int(r.hold_days),float(r.entry_pct),float(r.keep_pct))
        if cfg not in dedup: dedup.append(cfg)
    for cfg in dedup:
        sig,flt,reg,n,h,e,k=cfg; q=structures[(sig,flt,reg)]; st,eq,tr,tm=run_exact(q,h,n,e,k,cal,members,bm); st.update({'signal':sig,'filter':flt,'regime':reg}); exact.append(st); cache[cfg]=(eq,tr,tm)
        print('EXACT',cfg,'FULL',st['total_return'],'TRAIN',st['train_return'],'VAL',st['validation_return'],flush=True)
    ex=pd.DataFrame(exact); ex['train_score']=ex.train_cagr-.20*ex.train_mdd.abs(); ex=ex.sort_values(['train_score','train_cagr'],ascending=False); ex.to_csv(OUT/'exact_train_candidates.csv',index=False)
    win=ex.iloc[0]; wcfg=(str(win.signal),str(win['filter']),str(win.regime),int(win.n_hold),int(win.hold_days),float(win.entry_pct),float(win.keep_pct)); weq,wtr,wtm=cache[wcfg]

    # robustness of train-selected winner
    annual=sim.annual_returns(weq); annual.to_csv(OUT/'winner_annual.csv',index=False)
    rob=sim.robustness(weq,wtr); pd.DataFrame([rob]).to_csv(OUT/'winner_tail.csv',index=False)
    costs=[]
    q=structures[wcfg[:3]]
    for cm in (2.,4.,8.):
        st,_,_,_=run_exact(q,wcfg[4],wcfg[3],wcfg[5],wcfg[6],cal,members,bm,cost=cm); st.update({'cost_mult_test':cm}); costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'winner_costs.csv',index=False)

    # report exploratory full-sample winners but NEVER select them as formal champion
    s2full=s2.sort_values('total_return',ascending=False).head(20); s2full.to_csv(OUT/'exploratory_fullsample_top20_fast.csv',index=False)
    target5=float(win.total_return)>=4.0
    audit={**ua,'market_factor':market_code,'stage1_points':len(s1),'stage2_points':len(s2),'exact_points':len(ex),'selection':'ALL formal selection by 2016-2021 train score only; 2022-2026 reused validation; full-sample top exported exploratory only','winner':str(wcfg),'winner_total_return':float(win.total_return),'winner_cagr':float(win.cagr),'winner_mdd':float(win.max_drawdown),'winner_validation_return':float(win.validation_return),'five_x_target_met':int(target5),'timing_violations':int((pd.to_datetime(wtm.signal_date)>=pd.to_datetime(wtm.trade_date)).sum()) if len(wtm) else 0}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if audit['timing_violations']!=0: raise RuntimeError('timing violation')
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
    print('=== EXACT TRAIN SELECTED ==='); print(ex.to_string(index=False),flush=True)
    print('=== WINNER ANNUAL ==='); print(annual.to_string(index=False),flush=True)
    print('=== WINNER COSTS ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== WINNER TAIL ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== FULL SAMPLE EXPLORATORY FAST TOP20 ==='); print(s2full.to_string(index=False),flush=True)

if __name__=='__main__': main()
