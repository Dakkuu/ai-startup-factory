from __future__ import annotations
from pathlib import Path
import glob, itertools
import numpy as np
import pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_skewfilter_hard as hard
import run_10y_grand_opt as grand

OUT=Path('results_mega_pit'); OUT.mkdir(exist_ok=True)
VARIANTS=(
 'base','size15','size25','size35','size50','small60_base','small50_base','small40_base','small30_base',
 'value20','value30','size25_value15','size35_value15','size25_turn15','size35_turn15','small50_value20','small40_value20','small50_turn20','small40_turn20'
)
NS=(10,15,20); HOLDS=(20,40,60,80); BUFFERS=((.05,.20),(.10,.30)); TOP_EXACT=20


def load_bs():
    fs=sorted(glob.glob('baostock_shards/factor_shard_*.csv.gz'))
    if not fs: raise RuntimeError('no BaoStock shard files')
    z=pd.concat([pd.read_csv(f,compression='gzip',low_memory=False) for f in fs],ignore_index=True)
    z['signal_date']=pd.to_datetime(z.signal_date); z['code']=z.code.astype(str).str.upper()
    for c in ['float_mv_proxy','turn20','peTTM','pbMRQ','psTTM','pcfNcfTTM','isST','tradestatus','exact_obs']:
        z[c]=pd.to_numeric(z[c],errors='coerce')
    z=z.sort_values(['signal_date','code']).drop_duplicates(['signal_date','code'],keep='last')
    return z


def base_mask(q):
    # Strict PIT execution-status filter: no carry-forward status accepted.
    m=np.isfinite(q.ivol60)&np.isfinite(q.eff120)&np.isfinite(q.skew40)&(q.liq_rank_pct<=.70)
    m=m&(q.exact_obs==1)&(q.tradestatus==1)&(q.isST==0)&np.isfinite(q.float_mv_proxy)&(q.float_mv_proxy>0)
    sk=q.loc[m].groupby('signal_date').skew40.rank(pct=True,method='average',ascending=True)
    ok=pd.Series(False,index=q.index); ok.loc[sk.index]=sk<=.80
    return m&ok


def pct(q,m,col,ascending=True):
    return q.loc[m].groupby('signal_date')[col].rank(pct=True,method='average',ascending=ascending)


def add_value_score(q,m):
    # Cross-sectional value ranks. Negative/zero multiples are treated as unavailable rather than 'cheap'.
    comps=[]
    for c,hi in [('pbMRQ',50),('peTTM',500),('psTTM',100),('pcfNcfTTM',500)]:
        mm=m&q[c].gt(0)&q[c].lt(hi)&np.isfinite(q[c])
        r=pd.Series(np.nan,index=q.index,dtype=float); rr=pct(q,mm,c,True); r.loc[rr.index]=rr; comps.append(r)
    A=pd.concat(comps,axis=1); val=A.mean(axis=1,skipna=True); n=A.notna().sum(axis=1); val=val.where(n>=2)
    return val


def rank_variant(p,name):
    q=p.copy(); q['rank_test']=np.nan; m=base_mask(q)
    if not m.any(): return q
    size=pct(q,m,'float_mv_proxy',True); iv=pct(q,m,'ivol60',True); ef=pct(q,m,'eff120',False)
    turn=pd.Series(np.nan,index=q.index,dtype=float); mt=m&np.isfinite(q.turn20)&q.turn20.gt(0); rt=pct(q,mt,'turn20',True); turn.loc[rt.index]=rt
    val=add_value_score(q,m)
    # optional hard gates are defined by PIT cross-sectional size rank before final re-ranking
    gate=m.copy()
    if name.startswith('small60'): gate=m&(size<=.60)
    elif name.startswith('small50'): gate=m&(size<=.50)
    elif name.startswith('small40'): gate=m&(size<=.40)
    elif name.startswith('small30'): gate=m&(size<=.30)
    if name=='base' or name.endswith('_base'):
        raw=.60*iv+.40*ef
    elif name=='size15': raw=.51*iv+.34*ef+.15*size
    elif name=='size25': raw=.45*iv+.30*ef+.25*size
    elif name=='size35': raw=.39*iv+.26*ef+.35*size
    elif name=='size50': raw=.30*iv+.20*ef+.50*size
    elif name=='value20': raw=.48*iv+.32*ef+.20*val
    elif name=='value30': raw=.42*iv+.28*ef+.30*val
    elif name=='size25_value15': raw=.36*iv+.24*ef+.25*size+.15*val
    elif name=='size35_value15': raw=.30*iv+.20*ef+.35*size+.15*val
    elif name=='size25_turn15': raw=.36*iv+.24*ef+.25*size+.15*turn
    elif name=='size35_turn15': raw=.30*iv+.20*ef+.35*size+.15*turn
    elif name in ('small50_value20','small40_value20'): raw=.48*iv+.32*ef+.20*val
    elif name in ('small50_turn20','small40_turn20'): raw=.48*iv+.32*ef+.20*turn
    else: raise ValueError(name)
    gate=gate&np.isfinite(raw)
    q.loc[gate,'rank_test']=raw.loc[gate].groupby(q.loc[gate,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q


def runfast(q,h,n,e,k,cal,members,bm): return grand.run(q,h,n,e,k,cal,members,bm,fast=True)
def runexact(q,h,n,e,k,cal,members,bm,cost=1.): return grand.run(q,h,n,e,k,cal,members,bm,cost=cost,fast=False)


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members)
    bs=load_bs(); manifests=glob.glob('baostock_shards/manifest_*.csv'); mf=pd.concat([pd.read_csv(f) for f in manifests],ignore_index=True) if manifests else pd.DataFrame()
    cols=['signal_date','code','float_mv_proxy','turn20','peTTM','pbMRQ','psTTM','pcfNcfTTM','isST','tradestatus','exact_obs']
    p=p.merge(bs[cols],on=['signal_date','code'],how='left',validate='many_to_one')
    coverage=float(np.isfinite(p.float_mv_proxy).mean()); exact_cov=float((p.exact_obs==1).mean())
    bm=market_close.loc[sim.START:sim.END].dropna()
    keep=['signal_date','trade_date','code','liq20','liq_rank_pct','exec_open','exec_high','exec_low','exec_volume','exec_factor','ivol60','eff120','skew40','float_mv_proxy','turn20','peTTM','pbMRQ','psTTM','pcfNcfTTM','isST','tradestatus','exact_obs']
    p=p[keep].copy()

    # A. structural PIT search fixed at central construction; train selection only.
    rows=[]; qs={}
    for v in VARIANTS:
        q=rank_variant(p,v); qs[v]=q; st,_,_,_=runfast(q,60,20,.10,.30,cal,members,bm); st['variant']=v; rows.append(st)
        print('PIT S1',v,'TRAIN',st.get('train_return'),'VAL',st.get('validation_return'),'FULL',st.get('total_return'),flush=True)
    s1=pd.DataFrame(rows); s1.to_csv(OUT/'stage1_pit.csv',index=False)
    elig=s1[(s1.train_return>0)&(s1.train_mdd>-0.45)&(s1.positions_max<=20)].copy(); elig['train_score']=elig.train_cagr-.18*elig.train_mdd.abs(); elig=elig.sort_values(['train_score','train_cagr'],ascending=False)
    top=list(elig.variant.head(6)); pd.DataFrame({'variant':top}).to_csv(OUT/'stage1_selected.csv',index=False)

    # B. construction grid for top train structures.
    rows2=[]
    for v in top:
        q=qs[v]
        for n,h,(e,k) in itertools.product(NS,HOLDS,BUFFERS):
            st,_,_,_=runfast(q,h,n,e,k,cal,members,bm); st['variant']=v; rows2.append(st)
    s2=pd.DataFrame(rows2); s2.to_csv(OUT/'stage2_pit.csv',index=False)
    ok=s2[(s2.train_return>0)&(s2.train_mdd>-0.45)&(s2.positions_max<=s2.n_hold)].copy(); ok['train_score']=ok.train_cagr-.18*ok.train_mdd.abs(); ok=ok.sort_values(['train_score','train_cagr'],ascending=False)

    # C. exact daily MTM top train-only configurations.
    exact=[]; cache={}; seen=set()
    for _,r in ok.head(TOP_EXACT).iterrows():
        cfg=(str(r.variant),int(r.n_hold),int(r.hold_days),float(r.entry_pct),float(r.keep_pct))
        if cfg in seen: continue
        seen.add(cfg); st,eq,tr,tm=runexact(qs[cfg[0]],cfg[2],cfg[1],cfg[3],cfg[4],cal,members,bm); st['variant']=cfg[0]; exact.append(st); cache[cfg]=(eq,tr,tm)
        print('PIT EXACT',cfg,'FULL',st.total_return if hasattr(st,'total_return') else st['total_return'],flush=True)
    ex=pd.DataFrame(exact); ex['train_score']=ex.train_cagr-.18*ex.train_mdd.abs(); ex=ex.sort_values(['train_score','train_cagr'],ascending=False); ex.to_csv(OUT/'exact_train_candidates.csv',index=False)
    w=ex.iloc[0]; cfg=(str(w.variant),int(w.n_hold),int(w.hold_days),float(w.entry_pct),float(w.keep_pct)); eq,tr,tm=cache[cfg]
    sim.annual_returns(eq).to_csv(OUT/'winner_annual.csv',index=False); pd.DataFrame([sim.robustness(eq,tr)]).to_csv(OUT/'winner_tail.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=runexact(qs[cfg[0]],cfg[2],cfg[1],cfg[3],cfg[4],cal,members,bm,cost=cm); st['cost_mult_test']=cm; costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'winner_costs.csv',index=False)
    # Transparency: full-sample fast top is exploratory only.
    s2.sort_values('total_return',ascending=False).head(25).to_csv(OUT/'exploratory_fullsample_top25_fast.csv',index=False)
    audit={**ua,'market_factor':market_code,'baostock_rows':len(bs),'baostock_codes':int(bs.code.nunique()),'panel_factor_coverage':coverage,'panel_exact_observation_coverage':exact_cov,'shard_requested_codes':int(mf.requested_codes.sum()) if len(mf) else np.nan,'shard_returned_codes':int(mf.returned_codes.sum()) if len(mf) else np.nan,'selection':'formal selection uses 2016-2021 only; 2022-2026 reused validation; PIT ST and tradestatus exact-date required','winner':str(cfg),'winner_total_return':float(w.total_return),'winner_cagr':float(w.cagr),'winner_mdd':float(w.max_drawdown),'winner_validation_return':float(w.validation_return),'five_x_target_met':int(float(w.total_return)>=4.0),'timing_violations':int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if audit['timing_violations'] or coverage<.70: raise RuntimeError('PIT audit fail')
    print('=== PIT AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
    print('=== PIT EXACT ==='); print(ex.to_string(index=False),flush=True)
    print('=== PIT ANNUAL ==='); print(sim.annual_returns(eq).to_string(index=False),flush=True)
    print('=== PIT COST ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)

if __name__=='__main__': main()
