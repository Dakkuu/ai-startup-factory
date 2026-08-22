from __future__ import annotations
from pathlib import Path
import re
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_balanced_exact as be
import run_10y_max_audit as ma

OUT=Path('results_backward_holdout'); OUT.mkdir(exist_ok=True)
RAW_START=pd.Timestamp('2007-01-04'); END=pd.Timestamp('2016-07-28'); WARM=pd.Timestamp('2005-01-01')
RULE='liq top70%; remove highest 20% skew40; score=.60 low-IVOL60 rank + .40 efficiency120 rank; N20; 60d; entry10 keep30; next-open; market residual benchmark SH000300'
STOCK_RE=re.compile(r'^(?:SH(?:600|601|603|605|688)\d{3}|SZ(?:000|001|002|003|300|301)\d{3}|BJ\d{6})$')


def load_base_holdout():
    base.qb.RELEASE_TAG=base.RELEASE_TAG; base.qb.ROOT=Path('qlib_data'); base.qb.download_and_extract()
    cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(base.qb.ROOT/'calendars'/'day.txt',header=None)[0]))
    m=pd.read_csv(base.qb.ROOT/'instruments'/'all.txt',sep='\t',header=None,names=['code','start','end'],usecols=[0,1,2])
    m['code']=m.code.astype(str).str.upper(); m['start']=pd.to_datetime(m.start); m['end']=pd.to_datetime(m.end)
    m=m[m.code.str.match(STOCK_RE)].copy(); m=m[(m.end>=WARM)&(m.start<=END)]
    t=cal[(cal>=RAW_START)&(cal<=END)]; cnt=np.array([int(((m.start<=d)&(m.end>=d)).sum()) for d in t])
    union=int(m[(m.end>=RAW_START)&(m.start<=END)].code.nunique()); entered=int(m[(m.start>RAW_START)&(m.start<=END)].code.nunique()); exited=int(m[(m.end>=RAW_START)&(m.end<END)].code.nunique())
    audit={'release_tag':base.RELEASE_TAG,'raw_start':str(RAW_START.date()),'end':str(END.date()),'union_members':union,'entered':entered,'exited':exited,'min_daily_members':int(cnt.min()),'max_daily_members':int(cnt.max()),'daily_ratio':float(cnt.max()/cnt.min()),'calendar_days':len(t)}
    structural={'calendar_gt_2000':len(t)>2000,'min_daily_gt_1000':cnt.min()>1000,'union_ge_max_daily':union>=cnt.max(),'entered_gt_500':entered>500,'exited_gt_10':exited>10,'daily_ratio_1_to_2_5':1<=cnt.max()/cnt.min()<=2.5}
    pd.DataFrame([audit]).to_csv(OUT/'universe_audit.csv',index=False); pd.DataFrame([{'gate':k,'pass':int(v)} for k,v in structural.items()]).to_csv(OUT/'universe_gates.csv',index=False)
    if not all(structural.values()): raise RuntimeError(f'FAIL-CLOSED holdout universe {audit} {structural}')
    return cal,m,audit


def active_mask(mm,dates):
    out=np.zeros(len(dates),dtype=bool); valid=~pd.isna(dates)
    for r in mm.itertuples(index=False): out |= valid & (dates>=r.start)&(dates<=r.end)
    return out


def frozen_market(cal):
    s=base.qb.read_bin('SH000300','close',cal).loc[WARM:END].dropna()
    need=cal[(cal>=WARM)&(cal<=END)]; coverage=float(s.reindex(need).notna().mean())
    pd.DataFrame([{'code':'SH000300','coverage':coverage,'first':s.index.min(),'last':s.index.max()}]).to_csv(OUT/'market_coverage.csv',index=False)
    if coverage<.98: raise RuntimeError(f'FAIL-CLOSED frozen market coverage {coverage}')
    return 'SH000300',s


def find_data_ready_start(counts):
    # The backward holdout start is chosen ONLY from data availability, never returns.
    # Earliest date that leaves >=5 years and whose remaining panel has adequate breadth:
    # >=95% signal dates have >=300 eligible names, 10th percentile >=250, median eligible share >=25%.
    dates=pd.DatetimeIndex(counts.index)
    for d in dates:
        yrs=(END-d).days/365.25
        if yrs<5.0: break
        tail=counts.loc[d:]
        if len(tail)<200: continue
        if (tail.eligible>=300).mean()>=.95 and tail.eligible.quantile(.10)>=250 and tail.eligible_share.median()>=.25:
            return pd.Timestamp(d)
    raise RuntimeError('no data-ready backward holdout window satisfying predeclared coverage rules')


def build_minimal_panel(cal,members,market_close,ua):
    trade_cal=cal[(cal>=RAW_START)&(cal<=END)]; signal_dates=pd.DatetimeIndex(trade_cal[::5]); alltrade=cal[cal<=END]
    exec_dates=[]
    for s in signal_dates:
        k=alltrade.searchsorted(s,side='right'); exec_dates.append(alltrade[k] if k<len(alltrade) else pd.NaT)
    exec_dates=pd.DatetimeIndex(exec_dates)
    bm_ret=market_close.reindex(cal[(cal>=WARM)&(cal<=END)]).pct_change(fill_method=None)
    bm_mu=bm_ret.rolling(252,min_periods=126).mean().shift(1); bm_var=bm_ret.rolling(252,min_periods=126).var().shift(1)
    frames=[]; codes=sorted(members.code.unique())
    for i,code in enumerate(codes,1):
        mm=members[members.code==code]; cols={}
        for f in ['open','high','low','close','volume','factor']:
            s=base.qb.read_bin(code,f,cal)
            if not s.empty: cols[f]=s
        if not all(f in cols for f in ['open','high','low','close','volume']): continue
        z=pd.concat(cols,axis=1).loc[WARM:END].copy()
        if z.empty: continue
        if 'factor' not in z: z['factor']=1.
        z['factor']=z.factor.replace(0,np.nan).fillna(1.)
        r=z.close.pct_change(fill_method=None); count120=z.close.notna().rolling(120).sum(); liq20=(z.close.abs()*z.volume.abs()).rolling(20).mean()
        m=bm_ret.reindex(z.index); smu=r.rolling(252,min_periods=126).mean().shift(1); cov=r.rolling(252,min_periods=126).cov(m).shift(1); beta=cov/bm_var.reindex(z.index); alpha=smu-beta*bm_mu.reindex(z.index); resid=r-alpha-beta*m
        ivol60=resid.rolling(60,min_periods=48).std()
        sig=pd.DataFrame({'count120':count120.reindex(signal_dates).to_numpy(),'liq20':liq20.reindex(signal_dates).to_numpy(),'ivol60':ivol60.reindex(signal_dates).to_numpy()})
        ex=z.reindex(exec_dates).reset_index(drop=True)
        valid=active_mask(mm,signal_dates)&active_mask(mm,exec_dates)&(~pd.isna(exec_dates)); valid &= np.asarray(sig.count120>=120)
        valid &= np.isfinite(sig[['liq20','ivol60']].to_numpy()).all(axis=1); valid &= np.isfinite(ex[['open','high','low','volume']].to_numpy()).all(axis=1)
        if not valid.any(): continue
        idx=np.flatnonzero(valid)
        frames.append(pd.DataFrame({'signal_date':signal_dates[idx],'trade_date':exec_dates[idx],'code':code,'liq20':sig.liq20.to_numpy()[idx].astype(float),'ivol60':sig.ivol60.to_numpy()[idx].astype(float),'exec_open':ex.open.to_numpy()[idx].astype(float),'exec_high':ex.high.to_numpy()[idx].astype(float),'exec_low':ex.low.to_numpy()[idx].astype(float),'exec_volume':ex.volume.to_numpy()[idx].astype(float),'exec_factor':ex.factor.to_numpy()[idx].astype(float)}))
        if i%500==0: print('minimal histories',i,'/',len(codes),flush=True)
    if not frames: raise RuntimeError('no holdout panel')
    p=pd.concat(frames,ignore_index=True); p['liq_rank_pct']=p.groupby('signal_date').liq20.rank(pct=True,method='average',ascending=False)
    eligible=p.groupby('signal_date').size().reindex(signal_dates,fill_value=0).astype(int)
    active=pd.Series([int(((members.start<=d)&(members.end>=d)).sum()) for d in signal_dates],index=signal_dates,dtype=float)
    counts=pd.DataFrame({'eligible':eligible,'active':active}); counts['eligible_share']=counts.eligible/counts.active.replace(0,np.nan)
    counts.to_csv(OUT/'eligible_by_signal_date.csv',index_label='signal_date')
    ready=find_data_ready_start(counts)
    tail=counts.loc[ready:]
    audit={'raw_signal_dates':len(signal_dates),'raw_rows':len(p),'raw_min_eligible':int(eligible.min()),'raw_median_eligible':float(eligible.median()),'raw_max_eligible':int(eligible.max()),'data_ready_start':str(ready.date()),'post_ready_signal_dates':len(tail),'post_ready_min_eligible':int(tail.eligible.min()),'post_ready_p10_eligible':float(tail.eligible.quantile(.10)),'post_ready_median_eligible':float(tail.eligible.median()),'post_ready_median_share':float(tail.eligible_share.median()),'post_ready_share_ge300':float((tail.eligible>=300).mean()),'holdout_years':float((END-ready).days/365.25)}
    pd.DataFrame([audit]).to_csv(OUT/'panel_audit.csv',index=False)
    gates={'data_ready_found':True,'holdout_ge_5y':audit['holdout_years']>=5,'post_ready_95pct_dates_ge300':audit['post_ready_share_ge300']>=.95,'post_ready_p10_ge250':audit['post_ready_p10_eligible']>=250,'post_ready_median_share_ge25pct':audit['post_ready_median_share']>=.25}
    pd.DataFrame([{'gate':k,'pass':int(v)} for k,v in gates.items()]).to_csv(OUT/'panel_gates.csv',index=False)
    if not all(gates.values()): raise RuntimeError(f'FAIL-CLOSED data-ready panel {audit} {gates}')
    return p,ready


def period_metrics(eq,a,b):
    z=eq[(pd.to_datetime(eq.trade_date)>=pd.Timestamp(a))&(pd.to_datetime(eq.trade_date)<=pd.Timestamp(b))].copy()
    if len(z)<5:return {'return':np.nan,'cagr':np.nan,'mdd':np.nan,'sharpe':np.nan}
    s=z.set_index(pd.to_datetime(z.trade_date)).equity.astype(float); total=float(s.iloc[-1]/s.iloc[0]-1); yrs=max((s.index[-1]-s.index[0]).days/365.25,1e-9); rr=s.pct_change().dropna(); dd=s/s.cummax()-1
    return {'return':total,'cagr':float((s.iloc[-1]/s.iloc[0])**(1/yrs)-1),'mdd':float(dd.min()),'sharpe':float(rr.mean()/rr.std()*np.sqrt(252)) if rr.std()>0 else np.nan}


def thirds(start,end):
    a=int(start.value); b=int(end.value); e=[pd.Timestamp(int(x)) for x in np.linspace(a,b,4)]
    return [('block1',e[0],e[1]),('block2',e[1],e[2]),('block3',e[2],e[3])]


def main():
    # Build all factors before seeing any backward-holdout returns.
    base.START=RAW_START; base.END=END; base.WARM=WARM; base.OUT=OUT; v4.OUT=OUT
    sim.START=RAW_START; sim.END=END; sim.WARM=WARM
    cal,members,ua=load_base_holdout(); market_code,market_close=frozen_market(cal); p,ready=build_minimal_panel(cal,members,market_close,ua)
    p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close)
    # From here onward the start is the availability-only frozen start.
    sim.START=ready; base.START=ready
    bm=market_close.loc[ready:END].dropna(); q=be.anchor_weighted(p,'liq70',.60)
    st,eq,tr,tm=ma.run_q(q,60,0,cal,members,bm,n=20,entry=.10,keep=.30); pd.DataFrame([st]).to_csv(OUT/'summary.csv',index=False); sim.annual_returns(eq).to_csv(OUT/'annual.csv',index=False)
    phases=[]
    for ph0 in range(12):
        x,_,_,_=ma.run_q(q,60,ph0,cal,members,bm,n=20,entry=.10,keep=.30); phases.append({**x,'phase':ph0})
    ph=pd.DataFrame(phases); ph.to_csv(OUT/'phase_offsets.csv',index=False)
    blocks=[]
    for name,a,b in thirds(ready,END):
        z=period_metrics(eq,a,b); z.update({'block':name,'start':a,'end':b}); blocks.append(z)
    bl=pd.DataFrame(blocks); bl.to_csv(OUT/'blocks.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        z=ma.subset_phase(q,60,0); x,_,_,_=ma.run_panel(z,cal,members,bm,n=20,entry=.10,keep=.30,cost=cm,start=ready,end=END); costs.append({**x,'cost_mult_test':cm})
    co=pd.DataFrame(costs); co.to_csv(OUT/'costs.csv',index=False)
    delays=[]
    for d in (1,3,5):
        z=ma.delay_panel(q,d,cal,members); x,_,_,_=ma.run_panel(z,cal,members,bm,n=20,entry=.10,keep=.30,start=ready,end=END); delays.append({**x,'delay_sessions':d})
    de=pd.DataFrame(delays); de.to_csv(OUT/'delays.csv',index=False)
    bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    gates={'timing_zero':int(bad==0),'total_return_positive':int(st['total_return']>0),'cagr_ge_5pct':int(st['cagr']>=.05),'mdd_better_than_minus40pct':int(st['max_drawdown']>-0.40),'sharpe_ge_0_4':int(st['sharpe']>=.40),'all_12_phases_positive':int((ph.total_return>0).all()),'phase_median_cagr_ge_5pct':int(ph.cagr.median()>=.05),'all_3_equal_time_blocks_positive':int((bl['return']>0).all()),'cost4_positive':int(float(co.loc[co.cost_mult_test==4,'total_return'].iloc[0])>0),'delay3_positive':int(float(de.loc[de.delay_sessions==3,'total_return'].iloc[0])>0)}
    gd=pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]); gd.to_csv(OUT/'gates.csv',index=False)
    verdict={**ua,'market_factor':market_code,'holdout_start':str(ready.date()),'holdout_end':str(END.date()),'start_selection':'data availability only; no return metric used','frozen_rule':RULE,'total_return':st['total_return'],'cagr':st['cagr'],'mdd':st['max_drawdown'],'sharpe':st['sharpe'],'gates_passed':int(gd['pass'].sum()),'gates_total':len(gd),'backward_holdout_hard_pass':int(gd['pass'].all())}
    pd.DataFrame([verdict]).to_csv(OUT/'verdict.csv',index=False)
    print('=== BACKWARD HOLDOUT VERDICT ==='); print(pd.DataFrame([verdict]).to_string(index=False),flush=True)
    print('=== GATES ==='); print(gd.to_string(index=False),flush=True)
    print('=== PHASES ==='); print(ph[['phase','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True)
    print('=== BLOCKS ==='); print(bl.to_string(index=False),flush=True)
    print('=== COSTS ==='); print(co[['cost_mult_test','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True)
    print('=== DELAYS ==='); print(de[['delay_sessions','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True)

if __name__=='__main__': main()
