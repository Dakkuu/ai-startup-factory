from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd

# Correctness patches must be installed before v4 panel construction / hard execution imports.
import run_10y_hard_executor_v2 as hv2
import run_10y_signal_pure_panel as sp
hv2.patch(); sp.patch()

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_factor_mine2 as mine2
import run_10y_max_audit as ma

OUT=Path('results_alpha_discovery_qv'); OUT.mkdir(exist_ok=True)
TRAIN_END_20=pd.Timestamp('2021-11-15')
TRAIN_END_60=pd.Timestamp('2021-09-30')
LIQ_KEEP=.70
N_HOLD=20
ENTRY=.10
KEEP=.30

# Pre-registered directions: every factor is oriented so larger value means higher expected future return.
MANIFEST=[
 ('low_ivol',60,'behavioral leverage/lottery demand: lower idiosyncratic volatility should be rewarded'),
 ('efficiency',60,'persistent trends should have more return per unit of path length'),
 ('residual_momentum',60,'stock-specific momentum after removing market beta should persist'),
 ('anti_max',60,'avoid lottery-demand stocks with an extreme positive residual day'),
 ('anti_skew',60,'avoid positively skewed lottery-demand stocks'),
 ('low_downside',60,'lower downside semivariance should reduce distress/lottery exposure'),
 ('low_beta',60,'low-beta anomaly within a liquid long-only universe'),
 ('capture_asymmetry',60,'prefer stocks that capture upside market moves better than downside moves'),
 ('short_reversal_5',20,'very recent losers may mean-revert after liquidity/attention shocks'),
 ('short_reversal_20',20,'one-month losers may mean-revert in A-share retail-driven trading'),
 ('skip_momentum_63_5',60,'medium momentum excluding the most recent week should persist'),
 ('skip_momentum_126_20',60,'medium momentum excluding the most recent month should persist'),
 ('near_52w_high',60,'price proximity to its 252-day high proxies slow information diffusion'),
 ('vol_compression',60,'low short/long realized-vol ratio may precede persistent repricing'),
 ('range_compression',20,'contracting intraday range may precede directional expansion'),
 ('relative_volume',20,'abnormally high current trading activity may confirm new information'),
 ('quiet_trend',60,'medium momentum carried on relatively quiet volume may be less crowded'),
 ('oversold_volume',20,'oversold stocks with elevated volume may exhibit liquidity-shock reversal'),
 ('intraday_reversal',20,'a large same-day decline from open to close may mean-revert next month'),
 ('gap_reversal',20,'negative overnight gaps may partially reverse after forced/attention selling'),
 ('amihud_illiquidity',60,'within the top-70% liquid universe, moderate price-impact exposure may earn a premium'),
]


def add_qv_fields(panel,cal):
    p=panel.copy()
    cols=['short_reversal_5','short_reversal_20','skip_momentum_63_5','skip_momentum_126_20',
          'near_52w_high','vol_compression','range_compression','relative_volume','quiet_trend',
          'oversold_volume','intraday_reversal','gap_reversal','amihud_illiquidity','fwd20','fwd60']
    for c in cols: p[c]=np.nan
    groups=p.groupby('code').groups
    for i,(code,idx) in enumerate(groups.items(),1):
        c=base.qb.read_bin(code,'close',cal); o=base.qb.read_bin(code,'open',cal); h=base.qb.read_bin(code,'high',cal); l=base.qb.read_bin(code,'low',cal); v=base.qb.read_bin(code,'volume',cal); f=base.qb.read_bin(code,'factor',cal)
        if c.empty or o.empty or h.empty or l.empty or v.empty: continue
        z=pd.concat([c.rename('c'),o.rename('o'),h.rename('h'),l.rename('l'),v.rename('v')],axis=1).loc[sim.WARM:sim.END]
        if z.empty: continue
        ff=f.reindex(z.index).replace(0,np.nan) if not f.empty else pd.Series(1.0,index=z.index)
        r=z.c.pct_change(fill_method=None)
        rawv=z.v.abs()*ff.abs()*100.0
        rawp=z.c/ff
        amount=(rawp.abs()*rawv.abs()).replace(0,np.nan)
        rv20=r.rolling(20,min_periods=16).std(); rv120=r.rolling(120,min_periods=100).std()
        daily_range=(z.h-z.l).abs()/z.c.shift(1).abs().replace(0,np.nan)
        rg20=daily_range.rolling(20,min_periods=16).mean(); rg120=daily_range.rolling(120,min_periods=100).mean()
        relv=rawv.rolling(20,min_periods=16).mean()/rawv.rolling(60,min_periods=48).mean().replace(0,np.nan)
        rev5=-(z.c/z.c.shift(5)-1); rev20=-(z.c/z.c.shift(20)-1)
        mom635=z.c.shift(5)/z.c.shift(63)-1; mom12620=z.c.shift(20)/z.c.shift(126)-1
        high52=z.c/z.c.rolling(252,min_periods=200).max()
        volcomp=-(rv20/rv120.replace(0,np.nan)); rangecomp=-(rg20/rg120.replace(0,np.nan))
        intraday=-(z.c/z.o.replace(0,np.nan)-1); gap=-(z.o/z.c.shift(1).replace(0,np.nan)-1)
        amihud=(r.abs()/amount).rolling(20,min_periods=16).mean()
        quiet=mom635/(1.0+relv.clip(lower=0))
        oversold=rev20*np.log1p(relv.clip(lower=0))
        fac={
          'short_reversal_5':rev5,'short_reversal_20':rev20,'skip_momentum_63_5':mom635,
          'skip_momentum_126_20':mom12620,'near_52w_high':high52,'vol_compression':volcomp,
          'range_compression':rangecomp,'relative_volume':relv,'quiet_trend':quiet,
          'oversold_volume':oversold,'intraday_reversal':intraday,'gap_reversal':gap,
          'amihud_illiquidity':amihud,'fwd20':z.c.shift(-20)/z.c-1,'fwd60':z.c.shift(-60)/z.c-1,
        }
        ds=pd.DatetimeIndex(p.loc[idx,'signal_date'])
        for name,s in fac.items(): p.loc[idx,name]=s.reindex(ds).to_numpy(float)
        if i%1000==0: print('qv histories',i,'/',len(groups),flush=True)
    return p


def attach_oriented_existing(p):
    q=p.copy()
    q['low_ivol']=-q.ivol60
    q['efficiency']=q.eff120
    q['residual_momentum']=q.rmom126
    q['anti_max']=-q.max20
    q['anti_skew']=-q.skew60
    q['low_downside']=-q.dsemi60
    q['low_beta']=-q.beta252
    q['capture_asymmetry']=q.capture120
    return q


def factor_diag(p,name,horizon):
    label=f'fwd{horizon}'; cutoff=TRAIN_END_20 if horizon==20 else TRAIN_END_60
    z=p[(pd.to_datetime(p.signal_date)<=cutoff)&(p.liq_rank_pct<=LIQ_KEEP)][['signal_date','code',name,label]].dropna().copy()
    z=z[np.isfinite(z[name])&np.isfinite(z[label])]
    ics=[]; dec_rows=[]
    for d,g in z.groupby('signal_date',sort=True):
        if len(g)<300: continue
        ic=g[name].corr(g[label],method='spearman')
        if np.isfinite(ic): ics.append((pd.Timestamp(d),float(ic)))
        rr=g[name].rank(pct=True,method='average',ascending=True)
        dec=np.ceil(rr*10).clip(1,10).astype(int)
        x=pd.DataFrame({'decile':dec.to_numpy(),'ret':g[label].to_numpy()}).groupby('decile').ret.mean()
        for k,v in x.items(): dec_rows.append({'signal_date':pd.Timestamp(d),'decile':int(k),'ret':float(v)})
    icdf=pd.DataFrame(ics,columns=['signal_date','ic'])
    if icdf.empty:
        return {'factor':name,'horizon':horizon,'n_dates':0,'mean_ic':np.nan,'ic_t':np.nan,'ic_positive_share':np.nan,'monotonic_corr':np.nan,'top_bottom_spread':np.nan,'positive_years':0,'years':0,'screen_pass':0},pd.DataFrame(),pd.DataFrame()
    sd=float(icdf.ic.std(ddof=1)); t=float(icdf.ic.mean()/sd*math.sqrt(len(icdf))) if sd>0 else np.nan
    decdf=pd.DataFrame(dec_rows); decmean=decdf.groupby('decile').ret.mean().reindex(range(1,11)); mono=float(pd.Series(range(1,11),index=range(1,11)).corr(decmean,method='spearman')) if decmean.notna().sum()>=8 else np.nan
    spread=float(decmean.loc[10]-decmean.loc[1]) if 1 in decmean.index and 10 in decmean.index and np.isfinite(decmean.loc[1]) and np.isfinite(decmean.loc[10]) else np.nan
    annual=icdf.assign(year=icdf.signal_date.dt.year).groupby('year').ic.mean(); py=int((annual>0).sum()); years=int(len(annual))
    passed=int(len(icdf)>=200 and float(icdf.ic.mean())>0 and np.isfinite(t) and t>=2.0 and np.isfinite(mono) and mono>=.50 and np.isfinite(spread) and spread>0 and py>=max(3,math.ceil(.6*years)))
    row={'factor':name,'horizon':horizon,'n_dates':len(icdf),'mean_ic':float(icdf.ic.mean()),'ic_t':t,'ic_positive_share':float((icdf.ic>0).mean()),'monotonic_corr':mono,'top_bottom_spread':spread,'positive_years':py,'years':years,'screen_pass':passed}
    adec=decmean.rename('mean_forward_return').reset_index(); adec['factor']=name; adec['horizon']=horizon
    aic=annual.rename('mean_ic').reset_index(); aic['factor']=name; aic['horizon']=horizon
    return row,adec,aic


def make_ranked(p,name):
    q=p.copy(); q['rank_test']=np.nan
    m=(q.liq_rank_pct<=LIQ_KEEP)&np.isfinite(q[name])
    q.loc[m,'rank_test']=q.loc[m].groupby('signal_date')[name].rank(pct=True,method='average',ascending=False)
    return q


def period_fields(st,eq):
    st=st.copy(); st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31'); st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
    return st


def exact_run(q,horizon,cal,members,bm,cost=1.0):
    st,eq,tr,tm=ma.run_q(q,horizon,0,cal,members,bm,n=N_HOLD,entry=ENTRY,keep=KEEP,cost=cost)
    return period_fields(st,eq),eq,tr,tm


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT; sp.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close)
    p=fq.add_factors(p,cal)
    p=mine2.add_extra(p,cal,market_close)
    p=add_qv_fields(p,cal)
    p=attach_oriented_existing(p)
    bm=market_close.loc[sim.START:sim.END].dropna()

    pd.DataFrame([{'factor':n,'horizon':h,'mechanism':m,'direction':'larger_is_better','screen':'2016-2021 IC+decile only; no 2022-2026 selection'} for n,h,m in MANIFEST]).to_csv(OUT/'factor_manifest.csv',index=False)
    rows=[]; deciles=[]; annual_ic=[]
    for name,horizon,_ in MANIFEST:
        print('SCREEN',name,'H',horizon,flush=True)
        r,d,aic=factor_diag(p,name,horizon); rows.append(r)
        if len(d): deciles.append(d)
        if len(aic): annual_ic.append(aic)
    screen=pd.DataFrame(rows).sort_values(['screen_pass','ic_t','mean_ic'],ascending=[False,False,False]); screen.to_csv(OUT/'factor_screen.csv',index=False)
    if deciles: pd.concat(deciles,ignore_index=True).to_csv(OUT/'factor_deciles.csv',index=False)
    if annual_ic: pd.concat(annual_ic,ignore_index=True).to_csv(OUT/'factor_annual_ic.csv',index=False)

    passed=screen[screen.screen_pass==1].copy(); diagnostic_fallback=0
    if len(passed): finalists=passed.head(5).copy()
    else:
        finalists=screen[np.isfinite(screen.ic_t)&(screen.mean_ic>0)].head(3).copy(); diagnostic_fallback=1
    exact=[]; cache={}
    for r in finalists.itertuples(index=False):
        name=str(r.factor); horizon=int(r.horizon); print('EXACT',name,'H',horizon,'screen_pass',int(r.screen_pass),flush=True)
        q=make_ranked(p,name); st,eq,tr,tm=exact_run(q,horizon,cal,members,bm,1.0); st.update({'factor':name,'horizon':horizon,'screen_pass':int(r.screen_pass),'ic_t_train':float(r.ic_t),'mean_ic_train':float(r.mean_ic)}); exact.append(st); cache[(name,horizon)]=(q,eq,tr,tm)
    ex=pd.DataFrame(exact)
    if len(ex): ex=ex.sort_values(['screen_pass','train_2016_2021_return','max_drawdown'],ascending=[False,False,False])
    ex.to_csv(OUT/'exact_finalists.csv',index=False)

    costs=[]; annual=[]; robust=[]; winner=None
    if len(ex):
        eligible=ex[ex.screen_pass==1]
        if len(eligible):
            w=eligible.sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False]).iloc[0]; winner=(str(w.factor),int(w.horizon)); q,eq,tr,tm=cache[winner]
            for cm in (2.,4.,8.):
                st,_,_,_=exact_run(q,winner[1],cal,members,bm,cm); st.update({'factor':winner[0],'horizon':winner[1]}); costs.append(st)
            a=sim.annual_returns(eq); a['factor']=winner[0]; a['horizon']=winner[1]; annual.append(a)
            rr=sim.robustness(eq,tr); rr.update({'factor':winner[0],'horizon':winner[1]}); robust.append(rr)
    if costs: pd.DataFrame(costs).to_csv(OUT/'winner_costs.csv',index=False)
    if annual: pd.concat(annual,ignore_index=True).to_csv(OUT/'winner_annual.csv',index=False)
    if robust: pd.DataFrame(robust).to_csv(OUT/'winner_robust.csv',index=False)

    alltm=pd.concat([x[3].assign(factor=k[0],horizon=k[1]) for k,x in cache.items() if len(x[3])],ignore_index=True) if cache else pd.DataFrame()
    bad=int((pd.to_datetime(alltm.signal_date)>=pd.to_datetime(alltm.trade_date)).sum()) if len(alltm) else 0
    target_hits=int((ex.total_return>=5.0).sum()) if len(ex) else 0
    audit={**ua,'market_factor':market_code,'research_round':'Qlib price-volume Alpha Discovery A','factor_count':len(MANIFEST),'screen_pass_count':int(screen.screen_pass.sum()),'exact_finalists':len(ex),'diagnostic_fallback':diagnostic_fallback,'signal_universe':'T information only; T+1 quote/tradability cannot replace a T-ranked stock','volume_source_unit_shares':100,'portfolio':'N20; entry top10%; keep top30%; next-open; 100-share board lot; corrected volume participation','selection':'factor screen and finalist choice use 2016-2021 only; 2022-2026 is pseudo-OOS/exploratory','timing_violations':bad,'target_6x_hits':target_hits,'winner':winner[0] if winner else 'NONE','winner_horizon':winner[1] if winner else np.nan}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad: raise RuntimeError(f'timing violations {bad}')
    print('=== FACTOR SCREEN ==='); print(screen.to_string(index=False),flush=True)
    print('=== EXACT FINALISTS ==='); print(ex.to_string(index=False),flush=True)
    if costs: print('=== WINNER COSTS ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)

if __name__=='__main__': main()
