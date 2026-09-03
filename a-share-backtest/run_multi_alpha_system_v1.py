from __future__ import annotations
from pathlib import Path
import glob, json, math
import numpy as np
import pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_era_backtest as base
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_geff_fundamental_integrated_v3 as iv3
import run_geff_fundamental_ranktilt_v1 as rt
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_multi_alpha_system_v1'); OUT.mkdir(exist_ok=True)
START=pd.Timestamp('2016-08-02'); TRAIN_END=pd.Timestamp('2021-12-31')
PSEUDO=pd.Timestamp('2022-01-01'); END=pd.Timestamp('2026-07-29')
SHORT_FAMILIES=('rev5','mom20','breakout','accel')
SHORT_H=(5,10,20); SHORT_N=(10,15); SHORT_BUF=((.05,.20),(.10,.30))
LONG_FAMILIES=('value','quality','value_quality')
LONG_H=(120,180,250); LONG_N=(15,20,30); LONG_BUF=(.10,.30)
ALLOCATIONS={
 'A1_25_50_25':(.25,.50,.25),
 'A2_30_40_30':(.30,.40,.30),
 'A3_20_50_30':(.20,.50,.30),
 'A4_20_40_40':(.20,.40,.40),
 'A5_equal':(1/3,1/3,1/3),
}


def locate(pat):
    h=glob.glob(pat,recursive=True)
    if not h: raise FileNotFoundError(pat)
    return h[0]


def load_base_panel():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False)
    p=strict.attach_gap_flags(p,cal,'board')
    va=pd.read_csv(locate('artifact_cache/value/**/pit_value_attached.csv.gz'),compression='gzip',low_memory=False)
    sa=pd.read_csv(locate('artifact_cache/stmt/**/pit_attached.csv.gz'),compression='gzip',low_memory=False)
    iv3.verify_attach(p,va,'value'); iv3.verify_attach(p,sa,'3stmt')
    p2,z=iv3.fund_ranks(p,va,sa)
    return p2,z,cal,members,ua,market_code,bm


def add_short_features(p,cal):
    x=p.copy()
    for c in ['ret5','ret20','ret60','near_high60','relvol20_120','accel20_60']:
        x[c]=np.nan
    groups=x.groupby('code').groups
    warm=pd.Timestamp('2015-01-01')
    for i,(code,idxs) in enumerate(groups.items(),1):
        idxs=np.asarray(list(idxs))
        c=base.qb.read_bin(code,'close',cal).loc[warm:END]
        v=base.qb.read_bin(code,'volume',cal).loc[warm:END]
        if c.empty or v.empty: continue
        r5=c/c.shift(5)-1
        r20=c/c.shift(20)-1
        r60=c/c.shift(60)-1
        prior_high=c.shift(1).rolling(60,min_periods=48).max()
        near=c/prior_high
        v20=v.rolling(20,min_periods=16).mean()
        v120=v.rolling(120,min_periods=90).mean()
        rel=v20/v120.replace(0,np.nan)
        acc=r20-r60/3.0
        ds=pd.DatetimeIndex(x.loc[idxs,'signal_date'])
        x.loc[idxs,'ret5']=r5.reindex(ds).to_numpy(float)
        x.loc[idxs,'ret20']=r20.reindex(ds).to_numpy(float)
        x.loc[idxs,'ret60']=r60.reindex(ds).to_numpy(float)
        x.loc[idxs,'near_high60']=near.reindex(ds).to_numpy(float)
        x.loc[idxs,'relvol20_120']=rel.reindex(ds).to_numpy(float)
        x.loc[idxs,'accel20_60']=acc.reindex(ds).to_numpy(float)
        if i%1000==0: print('SHORT FEATURES',i,'/',len(groups),flush=True)
    x['liq_pct']=x.groupby('signal_date').liq20.rank(pct=True,ascending=False,method='average')
    return x


def add_long_ranks(p,z):
    x=p.reset_index(drop=True).copy(); x['_row']=np.arange(len(x))
    f=z.copy()
    value_cols=['earnings_yield_z','book_yield_z','cashflow_yield_z']
    qual_cols=['roe_z','gross_margin_z','cfo_assets_z','accrual_quality_z','cash_conversion_z']
    f['value_count']=f[value_cols].notna().sum(axis=1)
    f['quality_count']=f[qual_cols].notna().sum(axis=1)
    f['value_pure_raw']=f[value_cols].mean(axis=1,skipna=True).where(f.value_count>=2)
    f['quality_pure_raw']=f[qual_cols].mean(axis=1,skipna=True).where(f.quality_count>=3)
    f['value_pure_rank']=f.groupby('signal_date').value_pure_raw.rank(pct=True,method='average',ascending=False)
    f['quality_pure_rank']=f.groupby('signal_date').quality_pure_raw.rank(pct=True,method='average',ascending=False)
    x=x.merge(f[['_row','value_pure_rank','quality_pure_rank']],on='_row',how='left',validate='one_to_one').drop(columns='_row')
    return x


def rankpct(q,mask,col,ascending):
    return q.loc[mask].groupby('signal_date')[col].rank(pct=True,method='average',ascending=ascending)


def make_short_q(p,fam):
    q=p.copy(); q['rank_test']=np.nan
    liq=q.liq_pct<=.70
    if fam=='rev5':
        m=liq & np.isfinite(q.ret5)
        raw=rankpct(q,m,'ret5',True)
    elif fam=='mom20':
        m=liq & np.isfinite(q.ret20)
        raw=rankpct(q,m,'ret20',False)
    elif fam=='breakout':
        m=liq & np.isfinite(q.ret20) & np.isfinite(q.near_high60) & np.isfinite(q.relvol20_120)
        a=rankpct(q,m,'ret20',False); b=rankpct(q,m,'near_high60',False); c=rankpct(q,m,'relvol20_120',False)
        raw=.50*a+.30*b+.20*c
    elif fam=='accel':
        m=liq & np.isfinite(q.accel20_60) & np.isfinite(q.relvol20_120)
        a=rankpct(q,m,'accel20_60',False); b=rankpct(q,m,'relvol20_120',False)
        raw=.60*a+.40*b
    else: raise ValueError(fam)
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q


def make_long_q(p,fam):
    q=p.copy(); q['rank_test']=np.nan
    q['liq_pct2']=q.groupby('signal_date').liq20.rank(pct=True,ascending=False,method='average')
    liq=q.liq_pct2<=.70
    if fam=='value':
        m=liq & np.isfinite(q.value_pure_rank); raw=q.loc[m,'value_pure_rank']
    elif fam=='quality':
        m=liq & np.isfinite(q.quality_pure_rank); raw=q.loc[m,'quality_pure_rank']
    elif fam=='value_quality':
        m=liq & np.isfinite(q.value_pure_rank) & np.isfinite(q.quality_pure_rank)
        raw=.5*q.loc[m,'value_pure_rank']+.5*q.loc[m,'quality_pure_rank']
    else: raise ValueError(fam)
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q


def subset(q,h,ph):
    ds=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
    step=max(1,round(h/5)); chosen=set(ds[ph::step])
    cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')


def series_from_eq(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index()
    return s[~s.index.duplicated(keep='last')]


def perf_series(s,a=None,b=None):
    s=s.sort_index()
    if a is not None: s=s[s.index>=pd.Timestamp(a)]
    if b is not None: s=s[s.index<=pd.Timestamp(b)]
    if len(s)<20: return dict(cagr=np.nan,max_drawdown=np.nan,sharpe=np.nan,total_return=np.nan,calmar=np.nan)
    s=s/s.iloc[0]
    r=s.pct_change().dropna(); days=max(1,(s.index[-1]-s.index[0]).days)
    c=float(s.iloc[-1]**(365.25/days)-1); dd=float((s/s.cummax()-1).min())
    sh=float(r.mean()/r.std(ddof=1)*np.sqrt(252)) if len(r)>2 and r.std(ddof=1)>0 else np.nan
    return dict(cagr=c,max_drawdown=dd,sharpe=sh,total_return=float(s.iloc[-1]-1),calmar=float(c/abs(dd)) if dd<0 else np.nan)


def eq_from_series(s):
    return pd.DataFrame({'trade_date':s.index,'equity':s.to_numpy(float)})


def fixed_mix(eqs,weights):
    return rt.weighted_mix(eqs,weights)


def phase_list(h,long_mode=False):
    step=max(1,round(h/5))
    if not long_mode: return list(range(step))
    if step<=4:return list(range(step))
    return sorted(set(np.linspace(0,step-1,4).round().astype(int).tolist()))


def run_candidate(q,h,n,e,k,cal,members,bm,cost=1.0,long_mode=False):
    phases=phase_list(h,long_mode)
    eqs=[]; turns=[]
    for ph in phases:
        st,eq,tr,tm=ma.run_panel(subset(q,h,ph),cal,members,bm,n=n,entry=e,keep=k,cost=cost)
        eqs.append(eq); turns.append(float(st.get('turnover',np.nan)))
    eq=fixed_mix(eqs,[1/len(eqs)]*len(eqs))
    s=series_from_eq(eq)
    full=perf_series(s); train=perf_series(s,START,TRAIN_END); pseudo=perf_series(s,PSEUDO,END)
    return eq,{**full,'train_cagr':train['cagr'],'train_mdd':train['max_drawdown'],'train_sharpe':train['sharpe'],'train_calmar':train['calmar'],'pseudo_cagr':pseudo['cagr'],'pseudo_mdd':pseudo['max_drawdown'],'pseudo_sharpe':pseudo['sharpe'],'phase_count':len(phases),'turnover_mean':float(np.nanmean(turns)) if len(turns) else np.nan}


def select_short(p,cal,members,bm):
    rows=[]; cache={}
    for fam in SHORT_FAMILIES:
        q=make_short_q(p,fam)
        for h in SHORT_H:
            for n in SHORT_N:
                for e,k in SHORT_BUF:
                    print('SHORT',fam,h,n,e,k,flush=True)
                    eq,st=run_candidate(q,h,n,e,k,cal,members,bm,1.0,long_mode=False)
                    key=f'{fam}|h{h}|n{n}|e{e}|k{k}'
                    rows.append({**st,'family':fam,'H':h,'N':n,'entry':e,'keep':k,'key':key})
                    cache[key]=(q,eq,(h,n,e,k))
    d=pd.DataFrame(rows)
    ok=d[(d.train_cagr>0)&(d.train_mdd>-0.45)].copy()
    if len(ok)==0:ok=d.copy()
    win=ok.sort_values(['train_calmar','train_sharpe','turnover_mean'],ascending=[False,False,True]).iloc[0]
    return d,cache,str(win.key)


def select_long(p,cal,members,bm):
    rows=[]; cache={}
    for fam in LONG_FAMILIES:
        q=make_long_q(p,fam)
        for h in LONG_H:
            for n in LONG_N:
                e,k=LONG_BUF
                print('LONG',fam,h,n,flush=True)
                eq,st=run_candidate(q,h,n,e,k,cal,members,bm,1.0,long_mode=True)
                key=f'{fam}|h{h}|n{n}|e{e}|k{k}'
                rows.append({**st,'family':fam,'H':h,'N':n,'entry':e,'keep':k,'key':key})
                cache[key]=(q,eq,(h,n,e,k))
    d=pd.DataFrame(rows)
    ok=d[(d.train_cagr>0)&(d.train_mdd>-0.45)].copy()
    if len(ok)==0:ok=d.copy()
    win=ok.sort_values(['train_calmar','train_sharpe'],ascending=[False,False]).iloc[0]
    return d,cache,str(win.key)


def medium_equity(p,cal,members,bm,cost=1.0):
    q=iv3.build_candidates(p)['mom_cfo10_qv10']
    phase_eq=[]
    for ph in (0,4,8):
        z=subset(q,60,ph)
        _,e5,_,_=ma.run_panel(z,cal,members,bm,n=5,entry=.10,keep=.30,cost=cost)
        _,e10,_,_=ma.run_panel(z,cal,members,bm,n=10,entry=.10,keep=.30,cost=cost)
        phase_eq.append(fixed_mix([e5,e10],[.25,.75]))
    return fixed_mix(phase_eq,[1/3,1/3,1/3])


def correlations(eqs,bm):
    names=list(eqs)
    r={}
    for k,e in eqs.items():
        s=series_from_eq(e); r[k]=s.pct_change()
    R=pd.concat(r,axis=1).dropna(how='all').fillna(0.0)
    train=R.loc[START:TRAIN_END].corr(); pseudo=R.loc[PSEUDO:END].corr(); full=R.corr()
    br=bm.pct_change(fill_method=None).reindex(R.index)
    down=R.loc[br<0].corr()
    roll=[]
    for a in names:
        for b in names:
            if a>=b: continue
            rc=R[a].rolling(252,min_periods=126).corr(R[b]).dropna()
            if len(rc): roll.append({'pair':f'{a}-{b}','median':rc.median(),'p10':rc.quantile(.10),'p90':rc.quantile(.90),'min':rc.min(),'max':rc.max()})
    return R,train,pseudo,full,down,pd.DataFrame(roll)


def allocation_grid(eqs,standalone):
    rows=[]; cache={}
    for name,w in ALLOCATIONS.items():
        eq=fixed_mix([eqs['short'],eqs['medium'],eqs['long']],list(w)); s=series_from_eq(eq)
        full=perf_series(s); tr=perf_series(s,START,TRAIN_END); ps=perf_series(s,PSEUDO,END)
        weighted_cagr=sum(wi*standalone[sl]['train_cagr'] for wi,sl in zip(w,['short','medium','long']))
        rows.append({'allocation':name,'w_short':w[0],'w_medium':w[1],'w_long':w[2],**full,'train_cagr':tr['cagr'],'train_mdd':tr['max_drawdown'],'train_sharpe':tr['sharpe'],'train_calmar':tr['calmar'],'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['max_drawdown'],'pseudo_sharpe':ps['sharpe'],'weighted_train_cagr_reference':weighted_cagr,'constraint_pass':int(tr['cagr']>=weighted_cagr-.02)})
        cache[name]=eq
    d=pd.DataFrame(rows)
    ok=d[d.constraint_pass==1].copy()
    if len(ok)==0:ok=d.copy()
    win=ok.sort_values(['train_calmar','train_sharpe'],ascending=False).iloc[0]
    return d,cache,str(win.allocation)


def annual_table(eqs):
    rows=[]
    for name,e in eqs.items():
        s=series_from_eq(e)
        for y,g in s.groupby(s.index.year):
            before=s[s.index<pd.Timestamp(f'{y}-01-01')]
            st=float(before.iloc[-1]) if len(before) else float(g.iloc[0])
            rows.append({'sleeve':name,'year':int(y),'return':float(g.iloc[-1]/st-1)})
    return pd.DataFrame(rows)


def selected_cost_equity(cache,key,cal,members,bm,cost,long_mode):
    q,_,cfg=cache[key]; h,n,e,k=cfg
    return run_candidate(q,h,n,e,k,cal,members,bm,cost,long_mode=long_mode)[0]


def main():
    p,z,cal,members,ua,market_code,bm=load_base_panel()
    p=add_short_features(p,cal); p=add_long_ranks(p,z)

    short_grid,short_cache,short_key=select_short(p,cal,members,bm)
    long_grid,long_cache,long_key=select_long(p,cal,members,bm)
    short_grid.to_csv(OUT/'short_grid.csv',index=False); long_grid.to_csv(OUT/'long_grid.csv',index=False)

    short_eq=short_cache[short_key][1]
    long_eq=long_cache[long_key][1]
    medium_eq=medium_equity(p,cal,members,bm,1.0)
    eqs={'short':short_eq,'medium':medium_eq,'long':long_eq}

    standalone={}
    sm=[]
    for name,e in eqs.items():
        s=series_from_eq(e); f=perf_series(s); tr=perf_series(s,START,TRAIN_END); ps=perf_series(s,PSEUDO,END)
        standalone[name]={'train_cagr':tr['cagr']}
        sm.append({'sleeve':name,**f,'train_cagr':tr['cagr'],'train_mdd':tr['max_drawdown'],'train_sharpe':tr['sharpe'],'train_calmar':tr['calmar'],'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['max_drawdown'],'pseudo_sharpe':ps['sharpe']})
    pd.DataFrame(sm).to_csv(OUT/'selected_sleeves.csv',index=False)

    R,ctr,cp,cf,cd,roll=correlations(eqs,bm)
    R.to_csv(OUT/'selected_daily_returns.csv')
    ctr.to_csv(OUT/'correlation_train.csv'); cp.to_csv(OUT/'correlation_pseudo.csv'); cf.to_csv(OUT/'correlation_full.csv'); cd.to_csv(OUT/'correlation_down_days.csv'); roll.to_csv(OUT/'rolling_corr_252.csv',index=False)

    ag,acache,alloc_key=allocation_grid(eqs,standalone); ag.to_csv(OUT/'allocation_grid.csv',index=False)
    portfolio=acache[alloc_key]
    selected_weights=ALLOCATIONS[alloc_key]

    # Marginal contribution: remove one sleeve and renormalize remaining original weights.
    marg=[]; basep=perf_series(series_from_eq(portfolio))
    for i,name in enumerate(['short','medium','long']):
        keep=[j for j in range(3) if j!=i]; ws=np.array([selected_weights[j] for j in keep],float); ws=ws/ws.sum()
        names=[['short','medium','long'][j] for j in keep]
        e=fixed_mix([eqs[n] for n in names],ws.tolist()); st=perf_series(series_from_eq(e))
        marg.append({'removed':name,'portfolio_cagr':basep['cagr'],'without_cagr':st['cagr'],'delta_cagr':basep['cagr']-st['cagr'],'portfolio_sharpe':basep['sharpe'],'without_sharpe':st['sharpe'],'delta_sharpe':basep['sharpe']-st['sharpe'],'portfolio_mdd':basep['max_drawdown'],'without_mdd':st['max_drawdown']})
    pd.DataFrame(marg).to_csv(OUT/'marginal_contribution.csv',index=False)

    # Cost stress only after train-only sleeve and allocation selection.
    costrows=[]
    for cm in (2.0,4.0):
        se=selected_cost_equity(short_cache,short_key,cal,members,bm,cm,False)
        me=medium_equity(p,cal,members,bm,cm)
        le=selected_cost_equity(long_cache,long_key,cal,members,bm,cm,True)
        pe=fixed_mix([se,me,le],list(selected_weights)); st=perf_series(series_from_eq(pe)); tr=perf_series(series_from_eq(pe),START,TRAIN_END); ps=perf_series(series_from_eq(pe),PSEUDO,END)
        costrows.append({'cost_mult':cm,**st,'train_cagr':tr['cagr'],'train_mdd':tr['max_drawdown'],'train_sharpe':tr['sharpe'],'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['max_drawdown'],'pseudo_sharpe':ps['sharpe']})
    pd.DataFrame(costrows).to_csv(OUT/'allocation_cost_stress.csv',index=False)

    alleqs={**eqs,'portfolio':portfolio}; annual_table(alleqs).to_csv(OUT/'annual_selected.csv',index=False)
    for name,e in alleqs.items(): e.to_csv(OUT/f'equity_{name}.csv',index=False)

    med=next(x for x in sm if x['sleeve']=='medium')
    sel=ag[ag.allocation==alloc_key].iloc[0]
    marginal=pd.read_csv(OUT/'marginal_contribution.csv')
    min_corr=min(float(cf.loc['short','medium']),float(cf.loc['long','medium']))
    gates={
      'all_sleeves_pseudo_positive':int(all(x['pseudo_cagr']>0 for x in sm)),
      'portfolio_pseudo_positive':int(sel.pseudo_cagr>0),
      'portfolio_train_calmar_gt_medium':int(sel.train_calmar>med['train_calmar']),
      'portfolio_mdd_not_gt_medium_by_5pp':int(sel.max_drawdown>=med['max_drawdown']-.05),
      'one_nonmedium_corr_le_060':int(min_corr<=.60),
      'no_sleeve_removal_sharpe_improves_gt_010':int((marginal.without_sharpe<=marginal.portfolio_sharpe+.10).all()),
    }
    cst=pd.read_csv(OUT/'allocation_cost_stress.csv')
    gates['cost2_positive_cagr']=int(float(cst[cst.cost_mult==2.0].cagr.iloc[0])>0)
    pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_csv(OUT/'gates.csv',index=False)

    meta={'status':'NEW_STOCK_LEVEL_MULTI_ALPHA_RESEARCH_NOT_ORIGINAL_EXACT','prereg':'MULTI_ALPHA_PREREG_2026-09-03.md committed before run','short_selected':short_key,'medium_fixed':'mom_cfo10_qv10 H60 ranktilt N5_25_N10_75 stagger phases 0/4/8','long_selected':long_key,'allocation_selected_train_only':alloc_key,'weights':{'short':selected_weights[0],'medium':selected_weights[1],'long':selected_weights[2]},'market_factor':market_code,'universe_audit':ua,'gates_passed':sum(gates.values()),'gates_total':len(gates)}
    (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
    print('=== SHORT WINNER ===',short_key,flush=True)
    print(short_grid.sort_values(['train_calmar','train_sharpe'],ascending=False).head(10).to_string(index=False),flush=True)
    print('=== LONG WINNER ===',long_key,flush=True)
    print(long_grid.sort_values(['train_calmar','train_sharpe'],ascending=False).head(10).to_string(index=False),flush=True)
    print('=== SLEEVES ==='); print(pd.DataFrame(sm).to_string(index=False),flush=True)
    print('=== CORR FULL ==='); print(cf.to_string(),flush=True)
    print('=== ALLOCATIONS ==='); print(ag.sort_values(['train_calmar','train_sharpe'],ascending=False).to_string(index=False),flush=True)
    print('=== SELECTED ===',alloc_key,selected_weights,flush=True)
    print('=== COST ==='); print(pd.DataFrame(costrows).to_string(index=False),flush=True)
    print('=== GATES ==='); print(pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_string(index=False),flush=True)

if __name__=='__main__': main()
