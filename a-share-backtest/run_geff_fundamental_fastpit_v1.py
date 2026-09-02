from __future__ import annotations

from pathlib import Path
import json, math, time
import numpy as np
import pandas as pd

import akshare as ak

import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT = Path('results_geff_fundamental_fastpit_v1'); OUT.mkdir(exist_ok=True)
START = pd.Timestamp('2016-08-02'); TRAIN_END = pd.Timestamp('2021-12-31')
PSEUDO_START = pd.Timestamp('2022-01-01'); END = pd.Timestamp('2026-07-29')
HORIZONS = (60, 75, 90); N = 10; ENTRY = .10; KEEP = .30
BASE_W = {'iv':.25,'down':.15,'rmom':.35,'tstat':.25}

# Pre-registered fundamental composites. High raw values are treated as good before cross-sectional ranking.
FUND_SPECS = {
    'profitability': ['roe','gross_margin'],
    'growth': ['revenue_yoy','profit_yoy'],
    'cash_quality': ['ocf_to_eps'],
    'quality_growth': ['roe','gross_margin','revenue_yoy','profit_yoy','ocf_to_eps'],
    'forecast': ['forecast_change'],
    'all_fund': ['roe','gross_margin','revenue_yoy','profit_yoy','ocf_to_eps','forecast_change'],
}
BLEND_WEIGHTS = (.10, .20, .30)


def quarter_ends(y0=2010, y1=2026):
    out=[]
    for y in range(y0,y1+1):
        for md in ('0331','0630','0930','1231'):
            d=f'{y}{md}'
            if d <= '20260630': out.append(d)
    return out


def retry(fn, date, tries=4):
    err=None
    for i in range(tries):
        try:
            df=fn(date=date)
            if df is None: return pd.DataFrame()
            return df.copy()
        except Exception as e:
            err=repr(e); time.sleep(1.5*(i+1))
    print('FETCH_FAIL', fn.__name__, date, err, flush=True)
    return pd.DataFrame()


def norm_code(x, valid_map):
    s=str(x).split('.')[0].zfill(6)
    return valid_map.get(s)


def fetch_all(valid_codes):
    cmap={c[-6:]:c for c in valid_codes}
    formal=[]; quick=[]; forecast=[]; audit=[]
    for qi,q in enumerate(quarter_ends(),1):
        print('QUARTER',qi,q,flush=True)
        for name,fn,bucket in [('formal',ak.stock_yjbb_em,formal),('quick',ak.stock_yjkb_em,quick),('forecast',ak.stock_yjyg_em,forecast)]:
            df=retry(fn,q)
            audit.append({'quarter':q,'source':name,'rows_raw':len(df)})
            if df.empty: continue
            df['report_date']=pd.to_datetime(q)
            codecol='股票代码'
            if codecol not in df: continue
            df['code']=df[codecol].map(lambda x:norm_code(x,cmap))
            df=df[df.code.notna()].copy()
            if name=='formal':
                ren={'最新公告日期':'ann_date','每股收益':'eps','营业总收入-同比增长':'revenue_yoy','营业总收入-季度环比增长':'revenue_qoq','净利润-同比增长':'profit_yoy','净利润-季度环比增长':'profit_qoq','净资产收益率':'roe','每股经营现金流量':'ocfps','销售毛利率':'gross_margin','所处行业':'industry'}
                df=df.rename(columns=ren)
                cols=['code','report_date','ann_date','eps','revenue_yoy','revenue_qoq','profit_yoy','profit_qoq','roe','ocfps','gross_margin','industry']
            elif name=='quick':
                ren={'公告日期':'ann_date','每股收益':'eps','营业收入-同比增长':'revenue_yoy','营业收入-季度环比增长':'revenue_qoq','净利润-同比增长':'profit_yoy','净利润-季度环比增长':'profit_qoq','净资产收益率':'roe','所处行业':'industry'}
                df=df.rename(columns=ren)
                cols=['code','report_date','ann_date','eps','revenue_yoy','revenue_qoq','profit_yoy','profit_qoq','roe','industry']
            else:
                ren={'公告日期':'ann_date','业绩变动幅度':'forecast_change','预测数值':'forecast_value','预告类型':'forecast_type','预测指标':'forecast_metric'}
                df=df.rename(columns=ren)
                cols=['code','report_date','ann_date','forecast_change','forecast_value','forecast_type','forecast_metric']
            for c in cols:
                if c not in df: df[c]=np.nan
            z=df[cols].copy(); z['source']=name; bucket.append(z)
    F=pd.concat(formal,ignore_index=True) if formal else pd.DataFrame()
    K=pd.concat(quick,ignore_index=True) if quick else pd.DataFrame()
    Y=pd.concat(forecast,ignore_index=True) if forecast else pd.DataFrame()
    pd.DataFrame(audit).to_csv(OUT/'fetch_audit.csv',index=False)
    for name,df in [('formal',F),('quick',K),('forecast',Y)]:
        if len(df): df.to_csv(OUT/f'{name}_raw.csv.gz',index=False,compression='gzip')
    return F,K,Y


def next_trade_date(cal, d):
    d=pd.Timestamp(d)
    k=cal.searchsorted(d,side='right')
    return cal[k] if k<len(cal) else pd.NaT


def prep_events(F,K,Y,cal):
    common=['code','report_date','ann_date','eps','revenue_yoy','revenue_qoq','profit_yoy','profit_qoq','roe','ocfps','gross_margin','industry','source']
    blocks=[]
    for df in (K,F):
        if df.empty: continue
        z=df.copy()
        for c in common:
            if c not in z:z[c]=np.nan
        blocks.append(z[common])
    E=pd.concat(blocks,ignore_index=True) if blocks else pd.DataFrame(columns=common)
    if len(E):
        E['ann_date']=pd.to_datetime(E.ann_date,errors='coerce')
        E=E[E.ann_date.notna()].copy()
        E['available_date']=E.ann_date.map(lambda d:next_trade_date(cal,d))
        E=E[E.available_date.notna()].copy()
        for c in ['eps','revenue_yoy','revenue_qoq','profit_yoy','profit_qoq','roe','ocfps','gross_margin']:
            E[c]=pd.to_numeric(E[c],errors='coerce')
        E['ocf_to_eps']=E.ocfps/E.eps.abs().replace(0,np.nan)
        E['ocf_to_eps']=E.ocf_to_eps.clip(-20,20)
        E=E.sort_values(['code','available_date','report_date','source'])
        # If multiple records become available together, formal sorts after quick and wins.
        E=E.drop_duplicates(['code','available_date'],keep='last')
    if len(Y):
        Y=Y.copy(); Y['ann_date']=pd.to_datetime(Y.ann_date,errors='coerce'); Y=Y[Y.ann_date.notna()].copy()
        Y['available_date']=Y.ann_date.map(lambda d:next_trade_date(cal,d)); Y=Y[Y.available_date.notna()].copy()
        Y['forecast_change']=pd.to_numeric(Y.forecast_change,errors='coerce').clip(-500,1000)
        Y=Y.sort_values(['code','available_date','report_date']).drop_duplicates(['code','available_date'],keep='last')
    return E,Y


def asof_attach(p,E,Y):
    base=p[['signal_date','code']].copy(); base['signal_date']=pd.to_datetime(base.signal_date)
    out=[]
    for code,g in base.groupby('code',sort=False):
        g=g.sort_values('signal_date').copy()
        e=E[E.code==code].sort_values('available_date') if len(E) else pd.DataFrame()
        if len(e):
            cols=['available_date','report_date','eps','revenue_yoy','revenue_qoq','profit_yoy','profit_qoq','roe','ocfps','gross_margin','ocf_to_eps','industry','source']
            x=pd.merge_asof(g,e[cols],left_on='signal_date',right_on='available_date',direction='backward')
        else:
            x=g.copy()
            for c in ['available_date','report_date','eps','revenue_yoy','revenue_qoq','profit_yoy','profit_qoq','roe','ocfps','gross_margin','ocf_to_eps','industry','source']: x[c]=np.nan
        y=Y[Y.code==code].sort_values('available_date') if len(Y) else pd.DataFrame()
        if len(y):
            y2=y[['available_date','forecast_change']].rename(columns={'available_date':'forecast_available_date'})
            x=pd.merge_asof(x.sort_values('signal_date'),y2,left_on='signal_date',right_on='forecast_available_date',direction='backward')
        else: x['forecast_change']=np.nan
        out.append(x)
    A=pd.concat(out,ignore_index=True)
    A['fund_age_days']=(A.signal_date-pd.to_datetime(A.available_date)).dt.days
    # stale accounting data is made unavailable, not forward-filled indefinitely
    stale=A.fund_age_days>550
    cols=['eps','revenue_yoy','revenue_qoq','profit_yoy','profit_qoq','roe','ocfps','gross_margin','ocf_to_eps']
    A.loc[stale,cols]=np.nan
    return A


def robust_z(s):
    x=pd.to_numeric(s,errors='coerce').replace([np.inf,-np.inf],np.nan)
    if x.notna().sum()<20:return pd.Series(np.nan,index=s.index)
    lo=x.quantile(.05); hi=x.quantile(.95); x=x.clip(lo,hi)
    med=x.median(); mad=(x-med).abs().median()
    scale=1.4826*mad if np.isfinite(mad) and mad>1e-12 else x.std(ddof=0)
    return (x-med)/(scale if np.isfinite(scale) and scale>1e-12 else 1.0)


def add_fund_scores(p,A):
    key=p.reset_index().rename(columns={'index':'_row'})[['_row','signal_date','code']]
    z=key.merge(A,on=['signal_date','code'],how='left')
    factor_cols=['roe','gross_margin','revenue_yoy','profit_yoy','ocf_to_eps','forecast_change']
    for c in factor_cols:
        z[c+'_z']=z.groupby('signal_date')[c].transform(robust_z)
    for name,fs in FUND_SPECS.items():
        zz=[f+'_z' for f in fs]
        z[name+'_raw']=z[zz].mean(axis=1,skipna=True)
        z[name+'_n']=z[zz].notna().sum(axis=1)
        # high raw good -> low percentile better, same orientation as GEff rank_test
        z[name+'_rank']=z.groupby('signal_date')[name+'_raw'].rank(pct=True,method='average',ascending=False)
    out=p.reset_index().rename(columns={'index':'_row'}).merge(z[['_row']+[n+'_rank' for n in FUND_SPECS]+['fund_age_days']],on='_row',how='left').drop(columns='_row')
    return out,z


def ic_summary(z,p):
    m=z.merge(p.reset_index().rename(columns={'index':'_row'})[['_row','fwd60']],on='_row',how='left')
    rows=[]
    for name in FUND_SPECS:
        col=name+'_rank'
        for split,a,b in [('train',START,TRAIN_END),('pseudo',PSEUDO_START,END)]:
            x=m[(m.signal_date>=a)&(m.signal_date<=b)][['signal_date',col,'fwd60']].dropna()
            ics=[]; spreads=[]
            for d,g in x.groupby('signal_date'):
                if len(g)<150:continue
                ic=(-g[col]).corr(g.fwd60,method='spearman')
                if np.isfinite(ic):ics.append(ic)
                hi=g[g[col]<=.10].fwd60; lo=g[g[col]>=.90].fwd60
                if len(hi)>10 and len(lo)>10:spreads.append(float(hi.mean()-lo.mean()))
            arr=np.asarray(ics,float); sd=arr.std(ddof=1) if len(arr)>1 else np.nan
            rows.append({'factor':name,'split':split,'dates':len(arr),'mean_ic':float(arr.mean()) if len(arr) else np.nan,'ic_t':float(arr.mean()/sd*np.sqrt(len(arr))) if len(arr)>1 and sd>0 else np.nan,'positive_ic_share':float((arr>0).mean()) if len(arr) else np.nan,'top_bottom_spread60':float(np.mean(spreads)) if spreads else np.nan})
    return pd.DataFrame(rows)


def make_blend_q(p2,name,fw):
    base=mega.make_rank(p2,{'name':'base_mom','kind':'gate','g':{'ef':.55},'w':BASE_W})
    f=base[name+'_rank'].copy()
    f=f.fillna(.50)
    m=np.isfinite(base.rank_test)
    raw=(1-fw)*base.loc[m,'rank_test']+fw*f.loc[m]
    base.loc[m,'rank_test']=raw.groupby(base.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return base


def subset_exact(q,h,phase=0):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
    step=max(1,round(h/5)); chosen=set(dates[phase::step])
    cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')


def period_cagr(eq,a,b):
    z=eq.copy(); z['trade_date']=pd.to_datetime(z.trade_date); z=z[(z.trade_date>=a)&(z.trade_date<=b)]
    if len(z)<2:return np.nan
    days=max(1,(z.trade_date.iloc[-1]-z.trade_date.iloc[0]).days)
    return float((float(z.equity.iloc[-1])/float(z.equity.iloc[0]))**(365.25/days)-1)


def backtest_candidates(p2,cal,members,bm,ics):
    # Only factors with positive TRAIN IC enter portfolio blending. Threshold fixed before pseudo/full portfolio validation.
    train=ics[ics.split=='train'].set_index('factor')
    eligible=[n for n in FUND_SPECS if n in train.index and train.loc[n,'mean_ic']>0 and train.loc[n,'ic_t']>=2]
    rows=[]
    tests=[('base_mom',None,0.0)] + [(n,n,w) for n in eligible for w in BLEND_WEIGHTS]
    for label,fname,fw in tests:
        q=mega.make_rank(p2,{'name':'base_mom','kind':'gate','g':{'ef':.55},'w':BASE_W}) if fname is None else make_blend_q(p2,fname,fw)
        for h in HORIZONS:
            step=max(1,round(h/5))
            for ph in range(step):
                st,eq,tr,tm=ma.run_panel(subset_exact(q,h,ph),cal,members,bm,n=N,entry=ENTRY,keep=KEEP)
                st.update(candidate=label if fname is None else f'{fname}_w{fw:.2f}',fund_factor=fname or '',fund_weight=fw,H=h,phase=ph,half1_cagr=period_cagr(eq,START,pd.Timestamp('2019-12-31')),half2_cagr=period_cagr(eq,pd.Timestamp('2020-01-01'),TRAIN_END),pseudo_cagr=period_cagr(eq,PSEUDO_START,END))
                rows.append(st)
    d=pd.DataFrame(rows); d.to_csv(OUT/'portfolio_all_phase.csv',index=False)
    agg=[]
    for cand,g in d.groupby('candidate'):
        agg.append({'candidate':cand,'runs':len(g),'full_cagr_median':g.cagr.median(),'full_cagr_p25':g.cagr.quantile(.25),'full_cagr_min':g.cagr.min(),'mdd_median':g.max_drawdown.median(),'mdd_worst':g.max_drawdown.min(),'sharpe_median':g.sharpe.median(),'train_maximin':min(g.half1_cagr.median(),g.half2_cagr.median()),'pseudo_cagr_median':g.pseudo_cagr.median(),'pseudo_cagr_p25':g.pseudo_cagr.quantile(.25),'h60_median':g[g.H==60].cagr.median(),'h75_median':g[g.H==75].cagr.median(),'h90_median':g[g.H==90].cagr.median()})
    a=pd.DataFrame(agg).sort_values(['train_maximin','full_cagr_p25'],ascending=False); a.to_csv(OUT/'portfolio_summary.csv',index=False)
    return d,a,eligible


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=True)
    p=strict.attach_gap_flags(p,cal,'board')
    valid=sorted(p.code.unique())
    F,K,Y=fetch_all(valid)
    E,Y=prep_events(F,K,Y,cal)
    A=asof_attach(p,E,Y); A.to_csv(OUT/'pit_attached_fundamentals.csv.gz',index=False,compression='gzip')
    p2,z=add_fund_scores(p,A)
    coverage=[]
    for c in ['roe','gross_margin','revenue_yoy','profit_yoy','ocf_to_eps','forecast_change']:
        coverage.append({'field':c,'coverage':float(z[c].notna().mean()),'train_coverage':float(z.loc[z.signal_date<=TRAIN_END,c].notna().mean()),'pseudo_coverage':float(z.loc[z.signal_date>=PSEUDO_START,c].notna().mean())})
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    ics=ic_summary(z,p); ics.to_csv(OUT/'fundamental_ic.csv',index=False)
    detail,summary,eligible=backtest_candidates(p2,cal,members,bm,ics)
    meta={'status':'POST_SOURCE_DISCOVERY_PREDECLARED_TEST','akshare_version':getattr(ak,'__version__','unknown'),'sources':['AKShare stock_yjbb_em / Eastmoney RPT_LICO_FN_CPD','AKShare stock_yjkb_em / Eastmoney RPT_FCI_PERFORMANCEE','AKShare stock_yjyg_em / Eastmoney RPT_PUBLIC_OP_NEWPREDICT'],'pit_rule':'announcement date becomes usable on next exchange trading day; merge_asof backward only','stale_after_days':550,'horizons':HORIZONS,'N':N,'entry':ENTRY,'keep':KEEP,'eligible_fund_factors_from_train_only':eligible,'market_factor':market_code,'universe_audit':ua}
    (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
    print('\n=== COVERAGE ==='); print(pd.DataFrame(coverage).to_string(index=False),flush=True)
    print('\n=== FUNDAMENTAL IC ==='); print(ics.to_string(index=False),flush=True)
    print('\n=== PORTFOLIO SUMMARY ==='); print(summary.to_string(index=False),flush=True)
    print('\n=== META ==='); print(json.dumps(meta,ensure_ascii=False,indent=2,default=str),flush=True)

if __name__=='__main__': main()
