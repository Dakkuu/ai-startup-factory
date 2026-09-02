from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import akshare as ak

import run_geff_fundamental_fastpit_v1 as fp
import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
import run_10y_era_backtest as base
hv3.patch()

OUT=Path('results_geff_fundamental_value_pit_v1'); OUT.mkdir(exist_ok=True)
START=pd.Timestamp('2016-08-02'); TRAIN_END=pd.Timestamp('2021-12-31'); PSEUDO=pd.Timestamp('2022-01-01'); END=pd.Timestamp('2026-07-29')
HORIZONS=(60,75,90); N=10; ENTRY=.10; KEEP=.30
BASE_W={'iv':.25,'down':.15,'rmom':.35,'tstat':.25}; BLEND=(.10,.20,.30)
FACTORS=('earnings_yield','book_yield','cashflow_yield','value3','quality_value')

def ncode(x,cmap): return fp.norm_code(x,cmap)
def retry(fn,q): return fp.retry(fn,q,tries=4)

def fetch_events(valid,cal):
    cmap={c[-6:]:c for c in valid}; rows=[]; audit=[]
    for i,q in enumerate(fp.quarter_ends(2010,2026),1):
        print('VALUE QUARTER',i,q,flush=True)
        f=retry(ak.stock_yjbb_em,q); k=retry(ak.stock_yjkb_em,q)
        audit += [{'quarter':q,'source':'formal','rows_raw':len(f)},{'quarter':q,'source':'quick','rows_raw':len(k)}]
        for src,df in [('quick',k),('formal',f)]:
            if df.empty: continue
            z=pd.DataFrame(); z['code']=df['股票代码'].map(lambda x:ncode(x,cmap)); z['report_date']=pd.to_datetime(q)
            ac='最新公告日期' if src=='formal' else '公告日期'; z['ann_date']=pd.to_datetime(df[ac],errors='coerce')
            z['eps']=pd.to_numeric(df.get('每股收益'),errors='coerce')
            z['bps']=pd.to_numeric(df.get('每股净资产'),errors='coerce')
            z['roe']=pd.to_numeric(df.get('净资产收益率'),errors='coerce')
            z['gross_margin']=pd.to_numeric(df.get('销售毛利率'),errors='coerce') if '销售毛利率' in df else np.nan
            z['ocfps']=pd.to_numeric(df.get('每股经营现金流量'),errors='coerce') if '每股经营现金流量' in df else np.nan
            z['source']=src; rows.append(z[z.code.notna()])
    pd.DataFrame(audit).to_csv(OUT/'fetch_audit.csv',index=False)
    E=pd.concat(rows,ignore_index=True); E=E[E.ann_date.notna()].copy(); E['available_date']=E.ann_date.map(lambda d:fp.next_trade_date(cal,d)); E=E[E.available_date.notna()]
    E=E.sort_values(['code','available_date','report_date','source']).drop_duplicates(['code','available_date'],keep='last')
    E.to_csv(OUT/'value_events.csv.gz',index=False,compression='gzip'); return E

def attach(p,E,cal):
    q=p.reset_index().rename(columns={'index':'_row'})[['_row','signal_date','code']].copy(); q['signal_date']=pd.to_datetime(q.signal_date); out=[]
    for code,g in q.groupby('code',sort=False):
        e=E[E.code==code].sort_values('available_date')
        if len(e): x=pd.merge_asof(g.sort_values('signal_date'),e[['available_date','report_date','eps','bps','ocfps','roe','gross_margin']],left_on='signal_date',right_on='available_date',direction='backward')
        else:
            x=g.copy();
            for c in ['available_date','report_date','eps','bps','ocfps','roe','gross_margin']: x[c]=np.nan
        c=base.qb.read_bin(code,'close',cal); f=base.qb.read_bin(code,'factor',cal)
        raw=(c/f.replace(0,np.nan)) if len(c) and len(f) else pd.Series(dtype=float)
        x['raw_price']=raw.reindex(pd.DatetimeIndex(x.signal_date)).to_numpy(float) if len(raw) else np.nan
        out.append(x)
    A=pd.concat(out,ignore_index=True); A['age_days']=(A.signal_date-pd.to_datetime(A.available_date)).dt.days; stale=A.age_days>550
    A.loc[stale,['eps','bps','ocfps','roe','gross_margin']]=np.nan
    A['earnings_yield']=A.eps/A.raw_price.replace(0,np.nan); A['book_yield']=A.bps/A.raw_price.replace(0,np.nan); A['cashflow_yield']=A.ocfps/A.raw_price.replace(0,np.nan)
    for c in ['earnings_yield','book_yield','cashflow_yield']:
        A.loc[~np.isfinite(A[c]),c]=np.nan
    return A

def score(p,A):
    z=A.copy()
    for c in ['earnings_yield','book_yield','cashflow_yield','roe','gross_margin']: z[c+'_z']=z.groupby('signal_date')[c].transform(fp.robust_z)
    z['value3_raw']=z[['earnings_yield_z','book_yield_z','cashflow_yield_z']].mean(axis=1,skipna=True)
    z['quality_value_raw']=z[['earnings_yield_z','book_yield_z','cashflow_yield_z','roe_z','gross_margin_z']].mean(axis=1,skipna=True)
    for c in FACTORS:
        raw=c+'_z' if c in ('earnings_yield','book_yield','cashflow_yield') else c+'_raw'; z[c+'_rank']=z.groupby('signal_date')[raw].rank(pct=True,method='average',ascending=False)
    p2=p.reset_index().rename(columns={'index':'_row'}).merge(z[['_row']+[c+'_rank' for c in FACTORS]],on='_row',how='left').drop(columns='_row'); return p2,z

def ictab(z,p):
    m=z.merge(p.reset_index().rename(columns={'index':'_row'})[['_row','fwd60']],on='_row',how='left'); rows=[]
    for c in FACTORS:
        for split,a,b in [('train',START,TRAIN_END),('pseudo',PSEUDO,END)]:
            x=m[(m.signal_date>=a)&(m.signal_date<=b)][['signal_date',c+'_rank','fwd60']].dropna(); ics=[]; sp=[]
            for d,g in x.groupby('signal_date'):
                if len(g)<150: continue
                ic=(-g[c+'_rank']).corr(g.fwd60,method='spearman');
                if np.isfinite(ic):ics.append(float(ic))
                hi=g[g[c+'_rank']<=.1].fwd60; lo=g[g[c+'_rank']>=.9].fwd60
                if len(hi)>10 and len(lo)>10:sp.append(float(hi.mean()-lo.mean()))
            ar=np.asarray(ics,float); sd=ar.std(ddof=1) if len(ar)>1 else np.nan; rows.append({'factor':c,'split':split,'dates':len(ar),'mean_ic':ar.mean() if len(ar) else np.nan,'ic_t':ar.mean()/sd*np.sqrt(len(ar)) if len(ar)>1 and sd>0 else np.nan,'positive_ic_share':(ar>0).mean() if len(ar) else np.nan,'spread60':np.mean(sp) if sp else np.nan})
    return pd.DataFrame(rows)
def baseq(p):return mega.make_rank(p,{'name':'base_mom','kind':'gate','g':{'ef':.55},'w':BASE_W})
def blendq(p,c,w):
    q=baseq(p); f=q[c+'_rank'].fillna(.5); m=np.isfinite(q.rank_test); raw=(1-w)*q.loc[m,'rank_test']+w*f.loc[m]; q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True); return q
def subset(q,h,ph):
    ds=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(ds[ph::max(1,round(h/5))]); cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]; z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')
def pc(eq,a,b):return fp.period_cagr(eq,a,b)
def portfolio(p,cal,members,bm,ic):
    tr=ic[ic.split=='train']; eligible=list(tr[(tr.mean_ic>0)&(tr.ic_t>=2)].sort_values('ic_t',ascending=False).factor); tests=[('base_mom','',0.0)]+[(f,f,w) for f in eligible for w in BLEND]; rows=[]
    for label,f,w in tests:
        q=baseq(p) if not f else blendq(p,f,w)
        for h in HORIZONS:
            for ph in range(max(1,round(h/5))):
                st,eq,trd,tm=ma.run_panel(subset(q,h,ph),cal,members,bm,n=N,entry=ENTRY,keep=KEEP); st.update(candidate=label if not f else f'{f}_w{w:.2f}',fund_factor=f,fund_weight=w,H=h,phase=ph,half1_cagr=pc(eq,START,pd.Timestamp('2019-12-31')),half2_cagr=pc(eq,pd.Timestamp('2020-01-01'),TRAIN_END),pseudo_cagr=pc(eq,PSEUDO,END)); rows.append(st)
    d=pd.DataFrame(rows); d.to_csv(OUT/'portfolio_all_phase.csv',index=False); a=[]
    for c,g in d.groupby('candidate'):a.append({'candidate':c,'runs':len(g),'full_cagr_median':g.cagr.median(),'full_cagr_p25':g.cagr.quantile(.25),'full_cagr_min':g.cagr.min(),'mdd_median':g.max_drawdown.median(),'mdd_worst':g.max_drawdown.min(),'sharpe_median':g.sharpe.median(),'train_maximin':min(g.half1_cagr.median(),g.half2_cagr.median()),'pseudo_cagr_median':g.pseudo_cagr.median(),'pseudo_cagr_p25':g.pseudo_cagr.quantile(.25),'h60_median':g[g.H==60].cagr.median(),'h75_median':g[g.H==75].cagr.median(),'h90_median':g[g.H==90].cagr.median()})
    s=pd.DataFrame(a).sort_values(['train_maximin','full_cagr_p25'],ascending=False); s.to_csv(OUT/'portfolio_summary.csv',index=False); return d,s,eligible

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=True); p=strict.attach_gap_flags(p,cal,'board'); E=fetch_events(sorted(p.code.unique()),cal); A=attach(p,E,cal); A.to_csv(OUT/'pit_value_attached.csv.gz',index=False,compression='gzip'); p2,z=score(p,A); cov=pd.DataFrame([{'factor':c,'coverage':z[c].notna().mean() if c in z else z[c+'_raw'].notna().mean()} for c in FACTORS]); cov.to_csv(OUT/'coverage.csv',index=False); ic=ictab(z,p); ic.to_csv(OUT/'fundamental_ic.csv',index=False); d,s,eligible=portfolio(p2,cal,members,bm,ic); control=d[(d.candidate=='base_mom')&(d.H==90)&(d.phase==0)].iloc[0]; meta={'status':'PIT_PER_SHARE_VALUE_TEST','raw_price_rule':'qlib close/factor, derived from YahooNormalize1d algebra','control_h90_phase0_cagr':float(control.cagr),'control_pass':bool(abs(float(control.cagr)-0.2229800205653422)<1e-10),'eligible_by_train_ic_only':eligible,'sources':['AKShare/Eastmoney stock_yjbb_em','stock_yjkb_em'],'pit_rule':'next exchange day after latest announcement','horizons':HORIZONS,'N':N,'market_factor':market_code,'universe_audit':ua}; (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str)); print(cov.to_string(index=False)); print(ic.to_string(index=False)); print(s.to_string(index=False)); print(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__':main()
