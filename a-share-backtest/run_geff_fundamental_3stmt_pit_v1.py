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
hv3.patch()

OUT=Path('results_geff_fundamental_3stmt_pit_v1'); OUT.mkdir(exist_ok=True)
START=pd.Timestamp('2016-08-02'); TRAIN_END=pd.Timestamp('2021-12-31'); PSEUDO=pd.Timestamp('2022-01-01'); END=pd.Timestamp('2026-07-29')
HORIZONS=(60,75,90); N=10; ENTRY=.10; KEEP=.30
BASE_W={'iv':.25,'down':.15,'rmom':.35,'tstat':.25}
# Signs are economically predeclared; high 'good' is always better after transformation.
FACTORS={
 'cash_assets': +1,
 'low_leverage': +1,
 'low_receivables': +1,
 'low_inventory': +1,
 'roa_proxy': +1,
 'asset_turnover': +1,
 'net_margin': +1,
 'oper_margin': +1,
 'cfo_assets': +1,
 'accrual_quality': +1,
 'cash_conversion': +1,
 'low_finexp': +1,
 'balance_strength': +1,
 'quality_core': +1,
}
BLEND=(.10,.20)


def qends(): return fp.quarter_ends(2010,2026)
def retry(fn,date): return fp.retry(fn,date,tries=4)
def ncode(x,cmap): return fp.norm_code(x,cmap)

def fetch_tables(valid_codes):
    cmap={c[-6:]:c for c in valid_codes}; Bs=[]; Is=[]; Cs=[]; audit=[]
    for i,q in enumerate(qends(),1):
        print('3STMT QUARTER',i,'/',len(qends()),q,flush=True)
        b=retry(ak.stock_zcfz_em,q)
        try:
            bb=retry(ak.stock_zcfz_bj_em,q)
            if len(bb): b=pd.concat([b,bb],ignore_index=True)
        except Exception: pass
        inc=retry(ak.stock_lrb_em,q); cf=retry(ak.stock_xjll_em,q)
        for src,df in [('balance',b),('income',inc),('cashflow',cf)]: audit.append({'quarter':q,'source':src,'rows_raw':len(df)})
        if len(b):
            b=b.copy(); b['code']=b['股票代码'].map(lambda x:ncode(x,cmap)); b=b[b.code.notna()]
            b['report_date']=pd.to_datetime(q); b['b_ann']=pd.to_datetime(b['公告日期'],errors='coerce')
            b=b.rename(columns={'资产-货币资金':'cash','资产-应收账款':'recv','资产-存货':'inventory','资产-总资产':'assets','资产-总资产同比':'asset_yoy','负债-应付账款':'ap','负债-预收账款':'advance','负债-总负债':'liab','负债-总负债同比':'liab_yoy','资产负债率':'lev_pct','股东权益合计':'equity'})
            Bs.append(b[['code','report_date','b_ann','cash','recv','inventory','assets','asset_yoy','ap','advance','liab','liab_yoy','lev_pct','equity']])
        if len(inc):
            inc=inc.copy(); inc['code']=inc['股票代码'].map(lambda x:ncode(x,cmap)); inc=inc[inc.code.notna()]
            inc['report_date']=pd.to_datetime(q); inc['i_ann']=pd.to_datetime(inc['公告日期'],errors='coerce')
            inc=inc.rename(columns={'净利润':'net_income','净利润同比':'profit_yoy','营业总收入':'revenue','营业总收入同比':'revenue_yoy','营业总支出-营业支出':'oper_cost','营业总支出-销售费用':'sell_exp','营业总支出-管理费用':'admin_exp','营业总支出-财务费用':'fin_exp','营业总支出-营业总支出':'total_exp','营业利润':'oper_profit','利润总额':'total_profit'})
            Is.append(inc[['code','report_date','i_ann','net_income','profit_yoy','revenue','revenue_yoy','oper_cost','sell_exp','admin_exp','fin_exp','total_exp','oper_profit','total_profit']])
        if len(cf):
            cf=cf.copy(); cf['code']=cf['股票代码'].map(lambda x:ncode(x,cmap)); cf=cf[cf.code.notna()]
            cf['report_date']=pd.to_datetime(q); cf['c_ann']=pd.to_datetime(cf['公告日期'],errors='coerce')
            cf=cf.rename(columns={'净现金流-净现金流':'net_cashflow','净现金流-同比增长':'net_cashflow_yoy','经营性现金流-现金流量净额':'cfo','经营性现金流-净现金流占比':'cfo_share','投资性现金流-现金流量净额':'cfi','投资性现金流-净现金流占比':'cfi_share','融资性现金流-现金流量净额':'cff','融资性现金流-净现金流占比':'cff_share'})
            Cs.append(cf[['code','report_date','c_ann','net_cashflow','net_cashflow_yoy','cfo','cfo_share','cfi','cfi_share','cff','cff_share']])
    pd.DataFrame(audit).to_csv(OUT/'fetch_audit.csv',index=False)
    B=pd.concat(Bs,ignore_index=True) if Bs else pd.DataFrame(); I=pd.concat(Is,ignore_index=True) if Is else pd.DataFrame(); C=pd.concat(Cs,ignore_index=True) if Cs else pd.DataFrame()
    for nm,df in [('balance',B),('income',I),('cashflow',C)]:
        if len(df): df.to_csv(OUT/f'{nm}_raw.csv.gz',index=False,compression='gzip')
    return B,I,C


def safe_div(a,b):
    a=pd.to_numeric(a,errors='coerce'); b=pd.to_numeric(b,errors='coerce')
    return a/b.abs().replace(0,np.nan)

def prepare_events(B,I,C,cal):
    if B.empty and I.empty and C.empty:return pd.DataFrame()
    E=B.merge(I,on=['code','report_date'],how='outer').merge(C,on=['code','report_date'],how='outer')
    for c in E.columns:
        if c not in ('code','report_date','b_ann','i_ann','c_ann'): E[c]=pd.to_numeric(E[c],errors='coerce')
    # Conservative PIT: combined factors become usable only after the last available statement announcement.
    anns=pd.concat([pd.to_datetime(E.get('b_ann'),errors='coerce'),pd.to_datetime(E.get('i_ann'),errors='coerce'),pd.to_datetime(E.get('c_ann'),errors='coerce')],axis=1)
    E['ann_date']=anns.max(axis=1); E=E[E.ann_date.notna()].copy(); E['available_date']=E.ann_date.map(lambda d:fp.next_trade_date(cal,d)); E=E[E.available_date.notna()]
    assets=E.assets.replace(0,np.nan)
    revenue=E.revenue.replace(0,np.nan)
    E['cash_assets']=safe_div(E.cash,assets)
    E['low_leverage']=-safe_div(E.liab,assets)
    E['low_receivables']=-safe_div(E.recv,assets)
    E['low_inventory']=-safe_div(E.inventory,assets)
    E['roa_proxy']=safe_div(E.net_income,assets)
    E['asset_turnover']=safe_div(E.revenue,assets)
    E['net_margin']=safe_div(E.net_income,revenue)
    E['oper_margin']=safe_div(E.oper_profit,revenue)
    E['cfo_assets']=safe_div(E.cfo,assets)
    E['accrual_quality']=safe_div(E.cfo-E.net_income,assets)
    E['cash_conversion']=safe_div(E.cfo,E.net_income).clip(-20,20)
    E['low_finexp']=-safe_div(E.fin_exp,revenue)
    E['balance_strength']=E.cash_assets+E.low_leverage
    # Broad accounting quality blend; z-scored later, raw here is only placeholder components.
    E=E.sort_values(['code','available_date','report_date']).drop_duplicates(['code','available_date'],keep='last')
    return E


def attach(p,E):
    base=p.reset_index().rename(columns={'index':'_row'})[['_row','signal_date','code']].copy(); base['signal_date']=pd.to_datetime(base.signal_date)
    cols=list(FACTORS.keys())[:-1]
    out=[]
    for code,g in base.groupby('code',sort=False):
        e=E[E.code==code].sort_values('available_date')
        if len(e): x=pd.merge_asof(g.sort_values('signal_date'),e[['available_date','report_date']+cols],left_on='signal_date',right_on='available_date',direction='backward')
        else:
            x=g.copy(); x['available_date']=pd.NaT; x['report_date']=pd.NaT
            for c in cols:x[c]=np.nan
        out.append(x)
    A=pd.concat(out,ignore_index=True); A['age_days']=(A.signal_date-pd.to_datetime(A.available_date)).dt.days
    stale=A.age_days>550; A.loc[stale,cols]=np.nan
    return A


def score_panel(p,A):
    z=A.copy(); rawcols=list(FACTORS.keys())[:-1]
    for c in rawcols: z[c+'_z']=z.groupby('signal_date')[c].transform(fp.robust_z)
    z['quality_core_raw']=z[['cash_assets_z','low_leverage_z','roa_proxy_z','net_margin_z','cfo_assets_z','accrual_quality_z']].mean(axis=1,skipna=True)
    for c in FACTORS:
        raw='quality_core_raw' if c=='quality_core' else c+'_z'
        z[c+'_rank']=z.groupby('signal_date')[raw].rank(pct=True,method='average',ascending=False)
    p2=p.reset_index().rename(columns={'index':'_row'}).merge(z[['_row']+[c+'_rank' for c in FACTORS]],on='_row',how='left').drop(columns='_row')
    return p2,z


def ic_table(z,p):
    m=z.merge(p.reset_index().rename(columns={'index':'_row'})[['_row','fwd60']],on='_row',how='left'); rows=[]
    for c in FACTORS:
        col=c+'_rank'
        for split,a,b in [('train',START,TRAIN_END),('pseudo',PSEUDO,END)]:
            zz=m[(m.signal_date>=a)&(m.signal_date<=b)][['signal_date',col,'fwd60']].dropna(); ics=[]; spreads=[]
            for d,g in zz.groupby('signal_date'):
                if len(g)<150:continue
                ic=(-g[col]).corr(g.fwd60,method='spearman')
                if np.isfinite(ic):ics.append(float(ic))
                hi=g[g[col]<=.10].fwd60; lo=g[g[col]>=.90].fwd60
                if len(hi)>10 and len(lo)>10:spreads.append(float(hi.mean()-lo.mean()))
            ar=np.asarray(ics,float); sd=ar.std(ddof=1) if len(ar)>1 else np.nan
            rows.append({'factor':c,'split':split,'dates':len(ar),'mean_ic':ar.mean() if len(ar) else np.nan,'ic_t':ar.mean()/sd*np.sqrt(len(ar)) if len(ar)>1 and sd>0 else np.nan,'positive_ic_share':(ar>0).mean() if len(ar) else np.nan,'spread60':np.mean(spreads) if spreads else np.nan})
    return pd.DataFrame(rows)


def base_q(p): return mega.make_rank(p,{'name':'base_mom','kind':'gate','g':{'ef':.55},'w':BASE_W})
def blend_q(p,c,w):
    q=base_q(p); f=q[c+'_rank'].fillna(.5); m=np.isfinite(q.rank_test); raw=(1-w)*q.loc[m,'rank_test']+w*f.loc[m]; q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True); return q

def subset(q,h,ph):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[ph::max(1,round(h/5))]); cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]; s=q[q.signal_date.isin(chosen)][cols].copy(); s['ivol60_pct']=s.rank_test; return s.drop(columns='rank_test')
def pcagr(eq,a,b): return fp.period_cagr(eq,a,b)

def portfolio(p,cal,members,bm,ics):
    tr=ics[ics.split=='train'].copy(); tr=tr[(tr.mean_ic>0)&(tr.ic_t>=2)].sort_values('ic_t',ascending=False); selected=list(tr.factor.head(4)); tests=[('base_mom','',0.0)]+[(f,f,w) for f in selected for w in BLEND]
    rows=[]
    for label,f,w in tests:
        q=base_q(p) if not f else blend_q(p,f,w)
        for h in HORIZONS:
            for ph in range(max(1,round(h/5))):
                st,eq,trd,tm=ma.run_panel(subset(q,h,ph),cal,members,bm,n=N,entry=ENTRY,keep=KEEP); st.update(candidate=label if not f else f'{f}_w{w:.2f}',fund_factor=f,fund_weight=w,H=h,phase=ph,half1_cagr=pcagr(eq,START,pd.Timestamp('2019-12-31')),half2_cagr=pcagr(eq,pd.Timestamp('2020-01-01'),TRAIN_END),pseudo_cagr=pcagr(eq,PSEUDO,END)); rows.append(st)
    d=pd.DataFrame(rows); d.to_csv(OUT/'portfolio_all_phase.csv',index=False); out=[]
    for c,g in d.groupby('candidate'):
        out.append({'candidate':c,'runs':len(g),'full_cagr_median':g.cagr.median(),'full_cagr_p25':g.cagr.quantile(.25),'full_cagr_min':g.cagr.min(),'mdd_median':g.max_drawdown.median(),'mdd_worst':g.max_drawdown.min(),'sharpe_median':g.sharpe.median(),'train_maximin':min(g.half1_cagr.median(),g.half2_cagr.median()),'pseudo_cagr_median':g.pseudo_cagr.median(),'pseudo_cagr_p25':g.pseudo_cagr.quantile(.25),'h60_median':g[g.H==60].cagr.median(),'h75_median':g[g.H==75].cagr.median(),'h90_median':g[g.H==90].cagr.median()})
    s=pd.DataFrame(out).sort_values(['train_maximin','full_cagr_p25'],ascending=False); s.to_csv(OUT/'portfolio_summary.csv',index=False); return d,s,selected


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=True); p=strict.attach_gap_flags(p,cal,'board')
    B,I,C=fetch_tables(sorted(p.code.unique())); E=prepare_events(B,I,C,cal); E.to_csv(OUT/'fundamental_events.csv.gz',index=False,compression='gzip'); A=attach(p,E); A.to_csv(OUT/'pit_attached.csv.gz',index=False,compression='gzip'); p2,z=score_panel(p,A)
    cov=pd.DataFrame([{'factor':c,'coverage':z[c].notna().mean() if c in z else z.quality_core_raw.notna().mean()} for c in list(FACTORS.keys())[:-1]]+ [{'factor':'quality_core','coverage':z.quality_core_raw.notna().mean()}]); cov.to_csv(OUT/'coverage.csv',index=False)
    ics=ic_table(z,p); ics.to_csv(OUT/'fundamental_ic.csv',index=False); d,s,selected=portfolio(p2,cal,members,bm,ics)
    control=d[(d.candidate=='base_mom')&(d.H==90)&(d.phase==0)].iloc[0]; control_pass=bool(abs(float(control.cagr)-0.2229800205653422)<1e-10)
    meta={'status':'POST_SOURCE_DISCOVERY_PREDECLARED_3STMT_TEST','control_h90_phase0_cagr':float(control.cagr),'control_pass':control_pass,'sources':['AKShare/Eastmoney RPT_DMSK_FN_BALANCE','RPT_DMSK_FN_INCOME','RPT_DMSK_FN_CASHFLOW'],'pit_rule':'combined accounting metrics usable on next exchange day after latest statement announcement','selected_by_train_ic_only':selected,'horizons':HORIZONS,'N':N,'market_factor':market_code,'universe_audit':ua}; (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
    print('=== COVERAGE ===\n',cov.to_string(index=False)); print('=== IC ===\n',ics.to_string(index=False)); print('=== PORTFOLIO ===\n',s.to_string(index=False)); print('=== META ===\n',json.dumps(meta,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__': main()
