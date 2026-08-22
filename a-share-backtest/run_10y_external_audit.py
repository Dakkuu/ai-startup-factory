from __future__ import annotations
from pathlib import Path
from urllib.parse import unquote
import re
import numpy as np, pandas as pd
import requests
from bs4 import BeautifulSoup
import statsmodels.api as sm
import baostock as bs

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_skewfilter_hard as hard
import run_10y_grand_opt as grand
import run_10y_balanced_exact as be
import run_10y_max_audit as ma

OUT=Path('results_external_audit'); OUT.mkdir(exist_ok=True)

def bscode(c):
    c=str(c).upper()
    if c.startswith('SH'): return 'sh.'+c[2:]
    if c.startswith('SZ'): return 'sz.'+c[2:]
    if c.startswith('BJ'): return 'bj.'+c[2:]
    return c.lower()

def qcode(c):
    s=str(c).lower()
    if s.startswith('sh.'): return 'SH'+s[3:]
    if s.startswith('sz.'): return 'SZ'+s[3:]
    if s.startswith('bj.'): return 'BJ'+s[3:]
    return s.upper()

def rs_frame(rs):
    data=[]
    while rs.error_code=='0' and rs.next(): data.append(rs.get_row_data())
    return pd.DataFrame(data,columns=rs.fields)

def all_stock_dates(dates):
    rows=[]; uniq=sorted(set(pd.to_datetime(dates)))
    for i,d in enumerate(uniq,1):
        rs=bs.query_all_stock(day=pd.Timestamp(d).strftime('%Y-%m-%d')); z=rs_frame(rs)
        if len(z): z['date']=pd.Timestamp(d); rows.append(z)
        print('BAOSTOCK ALL',i,'/',len(uniq),d.date(),'rows',len(z),flush=True)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

def normalize_status(z):
    if z.empty: return z
    cols={c.lower():c for c in z.columns}; cc=cols.get('code','code'); nc=cols.get('code_name',cols.get('codename','code_name')); tc=cols.get('tradestatus',cols.get('trade_status','tradeStatus'))
    out=pd.DataFrame({'date':pd.to_datetime(z['date']),'bs_code':z[cc].astype(str).str.lower()}); out['code']=out.bs_code.map(qcode); out['code_name']=z[nc].astype(str) if nc in z else ''; out['trade_status']=pd.to_numeric(z[tc],errors='coerce') if tc in z else np.nan; out['risk_name']=out.code_name.str.upper().str.contains(r'(^|\*)ST|退',regex=True)
    return out

def pit_rerank(p,status,selected_signal_dates):
    q=p[p.signal_date.isin(selected_signal_dates)].copy(); s=status.rename(columns={'date':'signal_date'})[['signal_date','code','code_name','trade_status','risk_name']]
    q=q.merge(s,on=['signal_date','code'],how='left'); q['status_known']=q.code_name.notna(); q['blocked']=q.risk_name.fillna(True)|(q.trade_status.fillna(0)!=1); q['rank_test']=np.nan
    valid=(~q.blocked)&np.isfinite(q.ivol60)&np.isfinite(q.eff120)&np.isfinite(q.skew40)&np.isfinite(q.liq20)
    liq=q.loc[valid].groupby('signal_date').liq20.rank(pct=True,method='average',ascending=False); ok=pd.Series(False,index=q.index); ok.loc[liq.index]=liq<=.70; valid&=ok
    sp=q.loc[valid].groupby('signal_date').skew40.rank(pct=True,method='average',ascending=True); ok2=pd.Series(False,index=q.index); ok2.loc[sp.index]=sp<=.80; valid&=ok2
    iv=q.loc[valid].groupby('signal_date').ivol60.rank(pct=True,method='average',ascending=True); ef=q.loc[valid].groupby('signal_date').eff120.rank(pct=True,method='average',ascending=False); raw=.60*iv+.40*ef; q.loc[valid,'rank_test']=raw.groupby(q.loc[valid,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q

def pit_st_audit(p,q,cal,members,bm,base_tm):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); selected=pd.DatetimeIndex(dates[::12]); trade_map=q[q.signal_date.isin(selected)][['signal_date','trade_date']].drop_duplicates(); external_dates=list(selected)+list(pd.to_datetime(trade_map.trade_date))
    raw=all_stock_dates(external_dates); raw.to_csv(OUT/'baostock_all_stock_raw.csv',index=False); status=normalize_status(raw); status.to_csv(OUT/'baostock_status.csv',index=False)
    coverage=status[status.date.isin(selected)].groupby('date').code.nunique(); universe=p[p.signal_date.isin(selected)].groupby('signal_date').code.nunique(); cov=[]
    for d,n in universe.items(): cov.append({'date':d,'coverage':float(coverage.get(pd.Timestamp(d),0)/n),'baostock_n':int(coverage.get(pd.Timestamp(d),0)),'qlib_n':int(n)})
    cov=pd.DataFrame(cov); cov.to_csv(OUT/'baostock_universe_coverage.csv',index=False)
    qp=pit_rerank(p,status,selected); ts=status.rename(columns={'date':'trade_date'})[['trade_date','code','trade_status']].rename(columns={'trade_status':'exec_bs_status'}); qp=qp.merge(ts,on=['trade_date','code'],how='left'); qp.loc[qp.exec_bs_status.fillna(0)!=1,'exec_open']=np.nan
    z=qp[['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy(); z=z[np.isfinite(z[['exec_open','exec_high','exec_low','exec_volume']]).all(axis=1)]; z['ivol60_pct']=z.rank_test; z=z.drop(columns='rank_test')
    st,eq,tr,tm=ma.run_panel(z,cal,members,bm); pd.DataFrame([st]).to_csv(OUT/'pit_st_summary.csv',index=False); sim.annual_returns(eq).to_csv(OUT/'pit_st_annual.csv',index=False)
    buys=base_tm[base_tm.side=='buy'].copy(); sigstat=status.rename(columns={'date':'signal_date'})[['signal_date','code','code_name','risk_name','trade_status']]; buys=buys.merge(sigstat,on=['signal_date','code'],how='left'); buys.to_csv(OUT/'baseline_buys_pit_status.csv',index=False); held_st=int(buys.risk_name.fillna(False).sum()); unknown=int(buys.code_name.isna().sum())
    return st,eq,tr,tm,cov,held_st,unknown,status

def query_history(code,start='2016-07-01',end='2026-07-31',adjust='3'):
    fields='date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST'; rs=bs.query_history_k_data_plus(bscode(code),fields,start_date=start,end_date=end,frequency='d',adjustflag=adjust); z=rs_frame(rs)
    if z.empty: return z
    z['date']=pd.to_datetime(z.date); z['code']=code
    for c in ['open','high','low','close','preclose','volume','amount','turn','tradestatus','pctChg','isST']: z[c]=pd.to_numeric(z[c],errors='coerce')
    return z

def price_crosscheck(p,base_tm):
    t=base_tm.copy(); t=t.merge(p[['signal_date','trade_date','code','exec_open','exec_volume','exec_factor','liq20']].drop_duplicates(['signal_date','trade_date','code']),on=['signal_date','trade_date','code'],how='left'); rows=[]; hists={}; codes=sorted(t.code.unique())
    for i,c in enumerate(codes,1):
        z=query_history(c)
        if len(z): hists[c]=z
        if i%25==0: print('BAOSTOCK HIST',i,'/',len(codes),flush=True)
    for r in t.itertuples(index=False):
        z=hists.get(r.code)
        if z is None: continue
        m=z[z.date==pd.Timestamp(r.trade_date)]
        if m.empty: continue
        b=m.iloc[0]; fac=float(r.exec_factor) if np.isfinite(r.exec_factor) and r.exec_factor>0 else 1.; qraw=float(r.exec_open)/fac; qvol=float(r.exec_volume)*fac
        rows.append({'signal_date':r.signal_date,'trade_date':r.trade_date,'code':r.code,'side':r.side,'qlib_raw_open':qraw,'bs_raw_open':b.open,'open_rel_err':abs(qraw-b.open)/b.open if b.open>0 else np.nan,'qlib_raw_volume':qvol,'bs_volume':b.volume,'volume_ratio':qvol/b.volume if b.volume>0 else np.nan,'bs_amount':b.amount,'bs_isST':b.isST,'bs_tradestatus':b.tradestatus})
    x=pd.DataFrame(rows); x.to_csv(OUT/'cross_source_trade_prices.csv',index=False)
    if x.empty: return {'rows':0,'coverage':0,'median_open_rel_err':np.nan,'p95_open_rel_err':np.nan,'st_trade_rows':np.nan}
    return {'rows':len(x),'coverage':len(x)/max(1,len(t)),'median_open_rel_err':float(x.open_rel_err.median()),'p95_open_rel_err':float(x.open_rel_err.quantile(.95)),'p99_open_rel_err':float(x.open_rel_err.quantile(.99)),'median_volume_ratio':float(x.volume_ratio.replace([np.inf,-np.inf],np.nan).median()),'st_trade_rows':int((x.bs_isST==1).sum()),'halt_trade_rows':int((x.bs_tradestatus!=1).sum())}

def factor_links():
    url='https://www.factorwar.com/data/factor-models/'; html=requests.get(url,timeout=30).text; soup=BeautifulSoup(html,'lxml'); out={}; targets={'ff3':'Fama-French三因子','carhart4':'Carhart','ff5':'Fama-French五因子','betaplus4':'BetaPlusA股混合四因子'}
    for a in soup.find_all('a',href=True):
        u=unquote(a['href'])
        if '日收益率' not in u or '经典算法' not in u or not u.lower().endswith('.csv'): continue
        compact=re.sub(r'\s+','',u)
        for k,s in targets.items():
            if s.replace(' ','') in compact and k not in out: out[k]=a['href']
    return out

def read_factor(url):
    for enc in (None,'utf-8-sig','gbk'):
        try: return pd.read_csv(url,encoding=enc) if enc else pd.read_csv(url)
        except Exception: pass
    raise RuntimeError('factor csv unreadable')

def factor_regress(eq):
    ret=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index().pct_change().dropna(); links=factor_links(); rows=[]
    for name,url in links.items():
        try:
            f=read_factor(url); f.to_csv(OUT/f'factorwar_{name}.csv',index=False); dc=f.columns[0]; f[dc]=pd.to_datetime(f[dc].astype(str),errors='coerce'); f=f.dropna(subset=[dc]).set_index(dc); num=f.apply(pd.to_numeric,errors='coerce'); med=float(num.abs().stack().median()) if num.notna().any().any() else 0
            if med>.20: num=num/100.0
            rfcol=next((c for c in num.columns if str(c).lower() in ('rf','riskfree') or '无风险' in str(c)),None); rf=num[rfcol] if rfcol else pd.Series(0.,index=num.index); X=num.drop(columns=[rfcol] if rfcol else [],errors='ignore').copy(); mcol=next((c for c in X.columns if 'mkt' in str(c).lower() or '市场' in str(c)),None)
            if mcol is not None: X[mcol]=X[mcol]-rf.reindex(X.index).fillna(0)
            y=ret.reindex(X.index)-rf.reindex(X.index).fillna(0); d=pd.concat([y.rename('y'),X],axis=1).dropna()
            if len(d)<500: raise RuntimeError(f'too few rows {len(d)}')
            model=sm.OLS(d.y,sm.add_constant(d.drop(columns='y'))).fit(cov_type='HAC',cov_kwds={'maxlags':5}); rows.append({'model':name,'status':'OK','n':len(d),'alpha_ann':float(model.params['const']*252),'alpha_t_hac':float(model.tvalues['const']),'r2':float(model.rsquared),'factors':'|'.join(map(str,d.drop(columns='y').columns)),'url':url})
        except Exception as e: rows.append({'model':name,'status':'FAIL','error':repr(e),'url':url})
    z=pd.DataFrame(rows); z.to_csv(OUT/'external_factor_regression.csv',index=False); return z,links

def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal); p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members); bm=market_close.loc[sim.START:sim.END].dropna(); q=be.anchor_weighted(p,'liq70',.60)
    bst,beq,btr,btm=ma.run_q(q,60,0,cal,members,bm); pd.DataFrame([bst]).to_csv(OUT/'baseline.csv',index=False)
    lg=bs.login(); login_ok=(lg.error_code=='0'); print('BAOSTOCK LOGIN',lg.error_code,lg.error_msg,flush=True)
    if not login_ok: raise RuntimeError('baostock login failed')
    try:
        pst,peq,ptr,ptm,cov,held_st,unknown,status=pit_st_audit(p,q,cal,members,bm,btm); cross=price_crosscheck(p,btm); pd.DataFrame([cross]).to_csv(OUT/'cross_source_summary.csv',index=False)
    finally: bs.logout()
    freg,links=factor_regress(beq); mincov=float(cov.coverage.min()) if len(cov) else 0.; okreg=freg[freg.status=='OK'] if len(freg) else pd.DataFrame()
    gates={'baostock_signal_universe_coverage_ge_95pct':int(mincov>=.95),'baseline_buys_no_pit_st':int(held_st==0),'baseline_buy_status_unknown_le_1pct':int(unknown<=max(1,int(.01*max(1,len(btm[btm.side=='buy']))))),'pit_st_rerun_positive':int(pst['total_return']>0),'pit_st_rerun_cagr_ge_8pct':int(pst['cagr']>=.08),'cross_source_trade_coverage_ge_95pct':int(cross['coverage']>=.95),'raw_open_median_error_le_10bp':int(cross['median_open_rel_err']<=.001),'raw_open_p95_error_le_50bp':int(cross['p95_open_rel_err']<=.005),'external_factor_at_least_2_models':int(len(okreg)>=2),'external_alpha_positive_all_available':int(len(okreg)>=1 and (okreg.alpha_ann>0).all()),'external_alpha_significant_any':int(len(okreg)>=1 and (okreg.alpha_t_hac>1.96).any())}
    gd=pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]); gd.to_csv(OUT/'external_gates.csv',index=False); verdict={**ua,'market_factor':market_code,'baostock_min_universe_coverage':mincov,'held_st_buys':held_st,'unknown_buy_status':unknown,'pit_st_total_return':pst['total_return'],'pit_st_cagr':pst['cagr'],**cross,'external_factor_models_ok':len(okreg),'external_hard_pass':int(gd['pass'].all()),'gates_passed':int(gd['pass'].sum()),'gates_total':len(gd),'factor_links':'|'.join(links.keys())}
    pd.DataFrame([verdict]).to_csv(OUT/'external_verdict.csv',index=False); print('=== EXTERNAL VERDICT ==='); print(pd.DataFrame([verdict]).to_string(index=False),flush=True); print('=== GATES ==='); print(gd.to_string(index=False),flush=True); print('=== FACTOR REGRESSION ==='); print(freg.to_string(index=False),flush=True)

if __name__=='__main__': main()
