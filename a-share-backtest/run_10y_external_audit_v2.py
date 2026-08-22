from __future__ import annotations
from pathlib import Path
from urllib.parse import quote, unquote
import io, re
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import statsmodels.api as sm
import baostock as bs

import run_10y_external_audit as old
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_grand_opt as grand
import run_10y_balanced_exact as be
import run_10y_max_audit as ma

OUT=Path('results_external_audit_v2'); OUT.mkdir(exist_ok=True)
old.OUT=OUT


def normalize_status(z):
    if z.empty: return z
    cols={c.lower():c for c in z.columns}
    cc=cols.get('code','code'); nc=cols.get('code_name',cols.get('codename','code_name')); tc=cols.get('tradestatus',cols.get('trade_status','tradeStatus'))
    out=pd.DataFrame({'date':pd.to_datetime(z['date']),'bs_code':z[cc].astype(str).str.lower()})
    out['code']=out.bs_code.map(old.qcode)
    out['code_name']=z[nc].astype(str) if nc in z else ''
    out['trade_status']=pd.to_numeric(z[tc],errors='coerce') if tc in z else np.nan
    out['risk_name']=out.code_name.str.upper().str.contains(r'(?:^|\*)ST|退',regex=True)
    return out


def true_coverage(p,status,selected):
    rows=[]
    for d in selected:
        d=pd.Timestamp(d)
        qset=set(p.loc[p.signal_date==d,'code'].astype(str))
        bset=set(status.loc[status.date==d,'code'].astype(str))
        inter=qset & bset
        rows.append({'date':d,'coverage':len(inter)/max(1,len(qset)),'intersection_n':len(inter),'baostock_n':len(bset),'qlib_n':len(qset)})
    z=pd.DataFrame(rows); z.to_csv(OUT/'baostock_universe_coverage.csv',index=False); return z


def pit_rerank_v2(p,status,selected_signal_dates):
    q=p[p.signal_date.isin(selected_signal_dates)].copy()
    sig=status.rename(columns={'date':'signal_date'})[['signal_date','code','code_name','trade_status','risk_name']]
    q=q.merge(sig,on=['signal_date','code'],how='left')
    q['status_known']=q.code_name.notna()
    q['blocked_signal']=(~q.status_known)|q.risk_name.fillna(True)|(q.trade_status.fillna(0)!=1)
    q['rank_test']=np.nan
    valid=(~q.blocked_signal)&np.isfinite(q.ivol60)&np.isfinite(q.eff120)&np.isfinite(q.skew40)&np.isfinite(q.liq20)
    liq=q.loc[valid].groupby('signal_date').liq20.rank(pct=True,method='average',ascending=False)
    ok=pd.Series(False,index=q.index); ok.loc[liq.index]=liq<=.70; valid &= ok
    sp=q.loc[valid].groupby('signal_date').skew40.rank(pct=True,method='average',ascending=True)
    ok2=pd.Series(False,index=q.index); ok2.loc[sp.index]=sp<=.80; valid &= ok2
    iv=q.loc[valid].groupby('signal_date').ivol60.rank(pct=True,method='average',ascending=True)
    ef=q.loc[valid].groupby('signal_date').eff120.rank(pct=True,method='average',ascending=False)
    raw=.60*iv+.40*ef
    q.loc[valid,'rank_test']=raw.groupby(q.loc[valid,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q


def audit_trade_risk(tm,status):
    buys=tm[tm.side=='buy'].copy()
    sig=status.rename(columns={'date':'signal_date'})[['signal_date','code','code_name','risk_name','trade_status']].rename(columns={'code_name':'sig_name','risk_name':'sig_risk','trade_status':'sig_status'})
    trd=status.rename(columns={'date':'trade_date'})[['trade_date','code','code_name','risk_name','trade_status']].rename(columns={'code_name':'trade_name','risk_name':'trade_risk','trade_status':'trade_status_bs'})
    x=buys.merge(sig,on=['signal_date','code'],how='left').merge(trd,on=['trade_date','code'],how='left')
    x.to_csv(OUT/'final_buys_pit_status.csv',index=False)
    sig_bad=int(x.sig_risk.fillna(True).sum())
    trade_bad=int(x.trade_risk.fillna(True).sum())
    unknown=int((x.sig_name.isna()|x.trade_name.isna()).sum())
    halted=int((x.trade_status_bs.fillna(0)!=1).sum())
    return sig_bad,trade_bad,unknown,halted


def pit_st_audit_v2(p,q,cal,members,bm,base_tm):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
    selected=pd.DatetimeIndex(dates[::12])
    trade_map=q[q.signal_date.isin(selected)][['signal_date','trade_date']].drop_duplicates()
    external_dates=list(selected)+list(pd.to_datetime(trade_map.trade_date))
    raw=old.all_stock_dates(external_dates); raw.to_csv(OUT/'baostock_all_stock_raw.csv',index=False)
    status=normalize_status(raw); status.to_csv(OUT/'baostock_status.csv',index=False)
    cov=true_coverage(p,status,selected)
    qp=pit_rerank_v2(p,status,selected)
    tr=status.rename(columns={'date':'trade_date'})[['trade_date','code','code_name','trade_status','risk_name']].rename(columns={'code_name':'exec_bs_name','trade_status':'exec_bs_status','risk_name':'exec_bs_risk'})
    qp=qp.merge(tr,on=['trade_date','code'],how='left')
    # Fail closed for NEW buys when next-open status is unknown or ST/*ST/retirement-risk.
    # Keep quotes for tradable ST names so an existing position may still be sold.
    trade_buy_block=qp.exec_bs_name.isna()|qp.exec_bs_risk.fillna(True)
    qp.loc[trade_buy_block,'rank_test']=np.nan
    # A suspended/non-trading security cannot be bought or sold at the open.
    nontrade=qp.exec_bs_status.fillna(0)!=1
    qp.loc[nontrade,['exec_open','exec_high','exec_low','exec_volume']]=np.nan
    z=qp[['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy()
    z=z[np.isfinite(z[['exec_open','exec_high','exec_low','exec_volume']]).all(axis=1)]
    z['ivol60_pct']=z.rank_test; z=z.drop(columns='rank_test')
    st,eq,trades,tm=ma.run_panel(z,cal,members,bm)
    pd.DataFrame([st]).to_csv(OUT/'pit_st_summary.csv',index=False); sim.annual_returns(eq).to_csv(OUT/'pit_st_annual.csv',index=False)
    # Audit original baseline buys to measure the defect we corrected.
    b=base_tm[base_tm.side=='buy'].copy()
    sig=status.rename(columns={'date':'signal_date'})[['signal_date','code','code_name','risk_name','trade_status']]
    trd=status.rename(columns={'date':'trade_date'})[['trade_date','code','code_name','risk_name','trade_status']].rename(columns={'code_name':'trade_code_name','risk_name':'trade_risk_name','trade_status':'trade_status_bs'})
    b=b.merge(sig,on=['signal_date','code'],how='left').merge(trd,on=['trade_date','code'],how='left')
    b.to_csv(OUT/'baseline_buys_pit_status.csv',index=False)
    held_sig_st=int(b.risk_name.fillna(False).sum())
    held_trade_st=int(b.trade_risk_name.fillna(False).sum())
    unknown=int((b.code_name.isna()|b.trade_code_name.isna()).sum())
    final_sig_bad,final_trade_bad,final_unknown,final_halted=audit_trade_risk(tm,status)
    return st,eq,trades,tm,cov,status,held_sig_st,held_trade_st,unknown,final_sig_bad,final_trade_bad,final_unknown,final_halted


def price_crosscheck_v2(p,tm):
    t=tm.copy()
    t=t.merge(p[['signal_date','trade_date','code','exec_open','exec_volume','exec_factor','liq20']].drop_duplicates(['signal_date','trade_date','code']),on=['signal_date','trade_date','code'],how='left')
    rows=[]; hists={}; codes=sorted(t.code.unique())
    for i,c in enumerate(codes,1):
        z=old.query_history(c)
        if len(z): hists[c]=z
        if i%25==0: print('BAOSTOCK HIST',i,'/',len(codes),flush=True)
    for r in t.itertuples(index=False):
        z=hists.get(r.code)
        if z is None: continue
        m=z[z.date==pd.Timestamp(r.trade_date)]
        if m.empty: continue
        b=m.iloc[0]; fac=float(r.exec_factor) if np.isfinite(r.exec_factor) and r.exec_factor>0 else 1.
        qraw=float(r.exec_open)/fac
        # Qlib CN volume is in 100-share units after undoing the adjustment factor.
        qshares=float(r.exec_volume)*fac*100.0
        rows.append({'signal_date':r.signal_date,'trade_date':r.trade_date,'code':r.code,'side':r.side,'qlib_raw_open':qraw,'bs_raw_open':b.open,'open_rel_err':abs(qraw-b.open)/b.open if b.open>0 else np.nan,'qlib_raw_volume_shares':qshares,'bs_volume_shares':b.volume,'volume_ratio':qshares/b.volume if b.volume>0 else np.nan,'volume_rel_err':abs(qshares-b.volume)/b.volume if b.volume>0 else np.nan,'bs_amount':b.amount,'bs_isST':b.isST,'bs_tradestatus':b.tradestatus})
    x=pd.DataFrame(rows); x.to_csv(OUT/'cross_source_trade_prices.csv',index=False)
    if x.empty: return {'rows':0,'coverage':0.0}
    return {'rows':len(x),'coverage':len(x)/max(1,len(t)),'median_open_rel_err':float(x.open_rel_err.median()),'p95_open_rel_err':float(x.open_rel_err.quantile(.95)),'p99_open_rel_err':float(x.open_rel_err.quantile(.99)),'median_volume_ratio':float(x.volume_ratio.replace([np.inf,-np.inf],np.nan).median()),'p95_volume_rel_err':float(x.volume_rel_err.quantile(.95)),'st_trade_rows':int((x.bs_isST==1).sum()),'halt_trade_rows':int((x.bs_tradestatus!=1).sum())}


def factor_links_v2():
    url='https://www.factorwar.com/data/factor-models/'
    r=requests.get(url,timeout=60,headers={'User-Agent':'Mozilla/5.0'}); r.raise_for_status()
    soup=BeautifulSoup(r.text,'lxml')
    targets=[('ff3','Fama-French 三因子'),('carhart4','Carhart 四因子'),('ff5','Fama-French 五因子'),('betaplus4','BetaPlus A 股混合四因子')]
    out={}
    headings=soup.find_all(re.compile('^h[1-6]$'))
    for h in headings:
        title=re.sub(r'\s+',' ',h.get_text(' ',strip=True))
        key=next((k for k,s in targets if s.replace(' ','').lower() in title.replace(' ','').lower()),None)
        if not key: continue
        cur=h.next_sibling; daily=[]
        while cur is not None:
            name=getattr(cur,'name',None)
            if name and re.match(r'^h[1-6]$',name): break
            if hasattr(cur,'find_all'):
                for a in cur.find_all('a',href=True):
                    if a.get_text(' ',strip=True).lower()=='daily': daily.append(a['href'])
            cur=cur.next_sibling
        if daily: out[key]=daily[0]  # first daily link is the classic construction
    # Fallback to the older URL-name parser if site markup changes.
    fallback=old.factor_links()
    for k,v in fallback.items(): out.setdefault(k,v)
    return out


def read_factor_v2(url):
    # Normalize any Chinese/unescaped path into a legal percent-encoded URL.
    clean=quote(unquote(str(url)),safe=':/?=&')
    r=requests.get(clean,timeout=90,headers={'User-Agent':'Mozilla/5.0','Accept':'text/csv,text/plain,*/*'})
    r.raise_for_status(); content=r.content
    if not content or content.lstrip().lower().startswith((b'<html',b'<!doctype')): raise RuntimeError('factor response is HTML, not CSV')
    last=None
    for enc in ('utf-8-sig','gb18030','gbk','utf-8'):
        try:
            z=pd.read_csv(io.BytesIO(content),encoding=enc)
            if len(z)>=100 and z.shape[1]>=2: return z
        except Exception as e: last=e
    raise RuntimeError(f'factor csv unreadable: {last!r}')


def parse_factor_date(s):
    x=s.astype(str).str.strip().str.replace(r'\.0$','',regex=True)
    out=pd.to_datetime(x,errors='coerce')
    m=x.str.fullmatch(r'\d{8}')
    if m.any(): out.loc[m]=pd.to_datetime(x.loc[m],format='%Y%m%d',errors='coerce')
    return out


def factor_regress_v2(eq):
    ret=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index().pct_change().dropna()
    links=factor_links_v2(); rows=[]
    for name,url in links.items():
        try:
            f=read_factor_v2(url); f.to_csv(OUT/f'factorwar_{name}.csv',index=False)
            dc=f.columns[0]; f[dc]=parse_factor_date(f[dc]); f=f.dropna(subset=[dc]).set_index(dc)
            num=f.apply(pd.to_numeric,errors='coerce').dropna(axis=1,how='all')
            if num.empty: raise RuntimeError('no numeric factor columns')
            # Percent-vs-decimal unit detection: decimal daily returns rarely have a cross-factor 95th percentile >15%.
            q95=float(num.abs().quantile(.95).median())
            if q95>.15: num=num/100.0
            def norm(c): return re.sub(r'[^a-z0-9\u4e00-\u9fff]','',str(c).lower())
            rfcol=next((c for c in num.columns if norm(c) in ('rf','riskfree','riskfreerate') or '无风险' in norm(c)),None)
            rf=num[rfcol] if rfcol is not None else pd.Series(0.,index=num.index)
            X=num.drop(columns=[rfcol] if rfcol is not None else [],errors='ignore').copy()
            mcol=next((c for c in X.columns if ('mkt' in norm(c) or 'market' in norm(c) or '市场' in norm(c) or norm(c) in ('rm','rmarket'))),None)
            if mcol is not None and 'mktrf' not in norm(mcol): X[mcol]=X[mcol]-rf.reindex(X.index).fillna(0)
            y=ret.reindex(X.index)-rf.reindex(X.index).fillna(0)
            d=pd.concat([y.rename('y'),X],axis=1).dropna()
            if len(d)<500: raise RuntimeError(f'too few aligned rows {len(d)}')
            xx=sm.add_constant(d.drop(columns='y'))
            m20=sm.OLS(d.y,xx).fit(cov_type='HAC',cov_kwds={'maxlags':20})
            m60=sm.OLS(d.y,xx).fit(cov_type='HAC',cov_kwds={'maxlags':60})
            rows.append({'model':name,'status':'OK','n':len(d),'alpha_ann':float(m60.params['const']*252),'alpha_t_hac20':float(m20.tvalues['const']),'alpha_t_hac60':float(m60.tvalues['const']),'r2':float(m60.rsquared),'factors':'|'.join(map(str,d.drop(columns='y').columns)),'url':url,'scaled_from_percent':int(q95>.15)})
        except Exception as e:
            rows.append({'model':name,'status':'FAIL','error':repr(e),'url':url})
    z=pd.DataFrame(rows); z.to_csv(OUT/'external_factor_regression.csv',index=False); return z,links


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members)
    bm=market_close.loc[sim.START:sim.END].dropna(); q=be.anchor_weighted(p,'liq70',.60)
    bst,beq,btr,btm=ma.run_q(q,60,0,cal,members,bm); pd.DataFrame([bst]).to_csv(OUT/'baseline.csv',index=False)
    lg=bs.login(); print('BAOSTOCK LOGIN',lg.error_code,lg.error_msg,flush=True)
    if lg.error_code!='0': raise RuntimeError('baostock login failed')
    try:
        pst,peq,ptr,ptm,cov,status,base_sig_st,base_trade_st,base_unknown,final_sig_bad,final_trade_bad,final_unknown,final_halted=pit_st_audit_v2(p,q,cal,members,bm,btm)
        cross=price_crosscheck_v2(p,ptm); pd.DataFrame([cross]).to_csv(OUT/'cross_source_summary.csv',index=False)
    finally:
        bs.logout()
    freg,links=factor_regress_v2(peq); okreg=freg[freg.status=='OK'] if len(freg) else pd.DataFrame(); mincov=float(cov.coverage.min()) if len(cov) else 0.
    alpha_pos=bool(len(okreg)>=1 and (okreg.alpha_ann>0).all())
    alpha_sig=bool(len(okreg)>=1 and (okreg.alpha_t_hac60>1.96).any()) if 'alpha_t_hac60' in okreg else False
    gates={
      'baostock_qlib_intersection_coverage_ge_95pct':int(mincov>=.95),
      'baseline_defect_detected_or_zero':1,
      'corrected_strategy_signal_st_buys_zero':int(final_sig_bad==0),
      'corrected_strategy_trade_day_st_buys_zero':int(final_trade_bad==0),
      'corrected_strategy_unknown_buy_status_zero':int(final_unknown==0),
      'corrected_strategy_halted_buys_zero':int(final_halted==0),
      'pit_st_rerun_positive':int(pst['total_return']>0),
      'pit_st_rerun_cagr_ge_8pct':int(pst['cagr']>=.08),
      'cross_source_trade_coverage_ge_95pct':int(cross.get('coverage',0)>=.95),
      'raw_open_median_error_le_10bp':int(cross.get('median_open_rel_err',np.inf)<=.001),
      'raw_open_p95_error_le_50bp':int(cross.get('p95_open_rel_err',np.inf)<=.005),
      'raw_volume_median_ratio_0_99_to_1_01':int(.99<=cross.get('median_volume_ratio',np.nan)<=1.01),
      'raw_volume_p95_error_le_1pct':int(cross.get('p95_volume_rel_err',np.inf)<=.01),
      'external_factor_at_least_2_models':int(len(okreg)>=2),
      'external_alpha_positive_all_available':int(alpha_pos),
      'external_alpha_significant_any_hac60':int(alpha_sig),
    }
    gd=pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]); gd.to_csv(OUT/'external_gates.csv',index=False)
    verdict={**ua,'market_factor':market_code,'baostock_min_true_coverage':mincov,'baseline_signal_st_buys':base_sig_st,'baseline_trade_day_st_buys':base_trade_st,'baseline_unknown_buy_status':base_unknown,'corrected_signal_st_buys':final_sig_bad,'corrected_trade_day_st_buys':final_trade_bad,'corrected_unknown_buy_status':final_unknown,'corrected_halted_buys':final_halted,'pit_st_total_return':pst['total_return'],'pit_st_cagr':pst['cagr'],'pit_st_mdd':pst['max_drawdown'],'pit_st_sharpe':pst['sharpe'],**cross,'external_factor_models_ok':len(okreg),'external_hard_pass':int(gd['pass'].all()),'gates_passed':int(gd['pass'].sum()),'gates_total':len(gd),'factor_links':'|'.join(links.keys())}
    pd.DataFrame([verdict]).to_csv(OUT/'external_verdict.csv',index=False)
    print('=== EXTERNAL V2 VERDICT ==='); print(pd.DataFrame([verdict]).to_string(index=False),flush=True)
    print('=== GATES ==='); print(gd.to_string(index=False),flush=True)
    print('=== FACTOR REGRESSION ==='); print(freg.to_string(index=False),flush=True)

if __name__=='__main__': main()
