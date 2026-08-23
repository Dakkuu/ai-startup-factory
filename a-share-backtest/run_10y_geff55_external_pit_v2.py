from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import baostock as bs
import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_external_audit as ext
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_geff55_external_pit_v2'); OUT.mkdir(exist_ok=True); ext.OUT=OUT
SPEC=next(s for s in mega.specs_twostage() if s['name']=='g_eff_55')

def status_for_dates(dates):
    raw=ext.all_stock_dates(sorted(set(pd.to_datetime(dates)))); raw.to_csv(OUT/'baostock_all_stock_raw.csv',index=False)
    z=ext.normalize_status(raw); z.to_csv(OUT/'baostock_status.csv',index=False); return z

def attach_status(q,status):
    x=q.copy()
    ss=status.rename(columns={'date':'signal_date','trade_status':'sig_bs_trade','risk_name':'sig_risk','code_name':'sig_name'})[['signal_date','code','sig_name','sig_bs_trade','sig_risk']]
    es=status.rename(columns={'date':'trade_date','trade_status':'exec_bs_trade','risk_name':'exec_risk','code_name':'exec_name'})[['trade_date','code','exec_name','exec_bs_trade','exec_risk']]
    x=x.merge(ss,on=['signal_date','code'],how='left').merge(es,on=['trade_date','code'],how='left')
    x['sig_known']=x.sig_name.notna(); x['exec_known']=x.exec_name.notna()
    sig_ok=x.sig_known & (x.sig_bs_trade.fillna(0)==1) & (~x.sig_risk.fillna(True))
    exe_ok=x.exec_known & (x.exec_bs_trade.fillna(0)==1)
    # Preserve original T ranking. PIT/ST only blocks execution; blocked names are NOT replaced by next-ranked names.
    x['exec_buy_allowed']=x.exec_buy_allowed.fillna(False) & sig_ok & exe_ok & (~x.exec_risk.fillna(True))
    x['exec_sell_allowed']=x.exec_sell_allowed.fillna(False) & exe_ok
    return x

def coverage(q,status):
    rows=[]
    for d,g in q.groupby('signal_date'):
        have=status[status.date==pd.Timestamp(d)].code.nunique(); n=g.code.nunique(); rows.append({'date':d,'qlib_n':n,'baostock_n':have,'coverage':have/max(1,n)})
    z=pd.DataFrame(rows); z.to_csv(OUT/'signal_status_coverage.csv',index=False); return z

def crosscheck(q,tm,max_codes=250):
    if tm.empty:return {'trade_rows':0,'history_codes':0,'coverage':0.0}
    t=tm.merge(q[['signal_date','trade_date','code','exec_open','exec_volume','exec_factor']].drop_duplicates(['signal_date','trade_date','code']),on=['signal_date','trade_date','code'],how='left')
    codes=sorted(t.code.unique())[:max_codes]; hists={}
    for i,c in enumerate(codes,1):
        z=ext.query_history(c)
        if len(z): hists[c]=z
        if i%25==0: print('CROSS HIST',i,'/',len(codes),flush=True)
    rows=[]
    for r in t[t.code.isin(codes)].itertuples(index=False):
        z=hists.get(r.code)
        if z is None: continue
        m=z[z.date==pd.Timestamp(r.trade_date)]
        if m.empty: continue
        b=m.iloc[0]; fac=float(r.exec_factor) if np.isfinite(r.exec_factor) and r.exec_factor>0 else 1.; qopen=float(r.exec_open)/fac; qvol=float(r.exec_volume)*fac*100.0
        rows.append({'trade_date':r.trade_date,'code':r.code,'side':r.side,'qlib_raw_open':qopen,'bs_raw_open':b.open,'open_rel_err':abs(qopen-b.open)/b.open if b.open>0 else np.nan,'qlib_raw_volume_shares':qvol,'bs_volume_shares':b.volume,'volume_ratio':qvol/b.volume if b.volume>0 else np.nan,'bs_isST':b.isST,'bs_tradestatus':b.tradestatus})
    x=pd.DataFrame(rows); x.to_csv(OUT/'cross_source_trades.csv',index=False)
    denom=len(t[t.code.isin(codes)])
    return {'trade_rows':denom,'matched_rows':len(x),'history_codes':len(codes),'coverage':len(x)/max(1,denom),'median_open_rel_err':float(x.open_rel_err.median()) if len(x) else np.nan,'p95_open_rel_err':float(x.open_rel_err.quantile(.95)) if len(x) else np.nan,'median_volume_ratio':float(x.volume_ratio.replace([np.inf,-np.inf],np.nan).median()) if len(x) else np.nan,'st_trade_rows':int((x.bs_isST==1).sum()) if len(x) else 0,'halt_trade_rows':int((x.bs_tradestatus!=1).sum()) if len(x) else 0}

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board'); q=mega.make_rank(p,SPEC)
    lg=bs.login(); print('BAOSTOCK LOGIN',lg.error_code,lg.error_msg,flush=True)
    if lg.error_code!='0': raise RuntimeError('baostock login failed')
    try:
        status=status_for_dates(list(q.signal_date.unique())+list(q.trade_date.dropna().unique())); cov=coverage(q,status); qs=attach_status(q,status)
        phases=[]; eqs=[]; tm0=pd.DataFrame()
        for ph in range(18):
            st,e,tr,tm=strict.runq(qs,ph,cal,members,bm); st['phase']=ph; phases.append(st); eqs.append(e)
            if ph==0: tm0=tm.copy()
        ph=pd.DataFrame(phases); ph.to_csv(OUT/'phases.csv',index=False); es,ee=strict.ensemble_rows(eqs,bm); es['phase_count']=18; pd.DataFrame([es]).to_csv(OUT/'ensemble.csv',index=False); fa.annual(ee).to_csv(OUT/'ensemble_annual.csv',index=False)
        cross=crosscheck(qs,tm0); pd.DataFrame([cross]).to_csv(OUT/'cross_source_summary.csv',index=False)
    finally: bs.logout()
    freg,links=ext.factor_regress(ee); ok=freg[freg.status=='OK'] if len(freg) else pd.DataFrame(); mincov=float(cov.coverage.min()) if len(cov) else 0.; medcov=float(cov.coverage.median()) if len(cov) else 0.
    gates={'signal_status_min_coverage_ge_95pct':int(mincov>=.95),'all18_positive':int((ph.total_return>0).all()),'all18_pseudo_positive':int((ph.pseudo_oos_2022_2026_return>0).all()),'ensemble_positive':int(es['total_return']>0),'cross_source_coverage_ge_95pct':int(cross.get('coverage',0)>=.95),'open_median_error_le_10bp':int(cross.get('median_open_rel_err',1)<=.001),'volume_ratio_between_0_98_1_02':int(.98<=cross.get('median_volume_ratio',0)<=1.02),'factor_models_ge_2':int(len(ok)>=2),'alpha_positive_all':int(len(ok)>=1 and (ok.alpha_ann>0).all()),'alpha_significant_any':int(len(ok)>=1 and (ok.alpha_t_hac>1.96).any())}
    gd=pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]); gd.to_csv(OUT/'gates.csv',index=False)
    verdict={**ua,'market_factor':market_code,'candidate':'frozen GEff55 + strict board limit + PIT ST/trade status execution block','signal_status_min_coverage':mincov,'signal_status_median_coverage':medcov,'phase0_total':float(ph.loc[ph.phase==0,'total_return'].iloc[0]),'phase_median_total':float(ph.total_return.median()),'ensemble_total':es['total_return'],'ensemble_cagr':es['cagr'],'ensemble_mdd':es['max_drawdown'],'ensemble_pseudo':es['pseudo_oos_2022_2026_return'],**cross,'factor_models_ok':len(ok),'gates_passed':int(gd['pass'].sum()),'gates_total':len(gd),'hard_pass':int(gd['pass'].all())}
    pd.DataFrame([verdict]).to_csv(OUT/'verdict.csv',index=False); print('VERDICT');print(pd.DataFrame([verdict]).to_string(index=False),flush=True);print('GATES');print(gd.to_string(index=False),flush=True);print('FACTORS');print(freg.to_string(index=False),flush=True)

if __name__=='__main__': main()
