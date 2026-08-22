from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import baostock as bs

import run_10y_hard_executor_v2 as hv2
import run_10y_signal_pure_panel as sp
hv2.patch(); sp.patch()

import run_10y_external_audit_v2 as v2
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_grand_opt as grand
import run_10y_balanced_exact as be
import run_10y_max_audit as ma

OUT=Path('results_external_audit_v3'); OUT.mkdir(exist_ok=True)
v2.OUT=OUT; sp.OUT=OUT


def pit_st_audit_signal_pure(p,q,cal,members,bm,base_tm):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); selected=pd.DatetimeIndex(dates[::12])
    trade_map=q[q.signal_date.isin(selected)][['signal_date','trade_date']].drop_duplicates()
    external_dates=list(selected)+list(pd.to_datetime(trade_map.trade_date))
    raw=v2.old.all_stock_dates(external_dates); raw.to_csv(OUT/'baostock_all_stock_raw.csv',index=False)
    status=v2.normalize_status(raw); status.to_csv(OUT/'baostock_status.csv',index=False)
    cov=v2.true_coverage(p,status,selected)

    # Signal-date eligibility/ranking may use signal-date PIT status, which is known by T close.
    qp=v2.pit_rerank_v2(p,status,selected)
    # T+1 status is attached only as an execution flag. It MUST NOT change rank_test or
    # cause replacement by the next-ranked stock.
    tr=status.rename(columns={'date':'trade_date'})[['trade_date','code','code_name','trade_status','risk_name']].rename(columns={'code_name':'exec_bs_name','trade_status':'exec_bs_status','risk_name':'exec_bs_risk'})
    qp=qp.merge(tr,on=['trade_date','code'],how='left')
    qp['exec_can_trade']=qp.exec_bs_name.notna() & (qp.exec_bs_status==1)
    qp['exec_buy_allowed']=qp.exec_can_trade & (~qp.exec_bs_risk.fillna(True))
    cols=['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test','exec_can_trade','exec_buy_allowed']
    z=qp[cols].copy(); z['ivol60_pct']=z.rank_test; z=z.drop(columns='rank_test')
    st,eq,trades,tm=ma.run_panel(z,cal,members,bm)
    pd.DataFrame([st]).to_csv(OUT/'pit_st_summary.csv',index=False); sim.annual_returns(eq).to_csv(OUT/'pit_st_annual_legacy_boundary.csv',index=False)

    # Baseline defects from the non-PIT strategy, for documentation only.
    b=base_tm[base_tm.side=='buy'].copy()
    sig=status.rename(columns={'date':'signal_date'})[['signal_date','code','code_name','risk_name','trade_status']]
    trd=status.rename(columns={'date':'trade_date'})[['trade_date','code','code_name','risk_name','trade_status']].rename(columns={'code_name':'trade_code_name','risk_name':'trade_risk_name','trade_status':'trade_status_bs'})
    b=b.merge(sig,on=['signal_date','code'],how='left').merge(trd,on=['trade_date','code'],how='left')
    b.to_csv(OUT/'baseline_buys_pit_status.csv',index=False)
    base_sig_st=int(b.risk_name.fillna(False).sum()); base_trade_st=int(b.trade_risk_name.fillna(False).sum()); base_unknown=int((b.code_name.isna()|b.trade_code_name.isna()).sum())

    final_sig_bad,final_trade_bad,final_unknown,final_halted=v2.audit_trade_risk(tm,status)
    # Demonstrate future tradability is not used to construct the signal universe.
    selected_q=qp[np.isfinite(qp.rank_test)].copy()
    future_untradable=int((~selected_q.exec_can_trade).sum())
    future_buy_blocked=int((~selected_q.exec_buy_allowed).sum())
    pd.DataFrame([{'ranked_signal_rows':len(selected_q),'ranked_rows_untradable_next_open':future_untradable,'ranked_rows_buy_blocked_next_open':future_buy_blocked}]).to_csv(OUT/'future_tradability_positive_control.csv',index=False)
    return st,eq,trades,tm,cov,status,base_sig_st,base_trade_st,base_unknown,final_sig_bad,final_trade_bad,final_unknown,final_halted,future_untradable,future_buy_blocked


def exact_annual(eq,start=sim.START,initial_cash=sim.INITIAL_CASH):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index()
    rows=[]
    prev=float(initial_cash)
    for y in sorted(s.index.year.unique()):
        z=s[s.index.year==y]
        if z.empty: continue
        endv=float(z.iloc[-1]); rows.append({'year':int(y),'return':endv/prev-1,'start_equity':prev,'end_equity':endv}); prev=endv
    return pd.DataFrame(rows)


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members)
    bm=market_close.loc[sim.START:sim.END].dropna(); q=be.anchor_weighted(p,'liq70',.60)
    bst,beq,btr,btm=ma.run_q(q,60,0,cal,members,bm); pd.DataFrame([bst]).to_csv(OUT/'baseline_signal_pure_no_st.csv',index=False)
    lg=bs.login(); print('BAOSTOCK LOGIN',lg.error_code,lg.error_msg,flush=True)
    if lg.error_code!='0': raise RuntimeError('baostock login failed')
    try:
        pst,peq,ptr,ptm,cov,status,base_sig_st,base_trade_st,base_unknown,final_sig_bad,final_trade_bad,final_unknown,final_halted,future_untradable,future_buy_blocked=pit_st_audit_signal_pure(p,q,cal,members,bm,btm)
        cross=v2.price_crosscheck_v2(p,ptm); pd.DataFrame([cross]).to_csv(OUT/'cross_source_summary.csv',index=False)
    finally: bs.logout()
    exact_annual(peq).to_csv(OUT/'pit_st_annual.csv',index=False)
    freg,links=v2.factor_regress_v2(peq); okreg=freg[freg.status=='OK'] if len(freg) else pd.DataFrame(); mincov=float(cov.coverage.min()) if len(cov) else 0.
    alpha_pos=bool(len(okreg)>=1 and (okreg.alpha_ann>0).all()); alpha_sig=bool(len(okreg)>=1 and (okreg.alpha_t_hac60>1.96).any()) if 'alpha_t_hac60' in okreg else False
    gates={
      'signal_pure_positive_control_future_untradable_gt0':int(future_untradable>0),
      'baostock_qlib_intersection_coverage_ge_95pct':int(mincov>=.95),
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
    verdict={**ua,'market_factor':market_code,'strategy_rule':'signal-pure + PIT signal ST; T+1 status execution-only; corrected volume unit','baseline_signal_pure_total':bst['total_return'],'pit_st_total_return':pst['total_return'],'pit_st_cagr':pst['cagr'],'pit_st_mdd':pst['max_drawdown'],'pit_st_sharpe':pst['sharpe'],'baostock_min_true_coverage':mincov,'baseline_signal_st_buys':base_sig_st,'baseline_trade_day_st_buys':base_trade_st,'corrected_signal_st_buys':final_sig_bad,'corrected_trade_day_st_buys':final_trade_bad,'corrected_unknown_buy_status':final_unknown,'corrected_halted_buys':final_halted,'ranked_rows_untradable_next_open':future_untradable,'ranked_rows_buy_blocked_next_open':future_buy_blocked,**cross,'external_factor_models_ok':len(okreg),'external_hard_pass':int(gd['pass'].all()),'gates_passed':int(gd['pass'].sum()),'gates_total':len(gd),'factor_links':'|'.join(links.keys())}
    pd.DataFrame([verdict]).to_csv(OUT/'external_verdict.csv',index=False)
    print('=== SIGNAL-PURE EXTERNAL VERDICT ==='); print(pd.DataFrame([verdict]).to_string(index=False),flush=True)
    print('=== GATES ==='); print(gd.to_string(index=False),flush=True)
    print('=== FACTOR REGRESSION ==='); print(freg.to_string(index=False),flush=True)

if __name__=='__main__': main()
