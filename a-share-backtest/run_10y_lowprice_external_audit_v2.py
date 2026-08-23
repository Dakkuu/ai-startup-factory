from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import baostock as bs

import run_10y_baseline_maxopt_v3 as mo
import run_10y_lowprice_signalpure_v1 as lp
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
import run_10y_max_audit as ma
import run_10y_maxopt_v3_frozen_audit as fa
hv3.patch()

OUT=Path('results_lowprice_external_v2'); OUT.mkdir(exist_ok=True)
WEIGHTS={'price':.25,'iv':.20,'ef':.20,'rmom':.22,'tstat':.13}; CFG={'liq':.55,'floor':2.0,'hold':90,'n':8,'entry':.10,'keep':.30}; NPHASE=18


def bscode(c):
    c=str(c).upper()
    if c.startswith('SH'): return 'sh.'+c[2:]
    if c.startswith('SZ'): return 'sz.'+c[2:]
    if c.startswith('BJ'): return 'bj.'+c[2:]
    return c.lower()
def rs_frame(rs):
    data=[]
    while rs.error_code=='0' and rs.next(): data.append(rs.get_row_data())
    return pd.DataFrame(data,columns=rs.fields)
def history(code):
    fields='date,code,open,high,low,close,preclose,volume,amount,tradestatus,isST'
    rs=bs.query_history_k_data_plus(bscode(code),fields,start_date='2016-07-01',end_date='2026-07-31',frequency='d',adjustflag='3')
    z=rs_frame(rs)
    if z.empty:return z
    z['date']=pd.to_datetime(z.date)
    for c in ['open','high','low','close','preclose','volume','amount','tradestatus','isST']: z[c]=pd.to_numeric(z[c],errors='coerce')
    z['qcode']=code; return z

def subset(q,ph):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[ph::NPHASE]); cols=[c for c in strict.BASECOLS+['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]; z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')
def run_phase(q,ph,cal,members,bm,cash): return ma.run_panel(subset(q,ph),cal,members,bm,n=CFG['n'],entry=CFG['entry'],keep=CFG['keep'],initial_cash=float(cash))
def combine_abs(eqs,initials):
    start=pd.Timestamp(mo.START); idx={start}; ss=[]
    for e,init in zip(eqs,initials):
        s=e.set_index(pd.to_datetime(e.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]; s=pd.concat([pd.Series({start:float(init)}),s]); s=s[~s.index.duplicated(keep='last')].sort_index(); ss.append(s); idx.update(s.index)
    idx=pd.DatetimeIndex(sorted(idx)); total=pd.concat([s.reindex(idx).ffill().fillna(float(init)) for s,init in zip(ss,initials)],axis=1).sum(axis=1); return pd.DataFrame({'trade_date':idx,'equity':total.to_numpy(float)})
def summarize(eq,bm):
    s=fa.perf_eq(eq,bm); s['train_return']=fa.period_return(eq,mo.START,mo.TRAIN_END); s['pseudo_return']=fa.period_return(eq,mo.PSEUDO_START,mo.END); return s


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=lp.attach_price(p,cal); q0=lp.rank_signal(p,WEIGHTS,CFG['liq'],CFG['floor']); q=strict.attach_gap_flags(q0,cal,'board')
    per=1e6/NPHASE; eqs=[]; tms=[]
    for ph in range(NPHASE):
        st,e,tr,tm=run_phase(q,ph,cal,members,bm,per); eqs.append(e)
        if len(tm): tms.append(tm.assign(phase=ph))
    tm=pd.concat(tms,ignore_index=True); buys=tm[tm.side=='buy'].copy(); buys=buys.merge(p[['signal_date','code','raw_price']].drop_duplicates(['signal_date','code']),on=['signal_date','code'],how='left'); ex=q[['signal_date','trade_date','code','exec_open','exec_factor']].drop_duplicates(['signal_date','trade_date','code']); buys=buys.merge(ex,on=['signal_date','trade_date','code'],how='left')
    codes=sorted(buys.code.unique()); lg=bs.login();
    if lg.error_code!='0': raise RuntimeError('baostock login failed '+lg.error_msg)
    hs={}
    try:
        for i,c in enumerate(codes,1):
            z=history(c)
            if len(z): hs[c]=z
            if i%50==0: print('BAOSTOCK',i,'/',len(codes),flush=True)
    finally: bs.logout()
    rows=[]
    for r in buys.itertuples(index=False):
        z=hs.get(r.code)
        if z is None: continue
        s=z[z.date==pd.Timestamp(r.signal_date)]; e=z[z.date==pd.Timestamp(r.trade_date)]
        if s.empty and e.empty: continue
        sr=s.iloc[0] if len(s) else None; er=e.iloc[0] if len(e) else None; qrawopen=float(r.exec_open)/float(r.exec_factor) if np.isfinite(r.exec_open) and np.isfinite(r.exec_factor) and r.exec_factor!=0 else np.nan
        rows.append({'signal_date':r.signal_date,'trade_date':r.trade_date,'code':r.code,'phase':r.phase,'qlib_raw_signal_close':r.raw_price,'bs_signal_close':float(sr.close) if sr is not None else np.nan,'signal_close_rel_err':abs(float(r.raw_price)-float(sr.close))/float(sr.close) if sr is not None and sr.close>0 and np.isfinite(r.raw_price) else np.nan,'qlib_raw_exec_open':qrawopen,'bs_exec_open':float(er.open) if er is not None else np.nan,'exec_open_rel_err':abs(qrawopen-float(er.open))/float(er.open) if er is not None and er.open>0 and np.isfinite(qrawopen) else np.nan,'bs_isST_signal':float(sr.isST) if sr is not None else np.nan,'bs_isST_exec':float(er.isST) if er is not None else np.nan,'bs_trade_status_exec':float(er.tradestatus) if er is not None else np.nan})
    x=pd.DataFrame(rows); x.to_csv(OUT/'buy_crosscheck.csv',index=False)
    stpairs=set((pd.Timestamp(r.signal_date),r.code) for r in x.itertuples(index=False) if r.bs_isST_signal==1 or r.bs_isST_exec==1)
    # Conservative ST rerun: if a frozen target was historically ST, block its buy and do not replace it.
    qs=q.copy(); key=pd.Series(list(zip(pd.to_datetime(qs.signal_date),qs.code)),index=qs.index); block=key.isin(stpairs); qs.loc[block,'exec_buy_allowed']=False
    eqs2=[]
    for ph in range(NPHASE):
        st,e,tr,tm2=run_phase(qs,ph,cal,members,bm,per); eqs2.append(e)
    baseeq=combine_abs(eqs,[per]*NPHASE); steq=combine_abs(eqs2,[per]*NPHASE); bsum=summarize(baseeq,bm); ssum=summarize(steq,bm)
    pd.DataFrame([{**bsum,'test':'base_exact_split18'},{**ssum,'test':'block_observed_ST_targets_no_replacement'}]).to_csv(OUT/'st_sensitivity.csv',index=False)
    nbuys=len(buys); matched=len(x); summary={'buy_rows':nbuys,'crosscheck_rows':matched,'coverage':matched/max(1,nbuys),'unique_buy_codes':len(codes),'st_buy_rows':int(((x.bs_isST_signal==1)|(x.bs_isST_exec==1)).sum()) if len(x) else 0,'st_signal_code_date_pairs':len(stpairs),'signal_close_median_rel_err':float(x.signal_close_rel_err.median()) if len(x) else np.nan,'signal_close_p95_rel_err':float(x.signal_close_rel_err.quantile(.95)) if len(x) else np.nan,'exec_open_median_rel_err':float(x.exec_open_rel_err.median()) if len(x) else np.nan,'exec_open_p95_rel_err':float(x.exec_open_rel_err.quantile(.95)) if len(x) else np.nan,'base_total_return':bsum['total_return'],'st_block_total_return':ssum['total_return'],'base_pseudo_return':bsum['pseudo_oos_2022_2026_return'] if 'pseudo_oos_2022_2026_return' in bsum else bsum['pseudo_return'],'st_block_pseudo_return':ssum['pseudo_oos_2022_2026_return'] if 'pseudo_oos_2022_2026_return' in ssum else ssum['pseudo_return']}
    pd.DataFrame([summary]).to_csv(OUT/'external_summary.csv',index=False); pd.DataFrame([{**ua,'market_factor':market_code,'weights':str(WEIGHTS),'config':str(CFG),'baostock_adjustflag':'3 unadjusted','st_test':'block historically observed ST frozen targets, no replacement','price_test':'Qlib adjusted close/factor and exec open/factor vs BaoStock unadjusted'}]).to_csv(OUT/'audit.csv',index=False)
    print(pd.DataFrame([summary]).to_string(index=False),flush=True)
if __name__=='__main__':main()
