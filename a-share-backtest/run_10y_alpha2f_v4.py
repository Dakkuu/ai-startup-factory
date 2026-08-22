from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as a

OUT = Path("results_alpha2f_v4")
OUT.mkdir(exist_ok=True)
MARKET_CANDIDATES = ("SH000985", "SH000300", "SH000001")
MIN_MARKET_COVERAGE = 0.98

def pick_market(cal: pd.DatetimeIndex):
    idx = cal[(cal >= a.WARM) & (cal <= a.END)]
    rows=[]; chosen=None; chosen_s=None
    for code in MARKET_CANDIDATES:
        s=base.qb.read_bin(code,"close",cal).loc[a.WARM:a.END]
        cov=float(s.notna().sum()/max(1,len(idx)))
        rows.append({"code":code,"coverage":cov,"first":s.dropna().index.min() if s.notna().any() else pd.NaT,
                     "last":s.dropna().index.max() if s.notna().any() else pd.NaT})
        print("market coverage",code,cov,flush=True)
        if chosen is None and cov>=MIN_MARKET_COVERAGE:
            chosen=code; chosen_s=s.dropna()
    if chosen is None:
        best=max(rows,key=lambda x:x["coverage"])
        if best["coverage"]<0.95: raise RuntimeError(f"FAIL-CLOSED market coverage {rows}")
        chosen=best["code"]; chosen_s=base.qb.read_bin(chosen,"close",cal).loc[a.WARM:a.END].dropna()
    pd.DataFrame(rows).to_csv(OUT/"market_coverage.csv",index=False)
    print("chosen market",chosen,flush=True)
    return chosen,chosen_s,pd.DataFrame(rows)

def active_mask(mm,dates):
    out=np.zeros(len(dates),dtype=bool); valid=~pd.isna(dates)
    for r in mm.itertuples(index=False): out |= valid & (dates>=r.start) & (dates<=r.end)
    return out

def build_panel(cal,members,market_close):
    trade_cal=cal[(cal>=a.START)&(cal<=a.END)]
    signal_dates=pd.DatetimeIndex(trade_cal[::5])
    alltrade=cal[cal<=a.END]
    exec_dates=[]
    for s in signal_dates:
        k=alltrade.searchsorted(s,side="right")
        exec_dates.append(alltrade[k] if k<len(alltrade) else pd.NaT)
    exec_dates=pd.DatetimeIndex(exec_dates)

    bm_ret=market_close.reindex(cal[(cal>=a.WARM)&(cal<=a.END)]).pct_change(fill_method=None)
    bm_mu=bm_ret.rolling(a.BETA_LOOKBACK,min_periods=a.MIN_BETA_OBS).mean().shift(1)
    bm_var=bm_ret.rolling(a.BETA_LOOKBACK,min_periods=a.MIN_BETA_OBS).var().shift(1)
    factor_cols=[f"rmom{w}" for w in a.RMOM_WINDOWS]+[f"ivol{w}" for w in a.IVOL_WINDOWS]
    frames=[]; codes=sorted(members.code.unique())
    for i,code in enumerate(codes,1):
        mm=members[members.code==code]; cols={}
        for f in ["open","high","low","close","volume","factor"]:
            s=base.qb.read_bin(code,f,cal)
            if not s.empty: cols[f]=s
        if not all(f in cols for f in ["open","high","low","close","volume"]): continue
        z=pd.concat(cols,axis=1).loc[a.WARM:a.END].copy()
        if z.empty: continue
        if "factor" not in z: z["factor"]=1.0
        z["factor"]=z.factor.replace(0,np.nan).fillna(1.0)
        r=z.close.pct_change(fill_method=None)
        count120=z.close.notna().rolling(a.MIN_LIST_DAYS).sum()
        liq20=(z.close.abs()*z.volume.abs()).rolling(20).mean()
        m=bm_ret.reindex(z.index)
        smu=r.rolling(a.BETA_LOOKBACK,min_periods=a.MIN_BETA_OBS).mean().shift(1)
        cov=r.rolling(a.BETA_LOOKBACK,min_periods=a.MIN_BETA_OBS).cov(m).shift(1)
        beta=cov/bm_var.reindex(z.index); alpha=smu-beta*bm_mu.reindex(z.index)
        resid=r-alpha-beta*m
        fac={}
        for w in a.RMOM_WINDOWS:
            n=w-a.SKIP_RECENT+1
            fac[f"rmom{w}"]=resid.shift(a.SKIP_RECENT).rolling(n,min_periods=max(30,int(n*.8))).sum()
        for w in a.IVOL_WINDOWS:
            fac[f"ivol{w}"]=resid.rolling(w,min_periods=max(25,int(w*.8))).std()
        sig={"count120":count120.reindex(signal_dates).to_numpy(),"liq20":liq20.reindex(signal_dates).to_numpy()}
        for k,s in fac.items(): sig[k]=s.reindex(signal_dates).to_numpy()
        sig=pd.DataFrame(sig); ex=z.reindex(exec_dates).reset_index(drop=True)
        valid=active_mask(mm,signal_dates)&active_mask(mm,exec_dates)&(~pd.isna(exec_dates))
        valid &= np.asarray(sig.count120>=a.MIN_LIST_DAYS)
        valid &= np.isfinite(sig[["liq20"]].to_numpy()).all(axis=1)
        valid &= np.isfinite(ex[["open","high","low","volume"]].to_numpy()).all(axis=1)
        if not valid.any(): continue
        idx=np.flatnonzero(valid)
        rec={"signal_date":signal_dates[idx],"trade_date":exec_dates[idx],"code":code,
             "liq20":sig.liq20.to_numpy()[idx].astype(float),
             "exec_open":ex.open.to_numpy()[idx].astype(float),"exec_high":ex.high.to_numpy()[idx].astype(float),
             "exec_low":ex.low.to_numpy()[idx].astype(float),"exec_volume":ex.volume.to_numpy()[idx].astype(float),
             "exec_factor":ex.factor.to_numpy()[idx].astype(float)}
        for c in factor_cols: rec[c]=sig[c].to_numpy()[idx].astype(float)
        frames.append(pd.DataFrame(rec))
        if i%500==0: print("v4 histories",i,"/",len(codes),flush=True)
    if not frames: raise RuntimeError("no panel")
    p=pd.concat(frames,ignore_index=True)
    if not (pd.to_datetime(p.signal_date)<pd.to_datetime(p.trade_date)).all(): raise RuntimeError("panel timing")

    p["liq_rank_pct"]=p.groupby("signal_date").liq20.rank(pct=True,method="average",ascending=False)
    liq_ok=p.liq_rank_pct<=a.LIQ_KEEP_PCT
    for w in a.RMOM_WINDOWS:
        col=f"rmom{w}"; out=f"rmom{w}_pct"; p[out]=np.nan
        mask=liq_ok&p[col].notna()
        p.loc[mask,out]=p.loc[mask].groupby("signal_date")[col].rank(pct=True,method="average",ascending=False)
    for w in a.IVOL_WINDOWS:
        col=f"ivol{w}"; out=f"ivol{w}_pct"; p[out]=np.nan
        mask=liq_ok&p[col].notna()
        p.loc[mask,out]=p.loc[mask].groupby("signal_date")[col].rank(pct=True,method="average",ascending=True)
    for rw in a.RMOM_WINDOWS:
        for vw in a.IVOL_WINDOWS:
            rc=f"rmom{rw}_pct"; vc=f"ivol{vw}_pct"; sc=f"score_{rw}_{vw}"; pc=sc+"_pct"
            p[sc]=np.where(p[rc].notna()&p[vc].notna(),.5*(1-p[rc])+.5*(1-p[vc]),np.nan)
            p[pc]=np.nan; mask=p[sc].notna()
            p.loc[mask,pc]=p.loc[mask].groupby("signal_date")[sc].rank(pct=True,method="average",ascending=False)
    p["core_eligible"]=p[f"rmom{a.CORE_RMOM}_pct"].notna()&p[f"ivol{a.CORE_IVOL}_pct"].notna()
    core=p[p.core_eligible]; gs=core.groupby("signal_date").size(); full=p.groupby("signal_date").size()
    print("FULL",p.shape,"dates",p.signal_date.nunique(),"median",full.median(),"CORE",len(core),"dates",core.signal_date.nunique(),"min",gs.min(),"median",gs.median(),"max",gs.max(),flush=True)
    if core.signal_date.nunique()<480 or gs.median()<1500 or gs.min()<500: raise RuntimeError(f"FAIL-CLOSED core {core.signal_date.nunique()} {gs.min()} {gs.median()}")
    return p

def summary_row(panel,variant,cal,members,benchmark,cost=1.0,daily=True):
    eq,tr,tm,to=a.simulate(panel,variant,cal,members,cost,daily_mtm=daily)
    st=a.perf(eq,tr,to,benchmark if daily else None); st["variant"]=variant; st["cost_mult"]=cost
    st["train_2016_2021_return"]=a.period_return(eq,"2016-07-29","2021-12-31")
    st["sealed_2022_2026_return"]=a.period_return(eq,"2022-01-01","2026-07-29")
    st["positions_max"]=int(eq.positions.max()); st["positions_median"]=float(eq.positions.median())
    return st,eq,tr,tm

def main():
    base.START=a.START; base.WARM=a.WARM; base.END=a.END; base.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=pick_market(cal)
    panel=build_panel(cal,members,market_close); panel.to_pickle(OUT/"panel_core.pkl")
    benchmark=market_close.loc[a.START:a.END].dropna(); br=float(benchmark.iloc[-1]/benchmark.iloc[0]-1)
    core=panel[panel.core_eligible].copy()
    ic_ts,ic_sum=a.ic_stats(core); ic_ts.to_csv(OUT/"ic_timeseries.csv",index=False); ic_sum.to_csv(OUT/"ic_summary.csv",index=False)
    q=a.quintiles(core); q.to_csv(OUT/"quintiles.csv",index=False)

    rows=[]; eqs={}; trs={}; tms={}
    for v in ["rmom","ivol","2f"]:
        print("core simulate",v,flush=True); st,eq,tr,tm=summary_row(panel,v,cal,members,benchmark,1.0,True)
        st["benchmark_return"]=br; st["excess_total_return"]=st["total_return"]-br
        rows.append(st); eqs[v]=eq; trs[v]=tr; tms[v]=tm
    pd.DataFrame(rows).to_csv(OUT/"summary.csv",index=False)

    stress=[]
    for v in ["ivol","2f"]:
        for cm in [2.0,4.0]:
            print("cost stress",v,cm,flush=True); st,_,_,_=summary_row(panel,v,cal,members,benchmark,cm,True); stress.append(st)
    pd.DataFrame(stress).to_csv(OUT/"cost_stress.csv",index=False)

    # Pure low-IVOL window stability. Reuse the exact execution panel; only ranking column changes.
    ivol_grid=[]; orig=panel[f"ivol{a.CORE_IVOL}_pct"].copy()
    for w in a.IVOL_WINDOWS:
        panel[f"ivol{a.CORE_IVOL}_pct"]=panel[f"ivol{w}_pct"]
        print("ivol window",w,flush=True); st,_,_,_=summary_row(panel,"ivol",cal,members,benchmark,1.0,False); st["ivol_window"]=w; ivol_grid.append(st)
    panel[f"ivol{a.CORE_IVOL}_pct"]=orig
    pd.DataFrame(ivol_grid).to_csv(OUT/"ivol_window_grid.csv",index=False)

    # Portfolio construction stability.
    construct=[]; orig_n=a.N_HOLD; orig_keep=a.KEEP_PCT; orig_entry=a.ENTRY_PCT
    for n in [20,30,50]:
        for keep in [.20,.30,.40]:
            a.N_HOLD=n; a.KEEP_PCT=keep; a.ENTRY_PCT=.10
            print("construct",n,keep,flush=True); st,_,_,_=summary_row(panel,"ivol",cal,members,benchmark,1.0,False); st["n_hold"]=n; st["keep_pct"]=keep; construct.append(st)
    a.N_HOLD=orig_n; a.KEEP_PCT=orig_keep; a.ENTRY_PCT=orig_entry
    pd.DataFrame(construct).to_csv(OUT/"construction_grid.csv",index=False)

    robust=[]
    for v in ["ivol","2f"]:
        r=a.robustness(eqs[v],trs[v]); r["variant"]=v; robust.append(r)
    pd.DataFrame(robust).to_csv(OUT/"robustness.csv",index=False)

    annual=[]
    for v in ["rmom","ivol","2f"]:
        z=a.annual_returns(eqs[v]); z["variant"]=v; annual.append(z)
    pd.concat(annual,ignore_index=True).to_csv(OUT/"annual_returns.csv",index=False)
    pd.concat([eq.assign(variant=v) for v,eq in eqs.items()],ignore_index=True).to_csv(OUT/"equity_daily.csv",index=False)
    pd.concat([tr.assign(variant=v) for v,tr in trs.items()],ignore_index=True).to_csv(OUT/"trades.csv",index=False)
    allt=pd.concat([tm.assign(variant=v) for v,tm in tms.items()],ignore_index=True)
    allt.to_csv(OUT/"timing.csv",index=False)

    gs=core.groupby("signal_date").size(); fullgs=panel.groupby("signal_date").size()
    audit={**ua,"market_factor":market_code,"market_factor_coverage":float(mc.loc[mc.code==market_code,"coverage"].iloc[0]),
           "benchmark_return":br,"full_panel_rows":len(panel),"full_signal_dates":panel.signal_date.nunique(),
           "full_median":float(fullgs.median()),"core_eligible_rows":len(core),"core_signal_dates":core.signal_date.nunique(),
           "core_min":int(gs.min()),"core_median":float(gs.median()),"core_max":int(gs.max()),
           "liquidity_rule":"rank only: top80pct for target; all execution rows retained for exits",
           "core_rule":"low-IVOL60 and RMOM126..21; 2F equal rank blend",
           "trade_timing_violations":int((pd.to_datetime(allt.signal_date)>=pd.to_datetime(allt.trade_date)).sum()),
           "daily_mtm":1,"st_rule":"no current ST backfill; dynamic listing membership"}
    if audit["trade_timing_violations"]: raise RuntimeError("timing audit")
    pd.DataFrame([audit]).to_csv(OUT/"audit.csv",index=False)

    print("=== AUDIT ==="); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
    print("=== SUMMARY ==="); print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print("=== IC ==="); print(ic_sum.to_string(index=False),flush=True)
    print("=== QUINTILES ==="); print(q.to_string(index=False),flush=True)
    print("=== COST ==="); print(pd.DataFrame(stress).to_string(index=False),flush=True)
    print("=== IVOL WINDOWS ==="); print(pd.DataFrame(ivol_grid).to_string(index=False),flush=True)
    print("=== CONSTRUCTION ==="); print(pd.DataFrame(construct).to_string(index=False),flush=True)
    print("=== ROBUSTNESS ==="); print(pd.DataFrame(robust).to_string(index=False),flush=True)
    print("=== ANNUAL ==="); print(pd.concat(annual,ignore_index=True).pivot(index="year",columns="variant",values="return").to_string(),flush=True)

if __name__=="__main__": main()
