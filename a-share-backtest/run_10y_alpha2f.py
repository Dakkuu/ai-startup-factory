from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd

import run_10y_era_backtest as base

START = pd.Timestamp("2016-07-29")
END = pd.Timestamp("2026-07-29")
WARM = pd.Timestamp("2014-01-01")
OUT = Path("results_alpha2f")
OUT.mkdir(exist_ok=True)

N_HOLD = 30
ENTRY_PCT = 0.10
KEEP_PCT = 0.30
MIN_LIST_DAYS = 120
MIN_LIQ20 = 50_000_000.0
SLIPPAGE = 0.0010
VOLUME_PARTICIPATION = 0.05
INITIAL_CASH = 1_000_000.0

RMOM_LOOKBACK = 126
SKIP_RECENT = 21
IVOL_LOOKBACK = 60
BETA_LOOKBACK = 252
MIN_BETA_OBS = 126

PARAM_GRID = [
    (63,40),(63,60),(63,80),
    (126,40),(126,60),(126,80),
    (189,40),(189,60),(189,80),
]

def active_for(mm: pd.DataFrame, d: pd.Timestamp) -> bool:
    return bool(((mm.start <= d) & (mm.end >= d)).any())

def one_price_locked(r) -> bool:
    vals = [float(r.exec_open), float(r.exec_high), float(r.exec_low)]
    return all(np.isfinite(vals)) and abs(vals[1] - vals[2]) < 1e-12 and abs(vals[0] - vals[1]) < 1e-12

def build_panel(cal: pd.DatetimeIndex, members: pd.DataFrame, rmom_lb=RMOM_LOOKBACK, ivol_lb=IVOL_LOOKBACK):
    trade_cal = cal[(cal >= START) & (cal <= END)]
    signal_dates = pd.DatetimeIndex(trade_cal[::5])
    alltrade = cal[cal <= END]
    exec_map = {}
    for s in signal_dates:
        k = alltrade.searchsorted(s, side="right")
        exec_map[s] = alltrade[k] if k < len(alltrade) else pd.NaT

    bm_close = base.qb.read_bin("SH000985", "close", cal).loc[WARM:END].dropna()
    bm_ret = bm_close.pct_change()
    bm_mu_prev = bm_ret.rolling(BETA_LOOKBACK, min_periods=MIN_BETA_OBS).mean().shift(1)
    bm_var_prev = bm_ret.rolling(BETA_LOOKBACK, min_periods=MIN_BETA_OBS).var().shift(1)

    rows = []
    codes = sorted(members.code.unique())
    for i, code in enumerate(codes, 1):
        mm = members[members.code == code]
        cols = {}
        for f in ["open","high","low","close","volume","factor"]:
            s = base.qb.read_bin(code, f, cal)
            if not s.empty:
                cols[f] = s
        if not all(x in cols for x in ["open","high","low","close","volume"]):
            continue
        z = pd.concat(cols, axis=1).loc[WARM:END].copy()
        if z.empty:
            continue
        if "factor" not in z:
            z["factor"] = 1.0
        z["factor"] = z.factor.replace(0, np.nan).fillna(1.0)
        r = z.close.pct_change()
        cnt = z.close.notna().rolling(MIN_LIST_DAYS).sum()
        liq20 = (z.close.abs() * z.volume.abs()).rolling(20).mean()

        aligned_m = bm_ret.reindex(z.index)
        s_mu_prev = r.rolling(BETA_LOOKBACK, min_periods=MIN_BETA_OBS).mean().shift(1)
        cov_prev = r.rolling(BETA_LOOKBACK, min_periods=MIN_BETA_OBS).cov(aligned_m).shift(1)
        beta = cov_prev / bm_var_prev.reindex(z.index)
        alpha = s_mu_prev - beta * bm_mu_prev.reindex(z.index)
        resid = r - alpha - beta * aligned_m

        rmom_n = rmom_lb - SKIP_RECENT + 1
        rmom = resid.shift(SKIP_RECENT).rolling(rmom_n, min_periods=max(40, int(rmom_n*0.8))).sum()
        ivol = resid.rolling(ivol_lb, min_periods=max(30, int(ivol_lb*0.8))).std()

        for sd in signal_dates:
            ed = exec_map.get(sd, pd.NaT)
            if pd.isna(ed) or sd not in z.index or ed not in z.index:
                continue
            if not active_for(mm, sd) or not active_for(mm, ed):
                continue
            if cnt.get(sd, 0) < MIN_LIST_DAYS:
                continue
            vals = [rmom.get(sd, np.nan), ivol.get(sd, np.nan), liq20.get(sd, np.nan)]
            if not all(np.isfinite(vals)):
                continue
            er = z.loc[ed]
            if not all(np.isfinite(float(er.get(x, np.nan))) for x in ["open","high","low","volume"]):
                continue
            rows.append({
                "signal_date": sd,
                "trade_date": ed,
                "code": code,
                "rmom": float(rmom.loc[sd]),
                "ivol": float(ivol.loc[sd]),
                "liq20": float(liq20.loc[sd]),
                "exec_open": float(er.open),
                "exec_high": float(er.high),
                "exec_low": float(er.low),
                "exec_volume": float(er.volume),
                "exec_factor": float(er.factor) if np.isfinite(er.factor) and er.factor > 0 else 1.0,
            })
        if i % 500 == 0:
            print("factor histories", i, "/", len(codes), flush=True)

    p = pd.DataFrame(rows)
    if p.empty:
        raise RuntimeError("no factor panel")
    p = p[p.liq20 >= MIN_LIQ20].copy()

    def add_ranks(g):
        g = g.copy()
        g["rmom_pct"] = g.rmom.rank(ascending=False, pct=True, method="average")
        g["ivol_pct"] = g.ivol.rank(ascending=True, pct=True, method="average")
        g["score2f"] = 0.5*(1-g.rmom_pct) + 0.5*(1-g.ivol_pct)
        g["score2f_pct"] = g.score2f.rank(ascending=False, pct=True, method="average")
        return g
    p = p.groupby("signal_date", group_keys=False).apply(add_ranks).reset_index(drop=True)
    return p

@dataclass
class Pos:
    units: float
    entry_cost: float
    entry_date: pd.Timestamp
    last_price: float

def choose(g: pd.DataFrame, current: set[str], variant: str):
    x = g.copy()
    if variant == "rmom":
        x["rank_pct"] = x.rmom_pct
    elif variant == "ivol":
        x["rank_pct"] = x.ivol_pct
    else:
        x["rank_pct"] = x.score2f_pct
    x = x.sort_values(["rank_pct","liq20"], ascending=[True,False])
    keep = [c for c in current if c in set(x.loc[x.rank_pct <= KEEP_PCT, "code"])]
    keep = keep[:N_HOLD]
    if len(keep) < N_HOLD:
        entrants = [c for c in x.loc[x.rank_pct <= ENTRY_PCT, "code"] if c not in keep]
        keep.extend(entrants[:N_HOLD-len(keep)])
    return keep[:N_HOLD]

def fee(gross: float, side: str, d: pd.Timestamp, mult=1.0):
    return mult * base.fee(gross, side, d)

def simulate(panel: pd.DataFrame, variant: str, cost_mult=1.0):
    by = {d:g.set_index("code", drop=False) for d,g in panel.groupby("signal_date")}
    cash = INITIAL_CASH
    pos: dict[str,Pos] = {}
    eq, trades, timing = [], [], []
    turnover_notional = 0.0

    for d in sorted(by):
        g = by[d]
        target = choose(g.reset_index(drop=True), set(pos), variant)
        tgt = set(target)

        for c, pp in pos.items():
            if c in g.index and np.isfinite(g.loc[c].exec_open):
                pp.last_price = float(g.loc[c].exec_open)
        nav_open = cash + sum(pp.units*pp.last_price for pp in pos.values())

        for c in list(pos):
            if c in tgt:
                continue
            if c not in g.index:
                continue
            r = g.loc[c]
            if one_price_locked(r):
                continue
            px = float(r.exec_open)*(1-SLIPPAGE)
            gross = pos[c].units*px
            cost = fee(gross, "sell", pd.Timestamp(r.trade_date), cost_mult)
            old = pos.pop(c)
            cash += gross-cost
            turnover_notional += gross
            trades.append({
                "variant":variant,"code":c,"entry_date":old.entry_date,"exit_date":pd.Timestamp(r.trade_date),
                "net_pnl":gross-cost-old.entry_cost,"net_return":(gross-cost)/old.entry_cost-1
            })
            timing.append({"variant":variant,"signal_date":d,"trade_date":pd.Timestamp(r.trade_date),"side":"sell","code":c})

        per = nav_open*0.99/N_HOLD
        for c in target:
            if c in pos or c not in g.index:
                continue
            r = g.loc[c]
            if one_price_locked(r):
                continue
            factor = float(r.exec_factor) if np.isfinite(r.exec_factor) and r.exec_factor > 0 else 1.0
            adjpx = float(r.exec_open)*(1+SLIPPAGE)
            rawpx = adjpx/factor
            if rawpx <= 0:
                continue
            max_raw_by_volume = max(0, int(abs(float(r.exec_volume))*factor*VOLUME_PARTICIPATION//100)*100)
            raw_shares = int(min(per, cash*0.98)//(rawpx*100))*100
            if max_raw_by_volume > 0:
                raw_shares = min(raw_shares, max_raw_by_volume)
            if raw_shares <= 0:
                continue
            units = raw_shares/factor
            gross = units*adjpx
            cost = fee(gross, "buy", pd.Timestamp(r.trade_date), cost_mult)
            total = gross+cost
            if total > cash:
                continue
            cash -= total
            pos[c] = Pos(units,total,pd.Timestamp(r.trade_date),float(r.exec_open))
            turnover_notional += gross
            timing.append({"variant":variant,"signal_date":d,"trade_date":pd.Timestamp(r.trade_date),"side":"buy","code":c})

        nav = cash + sum(pp.units*pp.last_price for pp in pos.values())
        eq.append({"variant":variant,"signal_date":d,"trade_date":g.trade_date.iloc[0],
                   "equity":nav,"cash":cash,"positions":len(pos)})

    e = pd.DataFrame(eq)
    t = pd.DataFrame(trades)
    tm = pd.DataFrame(timing)
    if len(tm):
        bad = tm[pd.to_datetime(tm.signal_date) >= pd.to_datetime(tm.trade_date)]
        if len(bad):
            raise RuntimeError(f"trade timing violations={len(bad)}")
    return e,t,tm,turnover_notional

def perf(eq, tr, turnover_notional):
    s = eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float)
    r = s.pct_change().dropna()
    total = float(s.iloc[-1]/s.iloc[0]-1)
    years = max((s.index[-1]-s.index[0]).days/365.25, 1e-9)
    cagr = float((s.iloc[-1]/s.iloc[0])**(1/years)-1)
    dd = s/s.cummax()-1
    sharpe = float(r.mean()/r.std()*np.sqrt(52)) if r.std() > 0 else np.nan
    downside = r[r<0].std()
    sortino = float(r.mean()/downside*np.sqrt(52)) if pd.notna(downside) and downside>0 else np.nan
    wins = tr.loc[tr.net_pnl>0,"net_pnl"].sum() if len(tr) else 0.0
    losses = tr.loc[tr.net_pnl<0,"net_pnl"].sum() if len(tr) else 0.0
    pf = float(wins/abs(losses)) if losses < 0 else np.nan
    avg_win = tr.loc[tr.net_pnl>0,"net_return"].mean() if len(tr) else np.nan
    avg_loss = tr.loc[tr.net_pnl<0,"net_return"].mean() if len(tr) else np.nan
    payoff = float(avg_win/abs(avg_loss)) if pd.notna(avg_loss) and avg_loss<0 else np.nan
    return {
        "final_asset":float(s.iloc[-1]),"total_return":total,"cagr":cagr,
        "max_drawdown":float(dd.min()),"sharpe":sharpe,"sortino":sortino,
        "trades":int(len(tr)),"win_rate":float((tr.net_pnl>0).mean()) if len(tr) else np.nan,
        "profit_factor":pf,"payoff_ratio":payoff,
        "turnover_notional":float(turnover_notional),
        "turnover_over_initial":float(turnover_notional/INITIAL_CASH),
    }

def annual_returns(eq):
    out=[]
    q=eq.copy()
    q["year"]=pd.to_datetime(q.trade_date).dt.year
    for y,z in q.groupby("year"):
        out.append({"year":int(y),"return":float(z.equity.iloc[-1]/z.equity.iloc[0]-1)})
    return pd.DataFrame(out)

def period_return(eq, a, b):
    z=eq[(pd.to_datetime(eq.trade_date)>=pd.Timestamp(a))&(pd.to_datetime(eq.trade_date)<=pd.Timestamp(b))]
    if len(z)<2:return np.nan
    return float(z.equity.iloc[-1]/z.equity.iloc[0]-1)

def ic_stats(panel, horizon=5):
    p=panel.sort_values(["code","trade_date"]).copy()
    p["fwd"] = p.groupby("code").exec_open.shift(-1)/p.exec_open-1
    rows=[]
    for d,g in p.groupby("signal_date"):
        z=g.dropna(subset=["fwd"])
        if len(z)<100:continue
        rows.append({
            "signal_date":d,
            "ic_rmom":z.rmom.corr(z.fwd,method="spearman"),
            "ic_ivol":(-z.ivol).corr(z.fwd,method="spearman"),
            "ic_2f":z.score2f.corr(z.fwd,method="spearman")
        })
    q=pd.DataFrame(rows)
    stats=[]
    for c in ["ic_rmom","ic_ivol","ic_2f"]:
        x=q[c].dropna()
        stats.append({"factor":c,"mean_ic":x.mean(),"ic_std":x.std(),
                      "icir":x.mean()/x.std()*np.sqrt(52) if x.std()>0 else np.nan,
                      "positive_ic_rate":(x>0).mean(),"n":len(x)})
    return q,pd.DataFrame(stats)

def quintiles(panel):
    p=panel.sort_values(["code","trade_date"]).copy()
    p["fwd"]=p.groupby("code").exec_open.shift(-1)/p.exec_open-1
    out=[]
    for factor,col,asc in [("rmom","rmom",True),("ivol","ivol",False),("2f","score2f",True)]:
        for d,g in p.groupby("signal_date"):
            z=g.dropna(subset=["fwd",col]).copy()
            if len(z)<100:continue
            rank=z[col].rank(pct=True,ascending=asc)
            z["q"]=pd.cut(rank,[0,.2,.4,.6,.8,1.0],labels=[1,2,3,4,5],include_lowest=True)
            for q,zz in z.groupby("q",observed=True):
                out.append({"factor":factor,"signal_date":d,"quintile":int(q),"mean_fwd":zz.fwd.mean()})
    q=pd.DataFrame(out)
    return q.groupby(["factor","quintile"],as_index=False).mean(numeric_only=True)

def robustness(eq,tr):
    z=eq.copy()
    z["ret"]=z.equity.pct_change()
    weekly=z.ret.dropna().sort_values(ascending=False)
    base_total=z.equity.iloc[-1]/z.equity.iloc[0]-1
    def compounded_without(k):
        x=z.ret.dropna().copy()
        idx=x.nlargest(min(k,len(x))).index
        x.loc[idx]=0.0
        return float((1+x).prod()-1)
    pnl=float(tr.net_pnl.sum()) if len(tr) else 0.0
    best5=float(tr.nlargest(min(5,len(tr)),"net_pnl").net_pnl.sum()) if len(tr) else 0.0
    return {
        "base_total_return":float(base_total),
        "return_without_best5_weeks":compounded_without(5),
        "return_without_best1pct_weeks":compounded_without(max(1,int(math.ceil(len(weekly)*.01)))),
        "completed_trade_pnl":pnl,
        "best5_trade_pnl":best5,
        "pnl_without_best5_trades":pnl-best5,
    }

def main():
    base.START = START
    base.WARM = WARM
    base.END = END
    base.OUT = OUT
    cal,members,ua = base.load_base()

    panel = build_panel(cal,members)
    panel.to_pickle(OUT/"panel_core.pkl")

    ic_ts,ics=ic_stats(panel)
    ic_ts.to_csv(OUT/"ic_timeseries.csv",index=False)
    ics.to_csv(OUT/"ic_summary.csv",index=False)
    quintiles(panel).to_csv(OUT/"quintiles.csv",index=False)

    summaries=[]; all_eq=[];all_tr=[];all_tm=[];annual=[]
    for v in ["rmom","ivol","2f"]:
        eq,tr,tm,to=simulate(panel,v,1.0)
        st=perf(eq,tr,to);st["variant"]=v
        st["train_2016_2021_return"]=period_return(eq,"2016-07-29","2021-12-31")
        st["sealed_2022_2026_return"]=period_return(eq,"2022-01-01","2026-07-29")
        summaries.append(st)
        all_eq.append(eq);all_tr.append(tr);all_tm.append(tm)
        ar=annual_returns(eq);ar["variant"]=v;annual.append(ar)

    eq2,tr2,tm2,to2=simulate(panel,"2f",2.0)
    stress=perf(eq2,tr2,to2);stress["variant"]="2f_double_cost"

    sm=pd.DataFrame(summaries)
    sm.to_csv(OUT/"summary.csv",index=False)
    pd.DataFrame([stress]).to_csv(OUT/"double_cost.csv",index=False)
    pd.concat(all_eq,ignore_index=True).to_csv(OUT/"equity.csv",index=False)
    pd.concat(all_tr,ignore_index=True).to_csv(OUT/"trades.csv",index=False)
    pd.concat(all_tm,ignore_index=True).to_csv(OUT/"timing.csv",index=False)
    pd.concat(annual,ignore_index=True).to_csv(OUT/"annual_returns.csv",index=False)

    core_eq=all_eq[2]; core_tr=all_tr[2]
    pd.DataFrame([robustness(core_eq,core_tr)]).to_csv(OUT/"robustness.csv",index=False)

    bm=base.qb.read_bin("SH000985","close",cal).loc[START:END].dropna()
    bench=float(bm.iloc[-1]/bm.iloc[0]-1)
    all_timing=pd.concat(all_tm,ignore_index=True)
    audit={
        **ua,
        "benchmark":"SH000985",
        "benchmark_return":bench,
        "core_rule":"0.5*rank(residual momentum 126..21)+0.5*rank(-residual vol 60)",
        "beta_rule":"rolling OLS alpha/beta 252d, coefficients shifted 1 day; min 126 obs",
        "portfolio_rule":"30 names equal-weight; enter top10%; retain until below top30%; rebalance every 5 trading days",
        "execution_rule":"close signal -> next exchange open; 100-share lots; <=5% volume; one-price day fail-closed; 10bp slippage",
        "cost_rule":"commission 2.5bp each side + historical transfer fee + historical sell stamp duty; separate 2x-cost stress",
        "st_rule":"no present-day ST-name backfill; dynamic listing membership only",
        "panel_rows":int(len(panel)),
        "signal_dates":int(panel.signal_date.nunique()),
        "trade_timing_violations":int((pd.to_datetime(all_timing.signal_date)>=pd.to_datetime(all_timing.trade_date)).sum())
    }
    if audit["trade_timing_violations"] != 0:
        raise RuntimeError("timing audit failed")
    pd.DataFrame([audit]).to_csv(OUT/"audit.csv",index=False)

    print("=== AUDIT ===",flush=True)
    print(pd.DataFrame([audit]).to_string(index=False),flush=True)
    print("=== SUMMARY ===",flush=True)
    print(sm.to_string(index=False),flush=True)
    print("=== IC ===",flush=True)
    print(ics.to_string(index=False),flush=True)
    print("=== DOUBLE COST ===",flush=True)
    print(pd.DataFrame([stress]).to_string(index=False),flush=True)
    print("=== ROBUSTNESS ===",flush=True)
    print(pd.DataFrame([robustness(core_eq,core_tr)]).to_string(index=False),flush=True)

if __name__ == "__main__":
    main()
