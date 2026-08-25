from __future__ import annotations

import math, os, tarfile, urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE_TAG = os.getenv("QLIB_RELEASE_TAG", "2026-07-29")
START = pd.Timestamp(os.getenv("BT_START", "2025-07-30"))
END = pd.Timestamp(os.getenv("BT_END", RELEASE_TAG))
INITIAL_CASH = 1_000_000.0
COMMISSION = 0.00025
MIN_COMMISSION = 5.0
STAMP_DUTY_SELL = 0.0005
TRANSFER_FEE = 0.00001
SLIPPAGE = 0.0005
MAX_NAMES = 5
MAX_WEIGHT = 0.20
MIN_LIQUIDITY = 50_000_000.0

ROOT = Path("qlib_data")
OUT = Path("results_qlib")
OUT.mkdir(exist_ok=True)


def download_and_extract():
    if (ROOT / "calendars" / "day.txt").exists():
        return
    ROOT.mkdir(exist_ok=True)
    url = f"https://github.com/chenditc/investment_data/releases/download/{RELEASE_TAG}/qlib_bin.tar.gz"
    archive = Path("qlib_bin.tar.gz")
    print("downloading", url, flush=True)
    urllib.request.urlretrieve(url, archive)
    print("archive bytes", archive.stat().st_size, flush=True)
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        # archive normally has one top-level directory; strip it safely
        tops = {Path(m.name).parts[0] for m in members if Path(m.name).parts}
        if len(tops) == 1:
            top = next(iter(tops))
            for m in members:
                parts = Path(m.name).parts
                if len(parts) <= 1:
                    continue
                m.name = str(Path(*parts[1:]))
                tf.extract(m, ROOT)
        else:
            tf.extractall(ROOT)
    archive.unlink(missing_ok=True)
    if not (ROOT / "calendars" / "day.txt").exists():
        # tolerate an extra nesting level
        hits = list(ROOT.rglob("calendars/day.txt"))
        if len(hits) != 1:
            raise RuntimeError(f"cannot locate qlib calendar; found={hits}")
        base = hits[0].parent.parent
        for child in base.iterdir():
            target = ROOT / child.name
            if target.exists():
                continue
            child.rename(target)


def load_calendar():
    cal = pd.to_datetime(pd.read_csv(ROOT / "calendars" / "day.txt", header=None)[0]).tolist()
    if not cal:
        raise RuntimeError("empty calendar")
    if max(cal) < END:
        raise RuntimeError(f"release ends {max(cal)} before requested END={END.date()}")
    return pd.DatetimeIndex(cal)


def load_membership(cal):
    p = ROOT / "instruments" / "csi500.txt"
    if not p.exists():
        raise RuntimeError("csi500 instrument file missing")
    df = pd.read_csv(p, sep="\t", header=None, names=["code", "start", "end"], usecols=[0,1,2])
    df["code"] = df.code.astype(str).str.upper()
    df["start"] = pd.to_datetime(df.start)
    df["end"] = pd.to_datetime(df.end)
    df = df[(df.end >= START - pd.Timedelta(days=120)) & (df.start <= END)].copy()
    union = set(df.code)
    exited = df[(df.end >= START) & (df.end < END)].code.nunique()
    entered = df[(df.start > START) & (df.start <= END)].code.nunique()
    counts = []
    for d in cal[(cal >= START) & (cal <= END)]:
        counts.append(((df.start <= d) & (df.end >= d)).sum())
    if len(union) < 520:
        raise RuntimeError(f"FAIL-CLOSED: historical union too small ({len(union)}); survivor-bias risk")
    if exited < 10 or entered < 10:
        raise RuntimeError(f"FAIL-CLOSED: too few historical changes entered={entered}, exited={exited}")
    if min(counts) < 450 or max(counts) > 550:
        raise RuntimeError(f"FAIL-CLOSED: abnormal daily membership range {min(counts)}..{max(counts)}")
    audit = pd.DataFrame([{
        "release_tag": RELEASE_TAG,
        "start": START.date(), "end": END.date(),
        "union_members": len(union), "entered": entered, "exited": exited,
        "min_daily_members": min(counts), "max_daily_members": max(counts),
        "calendar_last_date": max(cal).date(),
    }])
    audit.to_csv(OUT / "universe_audit.csv", index=False)
    print(audit.to_string(index=False), flush=True)
    return df


def read_bin(code: str, field: str, cal: pd.DatetimeIndex):
    folder = ROOT / "features" / code.lower()
    p = folder / f"{field}.day.bin"
    if not p.exists():
        return pd.Series(dtype=float)
    arr = np.fromfile(p, dtype="<f4")
    if len(arr) <= 1:
        return pd.Series(dtype=float)
    start_idx = int(arr[0])
    vals = arr[1:].astype(float)
    end_idx = min(start_idx + len(vals), len(cal))
    vals = vals[: max(0, end_idx - start_idx)]
    return pd.Series(vals, index=cal[start_idx:end_idx], name=field)


def build_panel(cal, membership_df):
    warm = START - pd.Timedelta(days=180)
    codes = sorted(set(membership_df.code))
    frames=[]
    missing=0
    for i, code in enumerate(codes, 1):
        cols={}
        for fld in ["open","high","low","close","volume","factor"]:
            s=read_bin(code,fld,cal)
            if not s.empty: cols[fld]=s
        if not all(x in cols for x in ["open","high","low","close","volume"]):
            missing += 1; continue
        d=pd.concat(cols,axis=1)
        d=d[(d.index>=warm)&(d.index<=END)].copy()
        if d.empty: continue
        d["code"]=code; d["date"]=d.index
        if "factor" not in d: d["factor"]=1.0
        d["factor"]=d.factor.replace(0,np.nan)
        # qlib prices are adjusted; close*volume is approximately raw turnover when volume is inversely adjusted.
        d["liquidity"]=(d.close.abs()*d.volume.abs()).replace([np.inf,-np.inf],np.nan)
        frames.append(d.reset_index(drop=True))
        if i%100==0: print("loaded",i,"/",len(codes),flush=True)
    if len(frames)<500:
        raise RuntimeError(f"FAIL-CLOSED: only {len(frames)} member histories available, missing={missing}")
    panel=pd.concat(frames,ignore_index=True).sort_values(["code","date"])
    g=panel.groupby("code",group_keys=False)
    panel["ret1"]=g.close.pct_change()
    panel["mom5"]=g.close.pct_change(5)
    panel["mom20"]=g.close.pct_change(20)
    panel["mom60"]=g.close.pct_change(60)
    panel["ma20"]=g.close.transform(lambda s:s.rolling(20).mean())
    panel["ma60"]=g.close.transform(lambda s:s.rolling(60).mean())
    panel["vol20"]=g.ret1.transform(lambda s:s.rolling(20).std())
    panel["vol_ma20"]=g.volume.transform(lambda s:s.rolling(20).mean())
    panel["liq_ma20"]=g.liquidity.transform(lambda s:s.rolling(20).mean())
    panel["prev20_high"]=g.high.transform(lambda s:s.shift(1).rolling(20).max())
    panel["vol_ratio"]=panel.volume/panel.vol_ma20
    return panel


def member_codes(mdf, d):
    z=mdf[(mdf.start<=d)&(mdf.end>=d)]
    return set(z.code)


def rankpct(s, ascending=True):
    return s.rank(pct=True,ascending=ascending,method="average")


def select(day, strategy, risk_on=True):
    x=day.copy()
    needed=["mom5","mom20","mom60","ma20","ma60","vol20","prev20_high","vol_ratio","liq_ma20"]
    x=x.dropna(subset=needed)
    x=x[(x.liq_ma20>=MIN_LIQUIDITY)&(x.close>0)&(x.open>0)]
    if x.empty:return []
    if strategy=="trend_breakout":
        x=x[(x.close>x.ma20)&(x.ma20>x.ma60)&(x.mom20>0)]
        x["score"]=.40*rankpct(x.mom20)+.35*rankpct(x.mom60)+.25*rankpct(x.vol_ratio)
    elif strategy=="relative_momentum":
        x=x[(x.mom60>0)&(x.close>x.ma20)]
        x["score"]=.50*rankpct(x.mom60)+.35*rankpct(x.mom20)+.15*rankpct(x.liq_ma20)
    elif strategy=="mean_reversion":
        x=x[(x.mom60>-0.10)&(x.mom5<-0.05)]
        x["score"]=.70*rankpct(-x.mom5)+.30*rankpct(x.liq_ma20)
    elif strategy=="lowvol_trend":
        if not risk_on:return []
        x=x[(x.mom60>0)&(x.close>x.ma60)&(x.vol20>0)]
        x["score"]=.55*rankpct(x.mom60)+.45*rankpct(-x.vol20)
    elif strategy=="volume_breakout":
        x=x[(x.close>x.prev20_high)&(x.vol_ratio>1.5)&(x.mom20>0)]
        x["score"]=.45*rankpct(x.vol_ratio)+.35*rankpct(x.mom20)+.20*rankpct(x.liq_ma20)
    elif strategy=="multifactor":
        x=x[(x.mom60>-0.10)&(x.vol20>0)]
        x["score"]=.30*rankpct(x.mom20)+.25*rankpct(x.mom60)+.20*rankpct(-x.vol20)+.15*rankpct(x.liq_ma20)+.10*rankpct(x.vol_ratio)
    else: raise ValueError(strategy)
    return x.sort_values("score",ascending=False).code.head(MAX_NAMES).tolist() if not x.empty else []


def limit_pct(code):
    c=code.upper()
    if c.startswith("SH688") or c.startswith("SZ300") or c.startswith("SZ301"):
        return .20
    return .10


def one_price_locked(row, prev_adj_close, side):
    if row is None:return True
    if not all(np.isfinite(row.get(c,np.nan)) for c in ["open","high","low","close"]):return True
    if not (abs(row.high-row.low)<1e-8 and abs(row.open-row.high)<1e-8):return False
    if not np.isfinite(prev_adj_close) or prev_adj_close<=0:return False
    pct=row.open/prev_adj_close-1
    lim=limit_pct(row.code)
    return (side=="buy" and pct>=lim-0.005) or (side=="sell" and pct<=-lim+0.005)


def fee(gross, side):
    return max(MIN_COMMISSION,gross*COMMISSION)+gross*TRANSFER_FEE+(gross*STAMP_DUTY_SELL if side=="sell" else 0.0)

@dataclass
class Pos:
    units: float   # raw shares / entry factor; economic units compatible with adjusted prices
    entry_gross: float
    entry_cost: float
    entry_date: pd.Timestamp
    entry_raw_price: float


def backtest(panel, mdf, strategy, benchmark_close=None):
    dates=sorted(pd.Timestamp(d) for d in panel.date.unique() if START<=pd.Timestamp(d)<=END)
    by={pd.Timestamp(d):z.set_index("code",drop=False) for d,z in panel.groupby("date")}
    cash=INITIAL_CASH; pos={}; target=None; sigdate=None; eq=[]; trades=[]
    for k,d in enumerate(dates):
        day=by.get(d)
        if day is None:continue
        if target is not None:
            tgt=set(target)
            # sell names no longer targeted
            for code in list(pos):
                if code in tgt:continue
                row=day.loc[code] if code in day.index else None
                prev_row=by.get(dates[k-1]).loc[code] if k>0 and code in by.get(dates[k-1],pd.DataFrame()).index else None
                prev_close=float(prev_row.close) if prev_row is not None else np.nan
                if row is None or one_price_locked(row,prev_close,"sell"):continue
                adj_px=float(row.open)*(1-SLIPPAGE)
                gross=pos[code].units*adj_px
                cost=fee(gross,"sell")
                cash+=gross-cost
                p=pos.pop(code)
                pnl=(gross-cost)-p.entry_cost
                factor=float(row.factor) if np.isfinite(row.factor) and row.factor>0 else 1.0
                raw_px=adj_px/factor
                trades.append({"strategy":strategy,"code":code,"signal_date":sigdate,"entry_date":p.entry_date,"entry_raw_price":p.entry_raw_price,"exit_date":d,"exit_raw_price":raw_px,"net_pnl":pnl,"net_return":pnl/p.entry_cost})
            nav_open=cash+sum(p.units*float(day.loc[c].open) for c,p in pos.items() if c in day.index and np.isfinite(day.loc[c].open))
            per=min(nav_open*MAX_WEIGHT,nav_open/max(1,len(tgt)))
            for code in target:
                if code in pos or code not in day.index:continue
                row=day.loc[code]
                prev_row=by.get(dates[k-1]).loc[code] if k>0 and code in by.get(dates[k-1],pd.DataFrame()).index else None
                prev_close=float(prev_row.close) if prev_row is not None else np.nan
                if one_price_locked(row,prev_close,"buy"):continue
                factor=float(row.factor) if np.isfinite(row.factor) and row.factor>0 else 1.0
                adj_px=float(row.open)*(1+SLIPPAGE)
                raw_px=adj_px/factor
                if not np.isfinite(raw_px) or raw_px<=0:continue
                raw_shares=int(min(per,cash*.98)//(raw_px*100))*100
                if raw_shares<=0:continue
                units=raw_shares/factor
                gross=units*adj_px
                cost=fee(gross,"buy")
                total=gross+cost
                if total>cash:continue
                cash-=total
                pos[code]=Pos(units,total,total,d,raw_px)
        nav=cash
        for code,p in pos.items():
            if code in day.index and np.isfinite(day.loc[code].close): nav+=p.units*float(day.loc[code].close)
        eq.append({"date":d,"strategy":strategy,"equity":nav,"cash":cash,"n_positions":len(pos)})

        members=member_codes(mdf,d)
        universe=day[day.code.isin(members)].copy()
        # point-in-time market regime: equal-weight median member above MA60, using only today's data
        valid=universe.dropna(subset=["close","ma60"])
        risk_on=True if valid.empty else (valid.close>valid.ma60).mean()>=0.45
        # frequency fixed ex ante; no outcome-dependent timing
        rebalance = strategy in {"trend_breakout","mean_reversion","volume_breakout"} or (k%5==0)
        if rebalance:
            target=select(universe,strategy,risk_on); sigdate=d
    return pd.DataFrame(eq),pd.DataFrame(trades)


def calc_stats(eq,tr):
    e=eq.set_index("date").equity.astype(float)
    r=e.pct_change().dropna()
    total=e.iloc[-1]/INITIAL_CASH-1
    years=max((e.index[-1]-e.index[0]).days/365.25,1/252)
    cagr=(e.iloc[-1]/INITIAL_CASH)**(1/years)-1
    dd=e/e.cummax()-1; mdd=dd.min()
    sd=r.std(ddof=0); sharpe=np.sqrt(252)*r.mean()/sd if sd>0 else np.nan
    dn=r[r<0].std(ddof=0); sortino=np.sqrt(252)*r.mean()/dn if np.isfinite(dn) and dn>0 else np.nan
    if tr.empty:
        return {"final_asset":e.iloc[-1],"total_return":total,"cagr":cagr,"max_drawdown":mdd,"sharpe":sharpe,"sortino":sortino,"trades":0,"win_rate":np.nan,"avg_win":np.nan,"avg_loss":np.nan,"payoff":np.nan,"profit_factor":np.nan}
    wins=tr[tr.net_pnl>0]; losses=tr[tr.net_pnl<0]
    aw=wins.net_return.mean() if len(wins) else np.nan; al=losses.net_return.mean() if len(losses) else np.nan
    pf=wins.net_pnl.sum()/abs(losses.net_pnl.sum()) if len(losses) and losses.net_pnl.sum()!=0 else np.nan
    return {"final_asset":e.iloc[-1],"total_return":total,"cagr":cagr,"max_drawdown":mdd,"sharpe":sharpe,"sortino":sortino,"trades":len(tr),"win_rate":len(wins)/len(tr),"avg_win":aw,"avg_loss":al,"payoff":aw/abs(al) if np.isfinite(aw) and np.isfinite(al) and al!=0 else np.nan,"profit_factor":pf}


def benchmark(cal):
    for code in ["sh000905","SH000905"]:
        s=read_bin(code,"close",cal)
        if not s.empty:
            s=s[(s.index>=START)&(s.index<=END)].dropna()
            if len(s)>2:return s/s.iloc[0]
    return pd.Series(dtype=float)


def main():
    download_and_extract()
    cal=load_calendar()
    mdf=load_membership(cal)
    panel=build_panel(cal,mdf)
    b=benchmark(cal)
    strategies=["trend_breakout","relative_momentum","mean_reversion","lowvol_trend","volume_breakout","multifactor"]
    summaries=[]; alltr=[]; alleq=[]
    for s in strategies:
        eq,tr=backtest(panel,mdf,s,b)
        st=calc_stats(eq,tr); st["strategy"]=s
        if not b.empty:
            bb=b.reindex(eq.date,method="ffill").dropna()
            st["benchmark_return"]=float(bb.iloc[-1]-1) if len(bb) else np.nan
            st["excess_return"]=st["total_return"]-st["benchmark_return"] if np.isfinite(st["benchmark_return"]) else np.nan
        summaries.append(st); alltr.append(tr); alleq.append(eq)
    sm=pd.DataFrame(summaries).sort_values("total_return",ascending=False)
    tr=pd.concat(alltr,ignore_index=True) if alltr else pd.DataFrame()
    eq=pd.concat(alleq,ignore_index=True)
    sm.to_csv(OUT/"summary.csv",index=False); tr.to_csv(OUT/"trades.csv",index=False); eq.to_csv(OUT/"equity.csv",index=False)
    # robustness: first/second half and remove best 5 completed trades (trade-level diagnostic)
    mid=START+(END-START)/2
    seg=[]
    for s in strategies:
        e=eq[eq.strategy==s].set_index("date").equity
        for name,a,z in [("H1",START,mid),("H2",mid,END)]:
            x=e[(e.index>=a)&(e.index<=z)]
            if len(x)>1: seg.append({"strategy":s,"segment":name,"return":x.iloc[-1]/x.iloc[0]-1})
    pd.DataFrame(seg).to_csv(OUT/"segments.csv",index=False)
    robust=[]
    for s in strategies:
        t=tr[tr.strategy==s].copy()
        total_net=t.net_pnl.sum() if len(t) else 0
        drop5=t.nlargest(min(5,len(t)),"net_pnl").net_pnl.sum() if len(t) else 0
        robust.append({"strategy":s,"completed_trade_net_pnl":total_net,"best5_net_pnl":drop5,"trade_pnl_after_removing_best5":total_net-drop5})
    pd.DataFrame(robust).to_csv(OUT/"robustness.csv",index=False)
    print("\n=== AUDITED LEADERBOARD ===")
    print(sm.to_string(index=False),flush=True)

if __name__=="__main__": main()
