from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import run_backtest_qlib as qb

RELEASE_TAG = "2026-07-29"
START = pd.Timestamp("2016-07-29")
END = pd.Timestamp("2026-07-29")
WARM = START - pd.Timedelta(days=220)
OUT = Path("results_10y_context")
OUT.mkdir(exist_ok=True)
STOCK_RE = re.compile(r'^(?:SH(?:600|601|603|605|688)\d{3}|SZ(?:000|001|002|003|300|301)\d{3}|BJ\d{6})$')


def load():
    qb.RELEASE_TAG = RELEASE_TAG
    qb.ROOT = Path("qlib_data")
    qb.download_and_extract()
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(qb.ROOT/"calendars"/"day.txt", header=None)[0]))
    p = qb.ROOT/"instruments"/"all.txt"
    m = pd.read_csv(p, sep="\t", header=None, names=["code","start","end"], usecols=[0,1,2])
    m["code"] = m.code.astype(str).str.upper()
    m["start"] = pd.to_datetime(m.start); m["end"] = pd.to_datetime(m.end)
    m = m[m.code.str.match(STOCK_RE)].copy()
    m = m[(m.end>=WARM)&(m.start<=END)]
    dtest = cal[(cal>=START)&(cal<=END)]
    counts = [int(((m.start<=d)&(m.end>=d)).sum()) for d in dtest]
    audit = {
        "release_tag": RELEASE_TAG, "start": str(START.date()), "end": str(END.date()),
        "union_members": int(m[(m.end>=START)&(m.start<=END)].code.nunique()),
        "entered": int(m[(m.start>START)&(m.start<=END)].code.nunique()),
        "exited": int(m[(m.end>=START)&(m.end<END)].code.nunique()),
        "min_daily_members": min(counts), "max_daily_members": max(counts),
    }
    if audit["union_members"] < 3000 or audit["exited"] < 50 or min(counts) < 2500:
        raise RuntimeError(f"FAIL-CLOSED suspicious 10y universe: {audit}")
    pd.DataFrame([audit]).to_csv(OUT/"universe_audit.csv", index=False)
    print("UNIVERSE", audit, flush=True)
    return cal, m


def main():
    cal, m = load()
    dates = cal[(cal>=WARM)&(cal<=END)]
    idx = pd.Index(dates)
    cols = ["n","adv","ret_sum","ret2_sum","absret_sum","above_ma60","above_ma120","mom20_pos","mom60_pos","limit_up_like","limit_down_like","liq_sum"]
    A = pd.DataFrame(0.0, index=idx, columns=cols)
    active = pd.Series(0.0, index=idx)

    codes = sorted(m.code.unique())
    for i, code in enumerate(codes, 1):
        close = qb.read_bin(code, "close", cal)
        vol = qb.read_bin(code, "volume", cal)
        if close.empty or vol.empty:
            continue
        z = pd.concat([close.rename("close"), vol.rename("volume")], axis=1).loc[WARM:END]
        if z.empty: continue
        mm = m[m.code==code]
        # instruments may have multiple listing intervals; keep only active rows.
        mask = pd.Series(False, index=z.index)
        for r in mm.itertuples(index=False):
            mask |= (z.index>=r.start) & (z.index<=r.end)
        z = z[mask]
        if z.empty: continue
        r1 = z.close.pct_change()
        ma60 = z.close.rolling(60).mean(); ma120 = z.close.rolling(120).mean()
        mom20 = z.close.pct_change(20); mom60 = z.close.pct_change(60)
        q = pd.DataFrame(index=z.index)
        q["n"] = r1.notna().astype(float)
        q["adv"] = (r1>0).astype(float)
        q["ret_sum"] = r1.fillna(0)
        q["ret2_sum"] = r1.fillna(0).pow(2)
        q["absret_sum"] = r1.fillna(0).abs()
        q["above_ma60"] = (z.close>ma60).astype(float)
        q["above_ma120"] = (z.close>ma120).astype(float)
        q["mom20_pos"] = (mom20>0).astype(float)
        q["mom60_pos"] = (mom60>0).astype(float)
        q["limit_up_like"] = (r1>=0.095).astype(float)
        q["limit_down_like"] = (r1<=-0.095).astype(float)
        q["liq_sum"] = (z.close.abs()*z.volume.abs()).fillna(0)
        use = q.index.intersection(A.index)
        A.loc[use, cols] += q.loc[use, cols]
        active.loc[use] += 1.0
        if i % 500 == 0: print("processed", i, "/", len(codes), flush=True)

    d = pd.DataFrame(index=A.index)
    den = A.n.replace(0,np.nan)
    d["equal_weight_ret"] = A.ret_sum/den
    d["advancer_ratio"] = A.adv/den
    meanr = A.ret_sum/den
    d["cross_section_dispersion"] = np.sqrt(np.maximum(A.ret2_sum/den - meanr.pow(2), 0))
    d["absret_mean"] = A.absret_sum/den
    d["breadth_ma60"] = A.above_ma60/active.replace(0,np.nan)
    d["breadth_ma120"] = A.above_ma120/active.replace(0,np.nan)
    d["mom20_positive"] = A.mom20_pos/active.replace(0,np.nan)
    d["mom60_positive"] = A.mom60_pos/active.replace(0,np.nan)
    d["limit_up_like_ratio"] = A.limit_up_like/den
    d["limit_down_like_ratio"] = A.limit_down_like/den
    d["total_liquidity_proxy"] = A.liq_sum
    d["active_stocks"] = active

    bench = qb.read_bin("SH000985", "close", cal).loc[START:END].dropna()
    if bench.empty: raise RuntimeError("benchmark SH000985 missing")
    d = d.loc[START:END].copy()
    d["benchmark_close"] = bench.reindex(d.index)
    d["benchmark_ret"] = d.benchmark_close.pct_change()
    d["benchmark_120d_mom"] = d.benchmark_close.pct_change(120)
    d["benchmark_60d_mom"] = d.benchmark_close.pct_change(60)
    d["benchmark_ma120_gap"] = d.benchmark_close/d.benchmark_close.rolling(120).mean()-1
    d.to_csv(OUT/"market_context_daily.csv", index_label="date")

    rows=[]
    for year, z in d.groupby(d.index.year):
        b=z.benchmark_close.dropna()
        if b.empty: continue
        dd=b/b.cummax()-1
        eq=(1+z.equal_weight_ret.fillna(0)).cumprod()
        rows.append({
            "year": int(year),
            "benchmark_return": float(b.iloc[-1]/b.iloc[0]-1),
            "benchmark_max_drawdown": float(dd.min()),
            "benchmark_realized_vol": float(z.benchmark_ret.std()*np.sqrt(244)),
            "equal_weight_return": float(eq.iloc[-1]-1),
            "avg_advancer_ratio": float(z.advancer_ratio.mean()),
            "avg_breadth_ma60": float(z.breadth_ma60.mean()),
            "yearend_breadth_ma60": float(z.breadth_ma60.dropna().iloc[-1]),
            "avg_breadth_ma120": float(z.breadth_ma120.mean()),
            "avg_limit_up_like_ratio": float(z.limit_up_like_ratio.mean()),
            "avg_limit_down_like_ratio": float(z.limit_down_like_ratio.mean()),
            "avg_cross_section_dispersion": float(z.cross_section_dispersion.mean()),
            "median_daily_liquidity_proxy": float(z.total_liquidity_proxy.median()),
            "yearend_active_stocks": int(z.active_stocks.iloc[-1]),
        })
    y=pd.DataFrame(rows)
    y.to_csv(OUT/"market_context_yearly.csv", index=False)
    print("=== YEARLY CONTEXT ===")
    print(y.to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
