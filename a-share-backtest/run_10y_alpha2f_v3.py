from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as a

OUT = Path("results_alpha2f_v3")
OUT.mkdir(exist_ok=True)
MARKET_CANDIDATES = ("SH000985", "SH000300", "SH000001")
MIN_MARKET_COVERAGE = 0.98

def pick_market(cal: pd.DatetimeIndex):
    idx = cal[(cal >= a.WARM) & (cal <= a.END)]
    rows = []
    chosen = None
    chosen_s = None
    for code in MARKET_CANDIDATES:
        s = base.qb.read_bin(code, "close", cal).loc[a.WARM:a.END]
        cov = float(s.notna().sum() / max(1, len(idx)))
        rows.append({"code": code, "coverage": cov, "first": s.dropna().index.min() if s.notna().any() else pd.NaT,
                     "last": s.dropna().index.max() if s.notna().any() else pd.NaT})
        print("market coverage", code, cov, flush=True)
        if chosen is None and cov >= MIN_MARKET_COVERAGE:
            chosen, chosen_s = code, s.dropna()
    if chosen is None:
        best = max(rows, key=lambda x: x["coverage"])
        if best["coverage"] < 0.95:
            raise RuntimeError(f"FAIL-CLOSED no sufficiently complete market factor {rows}")
        chosen = best["code"]
        chosen_s = base.qb.read_bin(chosen, "close", cal).loc[a.WARM:a.END].dropna()
    pd.DataFrame(rows).to_csv(OUT / "market_coverage.csv", index=False)
    print("chosen market factor", chosen, flush=True)
    return chosen, chosen_s, pd.DataFrame(rows)

def _active_mask(mm: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    out = np.zeros(len(dates), dtype=bool)
    valid = ~pd.isna(dates)
    for r in mm.itertuples(index=False):
        out |= valid & (dates >= r.start) & (dates <= r.end)
    return out

def build_panel(cal: pd.DatetimeIndex, members: pd.DataFrame, market_close: pd.Series) -> pd.DataFrame:
    trade_cal = cal[(cal >= a.START) & (cal <= a.END)]
    signal_dates = pd.DatetimeIndex(trade_cal[::5])
    alltrade = cal[cal <= a.END]
    exec_dates = []
    for s in signal_dates:
        k = alltrade.searchsorted(s, side="right")
        exec_dates.append(alltrade[k] if k < len(alltrade) else pd.NaT)
    exec_dates = pd.DatetimeIndex(exec_dates)

    bm_ret = market_close.reindex(cal[(cal >= a.WARM) & (cal <= a.END)]).pct_change(fill_method=None)
    bm_mu_prev = bm_ret.rolling(a.BETA_LOOKBACK, min_periods=a.MIN_BETA_OBS).mean().shift(1)
    bm_var_prev = bm_ret.rolling(a.BETA_LOOKBACK, min_periods=a.MIN_BETA_OBS).var().shift(1)

    frames = []
    codes = sorted(members.code.unique())
    all_factor_cols = [f"rmom{w}" for w in a.RMOM_WINDOWS] + [f"ivol{w}" for w in a.IVOL_WINDOWS]
    core_factor_cols = [f"rmom{a.CORE_RMOM}", f"ivol{a.CORE_IVOL}"]

    for i, code in enumerate(codes, 1):
        mm = members[members.code == code]
        cols = {}
        for f in ["open", "high", "low", "close", "volume", "factor"]:
            s = base.qb.read_bin(code, f, cal)
            if not s.empty:
                cols[f] = s
        if not all(f in cols for f in ["open", "high", "low", "close", "volume"]):
            continue

        z = pd.concat(cols, axis=1).loc[a.WARM:a.END].copy()
        if z.empty:
            continue
        if "factor" not in z:
            z["factor"] = 1.0
        z["factor"] = z.factor.replace(0, np.nan).fillna(1.0)

        r = z.close.pct_change(fill_method=None)
        count120 = z.close.notna().rolling(a.MIN_LIST_DAYS).sum()
        liq20 = (z.close.abs() * z.volume.abs()).rolling(20).mean()

        m = bm_ret.reindex(z.index)
        s_mu_prev = r.rolling(a.BETA_LOOKBACK, min_periods=a.MIN_BETA_OBS).mean().shift(1)
        cov_prev = r.rolling(a.BETA_LOOKBACK, min_periods=a.MIN_BETA_OBS).cov(m).shift(1)
        beta = cov_prev / bm_var_prev.reindex(z.index)
        alpha = s_mu_prev - beta * bm_mu_prev.reindex(z.index)
        resid = r - alpha - beta * m

        fac = {}
        for w in a.RMOM_WINDOWS:
            n = w - a.SKIP_RECENT + 1
            fac[f"rmom{w}"] = resid.shift(a.SKIP_RECENT).rolling(
                n, min_periods=max(30, int(n * 0.80))
            ).sum()
        for w in a.IVOL_WINDOWS:
            fac[f"ivol{w}"] = resid.rolling(
                w, min_periods=max(25, int(w * 0.80))
            ).std()

        sig_data = {
            "count120": count120.reindex(signal_dates).to_numpy(),
            "liq20": liq20.reindex(signal_dates).to_numpy(),
        }
        for k, s in fac.items():
            sig_data[k] = s.reindex(signal_dates).to_numpy()
        sig = pd.DataFrame(sig_data)

        ex = z.reindex(exec_dates).reset_index(drop=True)
        active_s = _active_mask(mm, signal_dates)
        active_e = _active_mask(mm, exec_dates)

        valid = active_s & active_e & (~pd.isna(exec_dates))
        valid &= np.asarray(sig["count120"] >= a.MIN_LIST_DAYS)
        valid &= np.isfinite(sig[["liq20"] + core_factor_cols].to_numpy()).all(axis=1)
        valid &= np.isfinite(ex[["open", "high", "low", "volume"]].to_numpy()).all(axis=1)
        if not valid.any():
            continue

        idx = np.flatnonzero(valid)
        rec_dict = {
            "signal_date": signal_dates[idx],
            "trade_date": exec_dates[idx],
            "code": code,
            "liq20": sig["liq20"].to_numpy()[idx].astype(float),
            "exec_open": ex["open"].to_numpy()[idx].astype(float),
            "exec_high": ex["high"].to_numpy()[idx].astype(float),
            "exec_low": ex["low"].to_numpy()[idx].astype(float),
            "exec_volume": ex["volume"].to_numpy()[idx].astype(float),
            "exec_factor": ex["factor"].to_numpy()[idx].astype(float),
        }
        for c in all_factor_cols:
            rec_dict[c] = sig[c].to_numpy()[idx].astype(float)
        frames.append(pd.DataFrame(rec_dict))
        if i % 500 == 0:
            print("alpha2f-v3 factor histories", i, "/", len(codes), flush=True)

    if not frames:
        raise RuntimeError("no alpha2f-v3 factor panel")
    p = pd.concat(frames, ignore_index=True)
    if not (pd.to_datetime(p.signal_date) < pd.to_datetime(p.trade_date)).all():
        raise RuntimeError("panel signal/trade timing violation")

    p["liq_rank_pct"] = p.groupby("signal_date")["liq20"].rank(
        pct=True, method="average", ascending=False
    )
    p = p[p.liq_rank_pct <= a.LIQ_KEEP_PCT].copy()

    for w in a.RMOM_WINDOWS:
        p[f"rmom{w}_pct"] = p.groupby("signal_date")[f"rmom{w}"].rank(
            pct=True, method="average", ascending=False
        )
    for w in a.IVOL_WINDOWS:
        p[f"ivol{w}_pct"] = p.groupby("signal_date")[f"ivol{w}"].rank(
            pct=True, method="average", ascending=True
        )
    for rw in a.RMOM_WINDOWS:
        for vw in a.IVOL_WINDOWS:
            s = 0.5 * (1 - p[f"rmom{rw}_pct"]) + 0.5 * (1 - p[f"ivol{vw}_pct"])
            p[f"score_{rw}_{vw}"] = s
            p[f"score_{rw}_{vw}_pct"] = s.groupby(p["signal_date"]).rank(
                pct=True, method="average", ascending=False
            )

    gs = p.groupby("signal_date").size()
    print("PANEL", p.shape, "dates", p.signal_date.nunique(),
          "min", int(gs.min()), "median", float(gs.median()), "max", int(gs.max()), flush=True)
    if p.signal_date.nunique() < 480 or gs.median() < 1500 or gs.min() < 500:
        raise RuntimeError(
            f"FAIL-CLOSED abnormal eligible universe dates={p.signal_date.nunique()} "
            f"min={gs.min()} median={gs.median()} max={gs.max()}"
        )
    return p

def main():
    base.START = a.START
    base.WARM = a.WARM
    base.END = a.END
    base.OUT = OUT
    cal, members, ua = base.load_base()

    market_code, market_close, market_cov = pick_market(cal)
    panel = build_panel(cal, members, market_close)
    panel.to_pickle(OUT / "panel_core.pkl")

    benchmark = market_close.loc[a.START:a.END].dropna()
    benchmark_return = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)

    ic_ts, ic_sum = a.ic_stats(panel)
    ic_ts.to_csv(OUT / "ic_timeseries.csv", index=False)
    ic_sum.to_csv(OUT / "ic_summary.csv", index=False)
    q = a.quintiles(panel)
    q.to_csv(OUT / "quintiles.csv", index=False)

    summaries, all_eq, all_tr, all_tm, annual = [], [], [], [], []
    for v in ["rmom", "ivol", "2f"]:
        print("simulate", v, flush=True)
        eq, tr, tm, to = a.simulate(panel, v, cal, members, 1.0, daily_mtm=True)
        st = a.perf(eq, tr, to, benchmark)
        st["variant"] = v
        st["train_2016_2021_return"] = a.period_return(eq, "2016-07-29", "2021-12-31")
        st["sealed_2022_2026_return"] = a.period_return(eq, "2022-01-01", "2026-07-29")
        st["benchmark_return"] = benchmark_return
        st["excess_total_return"] = st["total_return"] - benchmark_return
        summaries.append(st)
        all_eq.append(eq); all_tr.append(tr); all_tm.append(tm)
        ar = a.annual_returns(eq); ar["variant"] = v; annual.append(ar)

    print("simulate double all costs", flush=True)
    eq2, tr2, tm2, to2 = a.simulate(panel, "2f", cal, members, 2.0, daily_mtm=True)
    stress = a.perf(eq2, tr2, to2, benchmark)
    stress["variant"] = "2f_double_all_costs"

    grid = []
    for rw in a.RMOM_WINDOWS:
        for vw in a.IVOL_WINDOWS:
            v = f"2f_{rw}_{vw}"
            rankc = a.rank_col(v)
            avail = panel.groupby("signal_date")[rankc].apply(lambda s: int(s.notna().sum()))
            valid_dates = int((avail >= 500).sum())
            if valid_dates < 400:
                grid.append({"variant": v, "valid_signal_dates": valid_dates, "status": "insufficient_history"})
                continue
            print("grid", v, flush=True)
            eqg, trg, tmg, tog = a.simulate(panel, v, cal, members, 1.0, daily_mtm=False)
            stg = a.perf(eqg, trg, tog, None)
            stg["variant"] = v
            stg["valid_signal_dates"] = valid_dates
            stg["status"] = "ok"
            stg["sealed_2022_2026_return"] = a.period_return(eqg, "2022-01-01", "2026-07-29")
            grid.append(stg)
    grid_df = pd.DataFrame(grid)

    sm = pd.DataFrame(summaries)
    rob = pd.DataFrame([a.robustness(all_eq[2], all_tr[2])])
    all_timing = pd.concat(all_tm, ignore_index=True)
    gs = panel.groupby("signal_date").size()
    writeoffs = int(sum(
        (t.exit_reason == "membership_end_writeoff").sum()
        for t in all_tr if len(t) and "exit_reason" in t
    ))

    audit = {
        **ua,
        "market_factor": market_code,
        "market_factor_coverage": float(market_cov.loc[market_cov.code == market_code, "coverage"].iloc[0]),
        "benchmark_return": benchmark_return,
        "panel_rows": int(len(panel)),
        "signal_dates": int(panel.signal_date.nunique()),
        "eligible_min": int(gs.min()),
        "eligible_median": float(gs.median()),
        "eligible_max": int(gs.max()),
        "liquidity_rule": "top 80% by PIT 20d close*volume cross-sectional rank; unit-invariant",
        "core_rule": "0.5*rank(residual momentum 126..21)+0.5*rank(-residual vol 60)",
        "beta_rule": f"rolling market model vs {market_code}; 252d lagged alpha/beta; min126 obs",
        "portfolio_rule": "30 equal target sleeves; enter top10%; retain through top30%; rebalance every 5 trading days",
        "execution_rule": "close signal -> next exchange open; 100-share lots; <=5% volume; one-price day fail-closed; 10bp/side slippage",
        "cost_rule": "2.5bp commission + historical transfer + historical sell stamp; 2x stress doubles fees and slippage",
        "daily_mtm": 1,
        "trade_timing_violations": int((pd.to_datetime(all_timing.signal_date) >= pd.to_datetime(all_timing.trade_date)).sum()),
        "membership_end_writeoffs": writeoffs,
        "st_rule": "no present-day ST-name backfill; dynamic listing membership only",
    }
    if audit["trade_timing_violations"] != 0:
        raise RuntimeError(f"timing audit failed {audit['trade_timing_violations']}")

    pd.DataFrame([audit]).to_csv(OUT / "audit.csv", index=False)
    sm.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame([stress]).to_csv(OUT / "double_cost.csv", index=False)
    rob.to_csv(OUT / "robustness.csv", index=False)
    grid_df.to_csv(OUT / "parameter_grid.csv", index=False)
    pd.concat(annual, ignore_index=True).to_csv(OUT / "annual_returns.csv", index=False)
    pd.concat(all_eq, ignore_index=True).to_csv(OUT / "equity_daily.csv", index=False)
    pd.concat(all_tr, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    all_timing.to_csv(OUT / "timing.csv", index=False)

    print("=== AUDIT ==="); print(pd.DataFrame([audit]).to_string(index=False), flush=True)
    print("=== SUMMARY ==="); print(sm.to_string(index=False), flush=True)
    print("=== IC ==="); print(ic_sum.to_string(index=False), flush=True)
    print("=== QUINTILES ==="); print(q.to_string(index=False), flush=True)
    print("=== DOUBLE COST ==="); print(pd.DataFrame([stress]).to_string(index=False), flush=True)
    print("=== ROBUSTNESS ==="); print(rob.to_string(index=False), flush=True)
    print("=== GRID ==="); print(grid_df.to_string(index=False), flush=True)
    print("=== ANNUAL ===")
    print(pd.concat(annual, ignore_index=True).pivot(index="year", columns="variant", values="return").to_string(), flush=True)

if __name__ == "__main__":
    main()
