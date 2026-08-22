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
OUT = Path("results_alpha2f_v2")
OUT.mkdir(exist_ok=True)

INITIAL_CASH = 1_000_000.0
N_HOLD = 30
ENTRY_PCT = 0.10
KEEP_PCT = 0.30
MIN_LIST_DAYS = 120
LIQ_KEEP_PCT = 0.80
SLIPPAGE = 0.0010
VOLUME_PARTICIPATION = 0.05

SKIP_RECENT = 21
BETA_LOOKBACK = 252
MIN_BETA_OBS = 126
RMOM_WINDOWS = (63, 126, 189)
IVOL_WINDOWS = (40, 60, 80)
CORE_RMOM = 126
CORE_IVOL = 60

def _active_mask(mm: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    out = np.zeros(len(dates), dtype=bool)
    valid = ~pd.isna(dates)
    for r in mm.itertuples(index=False):
        out |= valid & (dates >= r.start) & (dates <= r.end)
    return out

def build_panel(cal: pd.DatetimeIndex, members: pd.DataFrame) -> pd.DataFrame:
    trade_cal = cal[(cal >= START) & (cal <= END)]
    signal_dates = pd.DatetimeIndex(trade_cal[::5])
    alltrade = cal[cal <= END]
    exec_dates = []
    for s in signal_dates:
        k = alltrade.searchsorted(s, side="right")
        exec_dates.append(alltrade[k] if k < len(alltrade) else pd.NaT)
    exec_dates = pd.DatetimeIndex(exec_dates)

    bm_close = base.qb.read_bin("SH000985", "close", cal).loc[WARM:END].dropna()
    bm_ret = bm_close.pct_change(fill_method=None)
    bm_mu_prev = bm_ret.rolling(BETA_LOOKBACK, min_periods=MIN_BETA_OBS).mean().shift(1)
    bm_var_prev = bm_ret.rolling(BETA_LOOKBACK, min_periods=MIN_BETA_OBS).var().shift(1)

    frames = []
    codes = sorted(members.code.unique())
    factor_cols = [f"rmom{w}" for w in RMOM_WINDOWS] + [f"ivol{w}" for w in IVOL_WINDOWS]

    for i, code in enumerate(codes, 1):
        mm = members[members.code == code]
        cols = {}
        for f in ["open", "high", "low", "close", "volume", "factor"]:
            s = base.qb.read_bin(code, f, cal)
            if not s.empty:
                cols[f] = s
        if not all(f in cols for f in ["open", "high", "low", "close", "volume"]):
            continue

        z = pd.concat(cols, axis=1).loc[WARM:END].copy()
        if z.empty:
            continue
        if "factor" not in z:
            z["factor"] = 1.0
        z["factor"] = z.factor.replace(0, np.nan).fillna(1.0)

        r = z.close.pct_change(fill_method=None)
        count120 = z.close.notna().rolling(MIN_LIST_DAYS).sum()
        liq20 = (z.close.abs() * z.volume.abs()).rolling(20).mean()

        m = bm_ret.reindex(z.index)
        s_mu_prev = r.rolling(BETA_LOOKBACK, min_periods=MIN_BETA_OBS).mean().shift(1)
        cov_prev = r.rolling(BETA_LOOKBACK, min_periods=MIN_BETA_OBS).cov(m).shift(1)
        beta = cov_prev / bm_var_prev.reindex(z.index)
        alpha = s_mu_prev - beta * bm_mu_prev.reindex(z.index)
        resid = r - alpha - beta * m

        fac = {}
        for w in RMOM_WINDOWS:
            n = w - SKIP_RECENT + 1
            fac[f"rmom{w}"] = resid.shift(SKIP_RECENT).rolling(
                n, min_periods=max(30, int(n * 0.80))
            ).sum()
        for w in IVOL_WINDOWS:
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
        valid &= np.asarray(sig["count120"] >= MIN_LIST_DAYS)
        valid &= np.isfinite(sig[["liq20"] + factor_cols].to_numpy()).all(axis=1)
        valid &= np.isfinite(ex[["open", "high", "low", "volume"]].to_numpy()).all(axis=1)
        if not valid.any():
            continue

        idx = np.flatnonzero(valid)
        rec = pd.DataFrame({
            "signal_date": signal_dates[idx],
            "trade_date": exec_dates[idx],
            "code": code,
            "liq20": sig["liq20"].to_numpy()[idx].astype(float),
            **{c: sig[c].to_numpy()[idx].astype(float) for c in factor_cols},
            "exec_open": ex["open"].to_numpy()[idx].astype(float),
            "exec_high": ex["high"].to_numpy()[idx].astype(float),
            "exec_low": ex["low"].to_numpy()[idx].astype(float),
            "exec_volume": ex["volume"].to_numpy()[idx].astype(float),
            "exec_factor": ex["factor"].to_numpy()[idx].astype(float),
        })
        frames.append(rec)
        if i % 500 == 0:
            print("alpha2f factor histories", i, "/", len(codes), flush=True)

    if not frames:
        raise RuntimeError("no alpha2f factor panel")
    p = pd.concat(frames, ignore_index=True)

    if not (pd.to_datetime(p.signal_date) < pd.to_datetime(p.trade_date)).all():
        raise RuntimeError("panel signal/trade timing violation")

    p["liq_rank_pct"] = p.groupby("signal_date")["liq20"].rank(
        pct=True, method="average", ascending=False
    )
    p = p[p.liq_rank_pct <= LIQ_KEEP_PCT].copy()

    for w in RMOM_WINDOWS:
        p[f"rmom{w}_pct"] = p.groupby("signal_date")[f"rmom{w}"].rank(
            pct=True, method="average", ascending=False
        )
    for w in IVOL_WINDOWS:
        p[f"ivol{w}_pct"] = p.groupby("signal_date")[f"ivol{w}"].rank(
            pct=True, method="average", ascending=True
        )
    for rw in RMOM_WINDOWS:
        for vw in IVOL_WINDOWS:
            s = 0.5 * (1 - p[f"rmom{rw}_pct"]) + 0.5 * (1 - p[f"ivol{vw}_pct"])
            p[f"score_{rw}_{vw}"] = s
            p[f"score_{rw}_{vw}_pct"] = s.groupby(p["signal_date"]).rank(
                pct=True, method="average", ascending=False
            )

    gs = p.groupby("signal_date").size()
    print(
        "PANEL", p.shape,
        "signal_dates", int(p.signal_date.nunique()),
        "group_min", int(gs.min()),
        "group_median", float(gs.median()),
        "group_max", int(gs.max()),
        flush=True,
    )
    if p.signal_date.nunique() < 480 or gs.median() < 1500 or gs.min() < 500:
        raise RuntimeError(
            f"FAIL-CLOSED abnormal eligible universe dates={p.signal_date.nunique()} "
            f"min={gs.min()} median={gs.median()} max={gs.max()}"
        )
    return p

@dataclass
class Pos:
    units: float
    entry_cost: float
    entry_date: pd.Timestamp
    last_price: float

def rank_col(variant: str) -> str:
    if variant == "rmom":
        return f"rmom{CORE_RMOM}_pct"
    if variant == "ivol":
        return f"ivol{CORE_IVOL}_pct"
    if variant == "2f":
        return f"score_{CORE_RMOM}_{CORE_IVOL}_pct"
    if variant.startswith("2f_"):
        _, rw, vw = variant.split("_")
        return f"score_{int(rw)}_{int(vw)}_pct"
    raise ValueError(variant)

def choose(g: pd.DataFrame, current: set[str], variant: str) -> list[str]:
    col = rank_col(variant)
    x = g.sort_values([col, "liq20"], ascending=[True, False]).copy()
    keep_set = set(x.loc[x[col] <= KEEP_PCT, "code"])
    keep = [c for c in current if c in keep_set][:N_HOLD]
    if len(keep) < N_HOLD:
        entrants = [c for c in x.loc[x[col] <= ENTRY_PCT, "code"] if c not in keep]
        keep.extend(entrants[: N_HOLD - len(keep)])
    return keep[:N_HOLD]

def fee(gross: float, side: str, d: pd.Timestamp, mult: float) -> float:
    return mult * base.fee(gross, side, d)

def simulate(panel: pd.DataFrame, variant: str, cal: pd.DatetimeIndex, members: pd.DataFrame,
             cost_mult: float = 1.0, daily_mtm: bool = True):
    by = {d: g.set_index("code", drop=False) for d, g in panel.groupby("signal_date")}
    dates = sorted(by)
    cash = INITIAL_CASH
    pos: dict[str, Pos] = {}
    equity, trades, timing = [], [], []
    turnover_notional = 0.0

    member_end = members.groupby("code").end.max().to_dict()
    close_cache = {}

    def close_series(code: str):
        if code not in close_cache:
            close_cache[code] = base.qb.read_bin(code, "close", cal).loc[START:END]
        return close_cache[code]

    trade_cal = cal[(cal >= START) & (cal <= END)]
    slip = SLIPPAGE * cost_mult

    for j, d in enumerate(dates):
        g = by[d]
        trade_date = pd.Timestamp(g.trade_date.iloc[0])
        target = choose(g.reset_index(drop=True), set(pos), variant)
        tgt = set(target)

        for c, pp in list(pos.items()):
            if c in g.index and np.isfinite(g.loc[c].exec_open):
                pp.last_price = float(g.loc[c].exec_open)
            elif pd.Timestamp(member_end.get(c, END)) < trade_date:
                old = pos.pop(c)
                trades.append({
                    "variant": variant, "code": c, "entry_date": old.entry_date,
                    "exit_date": trade_date, "net_pnl": -old.entry_cost,
                    "net_return": -1.0, "exit_reason": "membership_end_writeoff",
                })

        nav_open = cash + sum(pp.units * pp.last_price for pp in pos.values())

        for c in list(pos):
            if c in tgt:
                continue
            if c not in g.index:
                continue
            r = g.loc[c]
            locked = (
                np.isfinite(r.exec_open) and np.isfinite(r.exec_high) and np.isfinite(r.exec_low)
                and abs(float(r.exec_high) - float(r.exec_low)) < 1e-12
                and abs(float(r.exec_open) - float(r.exec_high)) < 1e-12
            )
            if locked:
                continue
            px = float(r.exec_open) * (1 - slip)
            gross = pos[c].units * px
            cost = fee(gross, "sell", trade_date, cost_mult)
            old = pos.pop(c)
            cash += gross - cost
            turnover_notional += gross
            trades.append({
                "variant": variant, "code": c, "entry_date": old.entry_date,
                "exit_date": trade_date, "net_pnl": gross - cost - old.entry_cost,
                "net_return": (gross - cost) / old.entry_cost - 1,
                "exit_reason": "rank_exit",
            })
            timing.append({
                "variant": variant, "signal_date": pd.Timestamp(d),
                "trade_date": trade_date, "side": "sell", "code": c,
            })

        per = nav_open * 0.99 / N_HOLD
        for c in target:
            if c in pos or c not in g.index:
                continue
            r = g.loc[c]
            locked = (
                np.isfinite(r.exec_open) and np.isfinite(r.exec_high) and np.isfinite(r.exec_low)
                and abs(float(r.exec_high) - float(r.exec_low)) < 1e-12
                and abs(float(r.exec_open) - float(r.exec_high)) < 1e-12
            )
            if locked:
                continue
            factor = float(r.exec_factor) if np.isfinite(r.exec_factor) and r.exec_factor > 0 else 1.0
            adjpx = float(r.exec_open) * (1 + slip)
            rawpx = adjpx / factor
            if rawpx <= 0:
                continue
            max_raw_by_volume = max(
                0, int(abs(float(r.exec_volume)) * factor * VOLUME_PARTICIPATION // 100) * 100
            )
            raw_shares = int(min(per, cash * 0.98) // (rawpx * 100)) * 100
            if max_raw_by_volume > 0:
                raw_shares = min(raw_shares, max_raw_by_volume)
            if raw_shares <= 0:
                continue
            units = raw_shares / factor
            gross = units * adjpx
            cost = fee(gross, "buy", trade_date, cost_mult)
            total = gross + cost
            if total > cash:
                continue
            cash -= total
            pos[c] = Pos(units, total, trade_date, float(r.exec_open))
            turnover_notional += gross
            timing.append({
                "variant": variant, "signal_date": pd.Timestamp(d),
                "trade_date": trade_date, "side": "buy", "code": c,
            })

        next_trade = (
            pd.Timestamp(by[dates[j + 1]].trade_date.iloc[0])
            if j + 1 < len(dates) else END + pd.Timedelta(days=1)
        )
        seg = trade_cal[(trade_cal >= trade_date) & (trade_cal < next_trade)]
        if not daily_mtm:
            seg = pd.DatetimeIndex([trade_date])
        for day in seg:
            for c, pp in pos.items():
                s = close_series(c)
                px = s.get(day, np.nan)
                if np.isfinite(px) and px > 0:
                    pp.last_price = float(px)
            nav = cash + sum(pp.units * pp.last_price for pp in pos.values())
            equity.append({
                "variant": variant, "signal_date": pd.Timestamp(d),
                "trade_date": pd.Timestamp(day), "equity": nav,
                "cash": cash, "positions": len(pos),
            })

    e = pd.DataFrame(equity).drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    t = pd.DataFrame(trades)
    tm = pd.DataFrame(timing)
    if len(tm):
        bad = tm[pd.to_datetime(tm.signal_date) >= pd.to_datetime(tm.trade_date)]
        if len(bad):
            raise RuntimeError(f"trade timing violations={len(bad)}")
    return e, t, tm, turnover_notional

def perf(eq: pd.DataFrame, tr: pd.DataFrame, turnover_notional: float, bench: pd.Series | None = None):
    s = eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float)
    r = s.pct_change().dropna()
    total = float(s.iloc[-1] / INITIAL_CASH - 1)
    years = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    cagr = float((s.iloc[-1] / INITIAL_CASH) ** (1 / years) - 1)
    dd = s / s.cummax() - 1
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    downside = r[r < 0].std()
    sortino = float(r.mean() / downside * np.sqrt(252)) if pd.notna(downside) and downside > 0 else np.nan

    wins = tr.loc[tr.net_pnl > 0, "net_pnl"].sum() if len(tr) else 0.0
    losses = tr.loc[tr.net_pnl < 0, "net_pnl"].sum() if len(tr) else 0.0
    pf = float(wins / abs(losses)) if losses < 0 else np.nan
    avg_win = tr.loc[tr.net_pnl > 0, "net_return"].mean() if len(tr) else np.nan
    avg_loss = tr.loc[tr.net_pnl < 0, "net_return"].mean() if len(tr) else np.nan
    payoff = float(avg_win / abs(avg_loss)) if pd.notna(avg_loss) and avg_loss < 0 else np.nan

    capm_beta = np.nan
    ann_alpha = np.nan
    if bench is not None:
        br = bench.pct_change(fill_method=None).reindex(r.index).dropna()
        rr = r.reindex(br.index).dropna()
        br = br.reindex(rr.index)
        if len(rr) > 100 and br.var() > 0:
            capm_beta = float(rr.cov(br) / br.var())
            alpha_d = float((rr - capm_beta * br).mean())
            ann_alpha = float((1 + alpha_d) ** 252 - 1)

    return {
        "final_asset": float(s.iloc[-1]), "total_return": total, "cagr": cagr,
        "max_drawdown": float(dd.min()), "sharpe": sharpe, "sortino": sortino,
        "trades": int(len(tr)), "win_rate": float((tr.net_pnl > 0).mean()) if len(tr) else np.nan,
        "profit_factor": pf, "payoff_ratio": payoff,
        "turnover_notional": float(turnover_notional),
        "turnover_over_initial": float(turnover_notional / INITIAL_CASH),
        "capm_beta": capm_beta, "annualized_capm_alpha": ann_alpha,
    }

def period_return(eq: pd.DataFrame, a: str, b: str) -> float:
    z = eq[(pd.to_datetime(eq.trade_date) >= pd.Timestamp(a)) &
           (pd.to_datetime(eq.trade_date) <= pd.Timestamp(b))]
    if len(z) < 2:
        return np.nan
    return float(z.equity.iloc[-1] / z.equity.iloc[0] - 1)

def annual_returns(eq: pd.DataFrame) -> pd.DataFrame:
    q = eq.copy()
    q["year"] = pd.to_datetime(q.trade_date).dt.year
    out = []
    for y, z in q.groupby("year"):
        out.append({"year": int(y), "return": float(z.equity.iloc[-1] / z.equity.iloc[0] - 1)})
    return pd.DataFrame(out)

def add_forward(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["code", "trade_date"]).copy()
    p["fwd5"] = p.groupby("code").exec_open.shift(-1) / p.exec_open - 1
    return p

def ic_stats(panel: pd.DataFrame):
    p = add_forward(panel)
    rows = []
    for d, g in p.groupby("signal_date"):
        z = g.dropna(subset=["fwd5"])
        if len(z) < 500:
            continue
        rows.append({
            "signal_date": d,
            "ic_rmom": z[f"rmom{CORE_RMOM}"].corr(z.fwd5, method="spearman"),
            "ic_ivol": (-z[f"ivol{CORE_IVOL}"]).corr(z.fwd5, method="spearman"),
            "ic_2f": z[f"score_{CORE_RMOM}_{CORE_IVOL}"].corr(z.fwd5, method="spearman"),
        })
    q = pd.DataFrame(rows)
    if q.empty:
        raise RuntimeError("no valid IC cross-sections")
    stats = []
    for c in ["ic_rmom", "ic_ivol", "ic_2f"]:
        x = q[c].dropna()
        stats.append({
            "factor": c, "mean_ic": float(x.mean()), "ic_std": float(x.std()),
            "icir": float(x.mean() / x.std() * np.sqrt(52)) if x.std() > 0 else np.nan,
            "positive_ic_rate": float((x > 0).mean()), "n": int(len(x)),
        })
    return q, pd.DataFrame(stats)

def quintiles(panel: pd.DataFrame) -> pd.DataFrame:
    p = add_forward(panel)
    specs = [
        ("rmom", f"rmom{CORE_RMOM}", True),
        ("ivol", f"ivol{CORE_IVOL}", False),
        ("2f", f"score_{CORE_RMOM}_{CORE_IVOL}", True),
    ]
    out = []
    for factor, col, ascending in specs:
        for d, g in p.groupby("signal_date"):
            z = g.dropna(subset=["fwd5", col]).copy()
            if len(z) < 500:
                continue
            rank = z[col].rank(pct=True, ascending=ascending, method="average")
            z["q"] = pd.cut(rank, [0, .2, .4, .6, .8, 1.0], labels=[1, 2, 3, 4, 5], include_lowest=True)
            for qn, zz in z.groupby("q", observed=True):
                out.append({
                    "factor": factor, "signal_date": d, "quintile": int(qn),
                    "mean_fwd": float(zz.fwd5.mean()),
                })
    raw = pd.DataFrame(out)
    return raw.groupby(["factor", "quintile"], as_index=False).mean(numeric_only=True)

def robustness(eq: pd.DataFrame, tr: pd.DataFrame) -> dict:
    z = eq.copy()
    z["ret"] = z.equity.pct_change()
    r = z.ret.dropna()
    def without_best(k: int) -> float:
        x = r.copy()
        idx = x.nlargest(min(k, len(x))).index
        x.loc[idx] = 0.0
        return float((1 + x).prod() - 1)
    pnl = float(tr.net_pnl.sum()) if len(tr) else 0.0
    best5 = float(tr.nlargest(min(5, len(tr)), "net_pnl").net_pnl.sum()) if len(tr) else 0.0
    return {
        "base_total_return": float(z.equity.iloc[-1] / INITIAL_CASH - 1),
        "return_without_best5_days": without_best(5),
        "return_without_best1pct_days": without_best(max(1, int(math.ceil(len(r) * 0.01)))),
        "completed_trade_pnl": pnl,
        "best5_trade_pnl": best5,
        "pnl_without_best5_trades": pnl - best5,
    }

def main():
    base.START = START
    base.WARM = WARM
    base.END = END
    base.OUT = OUT
    cal, members, ua = base.load_base()

    panel = build_panel(cal, members)
    panel.to_pickle(OUT / "panel_core.pkl")

    bm = base.qb.read_bin("SH000985", "close", cal).loc[START:END].dropna()
    benchmark_return = float(bm.iloc[-1] / bm.iloc[0] - 1)

    ic_ts, ic_sum = ic_stats(panel)
    ic_ts.to_csv(OUT / "ic_timeseries.csv", index=False)
    ic_sum.to_csv(OUT / "ic_summary.csv", index=False)
    q = quintiles(panel)
    q.to_csv(OUT / "quintiles.csv", index=False)

    summaries, all_eq, all_tr, all_tm, annual = [], [], [], [], []
    for v in ["rmom", "ivol", "2f"]:
        print("simulate", v, flush=True)
        eq, tr, tm, to = simulate(panel, v, cal, members, 1.0, daily_mtm=True)
        st = perf(eq, tr, to, bm)
        st["variant"] = v
        st["train_2016_2021_return"] = period_return(eq, "2016-07-29", "2021-12-31")
        st["sealed_2022_2026_return"] = period_return(eq, "2022-01-01", "2026-07-29")
        st["benchmark_return"] = benchmark_return
        st["excess_total_return"] = st["total_return"] - benchmark_return
        summaries.append(st)
        all_eq.append(eq)
        all_tr.append(tr)
        all_tm.append(tm)
        ar = annual_returns(eq)
        ar["variant"] = v
        annual.append(ar)

    print("simulate double cost", flush=True)
    eq2, tr2, tm2, to2 = simulate(panel, "2f", cal, members, 2.0, daily_mtm=True)
    stress = perf(eq2, tr2, to2, bm)
    stress["variant"] = "2f_double_all_costs"

    grid = []
    for rw in RMOM_WINDOWS:
        for vw in IVOL_WINDOWS:
            v = f"2f_{rw}_{vw}"
            print("grid", v, flush=True)
            eqg, trg, tmg, tog = simulate(panel, v, cal, members, 1.0, daily_mtm=False)
            stg = perf(eqg, trg, tog, None)
            stg["variant"] = v
            stg["sealed_2022_2026_return"] = period_return(eqg, "2022-01-01", "2026-07-29")
            grid.append(stg)
    grid_df = pd.DataFrame(grid)

    sm = pd.DataFrame(summaries)
    rob = pd.DataFrame([robustness(all_eq[2], all_tr[2])])
    all_timing = pd.concat(all_tm, ignore_index=True)
    gs = panel.groupby("signal_date").size()
    writeoffs = int(sum((t.exit_reason == "membership_end_writeoff").sum() for t in all_tr if len(t) and "exit_reason" in t))

    audit = {
        **ua,
        "benchmark": "SH000985", "benchmark_return": benchmark_return,
        "panel_rows": int(len(panel)), "signal_dates": int(panel.signal_date.nunique()),
        "eligible_min": int(gs.min()), "eligible_median": float(gs.median()), "eligible_max": int(gs.max()),
        "liquidity_rule": "top 80% by PIT 20d close*volume cross-sectional rank; unit-invariant",
        "core_rule": "0.5*rank(residual momentum 126..21)+0.5*rank(-residual vol 60)",
        "beta_rule": "rolling market-model alpha/beta 252d, coefficients shifted 1 day; min 126 obs",
        "portfolio_rule": "30 equal target sleeves; enter top10%; keep until below top30%; rebalance every 5 trading days",
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

    print("=== AUDIT ===")
    print(pd.DataFrame([audit]).to_string(index=False), flush=True)
    print("=== SUMMARY ===")
    print(sm.to_string(index=False), flush=True)
    print("=== IC ===")
    print(ic_sum.to_string(index=False), flush=True)
    print("=== QUINTILES ===")
    print(q.to_string(index=False), flush=True)
    print("=== DOUBLE COST ===")
    print(pd.DataFrame([stress]).to_string(index=False), flush=True)
    print("=== ROBUSTNESS ===")
    print(rob.to_string(index=False), flush=True)
    print("=== GRID ===")
    print(grid_df[["variant", "total_return", "cagr", "max_drawdown", "sharpe", "trades", "profit_factor", "sealed_2022_2026_return"]].to_string(index=False), flush=True)
    print("=== ANNUAL ===")
    print(pd.concat(annual, ignore_index=True).pivot(index="year", columns="variant", values="return").to_string(), flush=True)

if __name__ == "__main__":
    main()
