from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

import run_backtest_qlib as base

OUT = Path("results_qlib_audited")
OUT.mkdir(exist_ok=True)
base.OUT = OUT

@dataclass
class Pos:
    units: float
    entry_cost: float
    entry_date: pd.Timestamp
    entry_raw_price: float
    entry_signal_date: pd.Timestamp


def conservative_one_price_locked(row, prev_adj_close, side):
    """Fail closed on one-price moves of ~5% or more.

    The frozen Qlib bundle does not carry a point-in-time ST flag, so using only
    10%/20% board limits could falsely fill an ST one-price 5% limit. Blocking
    any one-price move >=4.5% is deliberately conservative: it may reject a
    rare executable non-ST print, but it will not create optimistic impossible
    fills from a 5% ST limit lock.
    """
    if row is None:
        return True
    vals = [row.get(c, np.nan) for c in ["open", "high", "low", "close"]]
    if not all(np.isfinite(v) for v in vals):
        return True
    one_price = abs(float(row.high) - float(row.low)) < 1e-8 and abs(float(row.open) - float(row.high)) < 1e-8
    if not one_price:
        return False
    if not np.isfinite(prev_adj_close) or prev_adj_close <= 0:
        return True
    pct = float(row.open) / float(prev_adj_close) - 1.0
    return (side == "buy" and pct >= 0.045) or (side == "sell" and pct <= -0.045)


def audited_backtest(panel, mdf, strategy):
    dates = sorted(pd.Timestamp(d) for d in panel.date.unique() if base.START <= pd.Timestamp(d) <= base.END)
    by = {pd.Timestamp(d): z.set_index("code", drop=False) for d, z in panel.groupby("date")}
    cash = base.INITIAL_CASH
    pos = {}
    target = None
    target_signal_date = None
    eq, trades, timing_events = [], [], []

    for k, d in enumerate(dates):
        day = by.get(d)
        if day is None:
            continue

        # Orders generated from the last signal can only execute on a later date.
        if target is not None:
            if target_signal_date is None or not (pd.Timestamp(target_signal_date) < pd.Timestamp(d)):
                raise RuntimeError(f"LOOKAHEAD VIOLATION: target signal {target_signal_date} executed on {d}")
            tgt = set(target)

            for code in list(pos):
                if code in tgt:
                    continue
                row = day.loc[code] if code in day.index else None
                prev_day = by.get(dates[k - 1]) if k > 0 else None
                prev_row = prev_day.loc[code] if prev_day is not None and code in prev_day.index else None
                prev_close = float(prev_row.close) if prev_row is not None else np.nan
                if row is None or conservative_one_price_locked(row, prev_close, "sell"):
                    continue
                adj_px = float(row.open) * (1 - base.SLIPPAGE)
                gross = pos[code].units * adj_px
                cost = base.fee(gross, "sell")
                cash += gross - cost
                p = pos.pop(code)
                pnl = gross - cost - p.entry_cost
                factor = float(row.factor) if np.isfinite(row.factor) and row.factor > 0 else 1.0
                raw_px = adj_px / factor
                trades.append({
                    "strategy": strategy, "code": code,
                    "entry_signal_date": p.entry_signal_date,
                    "entry_date": p.entry_date, "entry_raw_price": p.entry_raw_price,
                    "exit_signal_date": pd.Timestamp(target_signal_date),
                    "exit_date": d, "exit_raw_price": raw_px,
                    "net_pnl": pnl, "net_return": pnl / p.entry_cost,
                })
                timing_events.append({"strategy": strategy, "side": "sell", "signal_date": pd.Timestamp(target_signal_date), "trade_date": d})

            nav_open = cash
            for c, p in pos.items():
                if c in day.index and np.isfinite(day.loc[c].open):
                    nav_open += p.units * float(day.loc[c].open)
            per = min(nav_open * base.MAX_WEIGHT, nav_open / max(1, len(tgt)))

            for code in target:
                if code in pos or code not in day.index:
                    continue
                row = day.loc[code]
                prev_day = by.get(dates[k - 1]) if k > 0 else None
                prev_row = prev_day.loc[code] if prev_day is not None and code in prev_day.index else None
                prev_close = float(prev_row.close) if prev_row is not None else np.nan
                if conservative_one_price_locked(row, prev_close, "buy"):
                    continue
                factor = float(row.factor) if np.isfinite(row.factor) and row.factor > 0 else 1.0
                adj_px = float(row.open) * (1 + base.SLIPPAGE)
                raw_px = adj_px / factor
                if not np.isfinite(raw_px) or raw_px <= 0:
                    continue
                raw_shares = int(min(per, cash * 0.98) // (raw_px * 100)) * 100
                if raw_shares <= 0:
                    continue
                units = raw_shares / factor
                gross = units * adj_px
                cost = base.fee(gross, "buy")
                total = gross + cost
                if total > cash:
                    continue
                cash -= total
                pos[code] = Pos(units, total, d, raw_px, pd.Timestamp(target_signal_date))
                timing_events.append({"strategy": strategy, "side": "buy", "signal_date": pd.Timestamp(target_signal_date), "trade_date": d})

        nav = cash
        for code, p in pos.items():
            if code in day.index and np.isfinite(day.loc[code].close):
                nav += p.units * float(day.loc[code].close)
        eq.append({"date": d, "strategy": strategy, "equity": nav, "cash": cash, "n_positions": len(pos)})

        # All signal inputs below are known only at/after today's close.
        members = base.member_codes(mdf, d)
        universe = day[day.code.isin(members)].copy()
        valid = universe.dropna(subset=["close", "ma60"])
        risk_on = True if valid.empty else (valid.close > valid.ma60).mean() >= 0.45
        rebalance = strategy in {"trend_breakout", "mean_reversion", "volume_breakout"} or (k % 5 == 0)
        if rebalance:
            target = base.select(universe, strategy, risk_on)
            target_signal_date = d

    eq = pd.DataFrame(eq)
    tr = pd.DataFrame(trades)
    te = pd.DataFrame(timing_events)
    if not te.empty:
        te["signal_date"] = pd.to_datetime(te.signal_date)
        te["trade_date"] = pd.to_datetime(te.trade_date)
        bad = te[te.signal_date >= te.trade_date]
        if len(bad):
            raise RuntimeError(f"FAIL-CLOSED: {len(bad)} timing violations\n{bad.head()}")
    return eq, tr, te


def main():
    base.download_and_extract()
    cal = base.load_calendar()
    mdf = base.load_membership(cal)
    panel = base.build_panel(cal, mdf)
    bench = base.benchmark(cal)

    strategies = ["trend_breakout", "relative_momentum", "mean_reversion", "lowvol_trend", "volume_breakout", "multifactor"]
    summaries, alltr, alleq, allte = [], [], [], []
    for s in strategies:
        eq, tr, te = audited_backtest(panel, mdf, s)
        st = base.calc_stats(eq, tr)
        st["strategy"] = s
        if not bench.empty:
            bb = bench.reindex(eq.date, method="ffill").dropna()
            st["benchmark_return"] = float(bb.iloc[-1] - 1) if len(bb) else np.nan
            st["excess_return"] = st["total_return"] - st["benchmark_return"] if np.isfinite(st["benchmark_return"]) else np.nan
        summaries.append(st); alltr.append(tr); alleq.append(eq); allte.append(te)

    sm = pd.DataFrame(summaries).sort_values("total_return", ascending=False)
    tr = pd.concat(alltr, ignore_index=True) if alltr else pd.DataFrame()
    eq = pd.concat(alleq, ignore_index=True)
    te = pd.concat(allte, ignore_index=True) if allte else pd.DataFrame()

    violations = 0
    min_lag = np.nan
    if not te.empty:
        te["lag_days"] = (pd.to_datetime(te.trade_date) - pd.to_datetime(te.signal_date)).dt.days
        violations = int((te.lag_days <= 0).sum())
        min_lag = int(te.lag_days.min())
    if violations:
        raise RuntimeError(f"FAIL-CLOSED: timing violations={violations}")
    pd.DataFrame([{
        "timing_events": len(te), "timing_violations": violations,
        "minimum_calendar_lag_days": min_lag,
        "execution_rule": "signal_at_T_close_execute_no_earlier_than_later_trading_day_open",
        "limit_lock_rule": "conservative_block_all_one_price_moves_ge_4.5pct_abs",
    }]).to_csv(OUT / "timing_audit.csv", index=False)

    sm.to_csv(OUT / "summary.csv", index=False)
    tr.to_csv(OUT / "trades.csv", index=False)
    eq.to_csv(OUT / "equity.csv", index=False)
    te.to_csv(OUT / "timing_events.csv", index=False)

    mid = base.START + (base.END - base.START) / 2
    seg = []
    for s in strategies:
        e = eq[eq.strategy == s].set_index("date").equity
        for name, a, z in [("H1", base.START, mid), ("H2", mid, base.END)]:
            x = e[(e.index >= a) & (e.index <= z)]
            if len(x) > 1:
                seg.append({"strategy": s, "segment": name, "return": x.iloc[-1] / x.iloc[0] - 1})
    pd.DataFrame(seg).to_csv(OUT / "segments.csv", index=False)

    robust = []
    for s in strategies:
        t = tr[tr.strategy == s].copy() if not tr.empty else pd.DataFrame()
        total_net = t.net_pnl.sum() if len(t) else 0.0
        drop5 = t.nlargest(min(5, len(t)), "net_pnl").net_pnl.sum() if len(t) else 0.0
        robust.append({"strategy": s, "completed_trade_net_pnl": total_net, "best5_net_pnl": drop5, "trade_pnl_after_removing_best5": total_net - drop5})
    pd.DataFrame(robust).to_csv(OUT / "robustness.csv", index=False)

    print("\n=== TIMING AUDIT ===")
    print(pd.read_csv(OUT / "timing_audit.csv").to_string(index=False), flush=True)
    print("\n=== FINAL AUDITED LEADERBOARD ===")
    print(sm.to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
