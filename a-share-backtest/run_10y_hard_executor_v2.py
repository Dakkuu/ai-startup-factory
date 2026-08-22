from __future__ import annotations
import numpy as np, pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_skewfilter_hard as old

VOLUME_SOURCE_UNIT_SHARES = 100.0


def raw_share_volume(exec_volume: float, factor: float) -> float:
    if not np.isfinite(exec_volume) or not np.isfinite(factor) or factor <= 0:
        return np.nan
    return abs(float(exec_volume)) * float(factor) * VOLUME_SOURCE_UNIT_SHARES


def max_participation_shares(exec_volume: float, factor: float, participation: float) -> int:
    rawvol = raw_share_volume(exec_volume, factor)
    if not np.isfinite(rawvol) or rawvol <= 0 or participation <= 0:
        return 0
    return max(0, int((rawvol * float(participation)) // 100) * 100)


def quote_available(r) -> bool:
    return bool(np.isfinite([r.exec_open, r.exec_high, r.exec_low, r.exec_volume]).all() and float(r.exec_open) > 0 and float(r.exec_volume) > 0)


def row_trade_allowed(r) -> bool:
    if 'exec_can_trade' in r.index and not bool(r.exec_can_trade):
        return False
    return quote_available(r)


def row_buy_allowed(r) -> bool:
    if not row_trade_allowed(r):
        return False
    if 'exec_buy_allowed' in r.index and not bool(r.exec_buy_allowed):
        return False
    return True


def hard_simulate(panel, cal, members, cost_mult=1.0):
    by = {d: g.set_index('code', drop=False) for d, g in panel.groupby('signal_date')}
    dates = sorted(by)
    cash = sim.INITIAL_CASH; pos = {}; equity = []; trades = []; timing = []; turnover = 0.0
    member_end = members.groupby('code').end.max().to_dict(); close_cache = {}
    def close_series(code):
        if code not in close_cache:
            close_cache[code] = base.qb.read_bin(code, 'close', cal).loc[sim.START:sim.END]
        return close_cache[code]
    trade_cal = cal[(cal >= sim.START) & (cal <= sim.END)]; slip = sim.SLIPPAGE * cost_mult
    for j, d in enumerate(dates):
        g = by[d]; td = pd.Timestamp(g.trade_date.iloc[0]); target = old.choose_det(g.reset_index(drop=True), set(pos)); tgt = set(target)
        for c, pp in list(pos.items()):
            if c in g.index and np.isfinite(g.loc[c].exec_open):
                pp.last_price = float(g.loc[c].exec_open)
            elif pd.Timestamp(member_end.get(c, sim.END)) < td:
                oldp = pos.pop(c); trades.append({'variant':'hard_v2','code':c,'entry_date':oldp.entry_date,'exit_date':td,'net_pnl':-oldp.entry_cost,'net_return':-1.0,'exit_reason':'membership_end_writeoff'})
        nav_open = cash + sum(pp.units * pp.last_price for pp in pos.values())
        for c in sorted(list(pos)):
            if c in tgt or c not in g.index: continue
            r = g.loc[c]
            if not row_trade_allowed(r): continue
            locked = abs(float(r.exec_high)-float(r.exec_low)) < 1e-12 and abs(float(r.exec_open)-float(r.exec_high)) < 1e-12
            if locked: continue
            px = float(r.exec_open) * (1-slip); gross = pos[c].units * px; cost = sim.fee(gross, 'sell', td, cost_mult); oldp = pos.pop(c); cash += gross-cost; turnover += gross
            trades.append({'variant':'hard_v2','code':c,'entry_date':oldp.entry_date,'exit_date':td,'net_pnl':gross-cost-oldp.entry_cost,'net_return':(gross-cost)/oldp.entry_cost-1,'exit_reason':'rank_exit'})
            timing.append({'variant':'hard_v2','signal_date':pd.Timestamp(d),'trade_date':td,'side':'sell','code':c,'gross':gross})
        per = nav_open * .99 / old.N_HOLD
        for c in target:
            if len(pos) >= old.N_HOLD: break
            if c in pos or c not in g.index: continue
            r = g.loc[c]
            if not row_buy_allowed(r): continue
            locked = abs(float(r.exec_high)-float(r.exec_low)) < 1e-12 and abs(float(r.exec_open)-float(r.exec_high)) < 1e-12
            if locked: continue
            if not np.isfinite(r.exec_factor) or float(r.exec_factor) <= 0: continue
            factor = float(r.exec_factor); adjpx = float(r.exec_open) * (1+slip); rawpx = adjpx / factor
            if not np.isfinite(rawpx) or rawpx <= 0: continue
            rawvol = raw_share_volume(float(r.exec_volume), factor); maxraw = max_participation_shares(float(r.exec_volume), factor, sim.VOLUME_PARTICIPATION)
            shares = int(min(per, cash*.98) // (rawpx*100)) * 100; shares = min(shares, maxraw)
            if shares <= 0: continue
            units = shares/factor; gross = units*adjpx; cost = sim.fee(gross, 'buy', td, cost_mult); total = gross+cost
            if total > cash: continue
            cash -= total; pos[c] = sim.Pos(units, total, td, float(r.exec_open)); turnover += gross
            actual_part = float(shares/rawvol) if np.isfinite(rawvol) and rawvol > 0 else np.nan
            timing.append({'variant':'hard_v2','signal_date':pd.Timestamp(d),'trade_date':td,'side':'buy','code':c,'gross':gross,'raw_shares':shares,'raw_daily_share_volume':rawvol,'actual_participation':actual_part})
        if len(pos) > old.N_HOLD:
            raise RuntimeError(f'position cap violation {len(pos)}')
        next_td = pd.Timestamp(by[dates[j+1]].trade_date.iloc[0]) if j+1 < len(dates) else sim.END + pd.Timedelta(days=1)
        seg = trade_cal[(trade_cal >= td) & (trade_cal < next_td)]
        for day in seg:
            for c, pp in pos.items():
                px = close_series(c).get(day, np.nan)
                if np.isfinite(px) and px > 0: pp.last_price = float(px)
            nav = cash + sum(pp.units*pp.last_price for pp in pos.values()); equity.append({'variant':'hard_v2','signal_date':pd.Timestamp(d),'trade_date':pd.Timestamp(day),'equity':nav,'cash':cash,'positions':len(pos)})
    e = pd.DataFrame(equity).drop_duplicates('trade_date', keep='last').sort_values('trade_date'); t = pd.DataFrame(trades); tm = pd.DataFrame(timing)
    if len(tm) and (pd.to_datetime(tm.signal_date) >= pd.to_datetime(tm.trade_date)).any():
        raise RuntimeError('timing violation')
    if len(tm) and 'actual_participation' in tm:
        ap = pd.to_numeric(tm.loc[tm.side=='buy','actual_participation'], errors='coerce').dropna()
        if len(ap) and (ap > sim.VOLUME_PARTICIPATION + 1e-12).any():
            raise RuntimeError(f'participation violation max={ap.max()} limit={sim.VOLUME_PARTICIPATION}')
    return e, t, tm, turnover


def patch():
    old.hard_simulate = hard_simulate
    return old

if __name__ == '__main__':
    print({'volume_source_unit_shares': VOLUME_SOURCE_UNIT_SHARES})
