from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4

OUT = Path('results_combo_rotation')
OUT.mkdir(exist_ok=True)

# Pre-registered medium/long rotation horizons: about 1/2/3/6 months.
HOLD_DAYS = (20, 40, 60, 120)
# Pre-registered equal-rank combinations. No fitted weights.
CORE_VARIANTS = ('ivol60', 'ivol_price', 'ivol_amount', 'price_amount', 'ivol_price_amount')
POSTHOC_VARIANTS = ('ivol_pricefloor3', 'ivol_amount_pricefloor3')


def add_raw_price(panel: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    panel = panel.copy()
    panel['raw_price'] = np.nan
    groups = panel.groupby('code').groups
    for i, (code, idx) in enumerate(groups.items(), 1):
        close = base.qb.read_bin(code, 'close', cal)
        factor = base.qb.read_bin(code, 'factor', cal)
        if close.empty or factor.empty:
            continue
        raw = close / factor.replace(0, np.nan)
        ds = pd.DatetimeIndex(panel.loc[idx, 'signal_date'])
        panel.loc[idx, 'raw_price'] = raw.reindex(ds).to_numpy(dtype=float)
        if i % 1000 == 0:
            print('raw-price histories', i, '/', len(groups), flush=True)
    return panel


def add_rank_columns(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    liq_ok = (p['liq_rank_pct'] <= sim.LIQ_KEEP_PCT)

    p['price_pct'] = np.nan
    m = liq_ok & np.isfinite(p['raw_price']) & (p['raw_price'] > 0)
    p.loc[m, 'price_pct'] = p.loc[m].groupby('signal_date')['raw_price'].rank(
        pct=True, method='average', ascending=True
    )

    p['amount_pct'] = np.nan
    m = liq_ok & np.isfinite(p['liq20']) & (p['liq20'] > 0)
    p.loc[m, 'amount_pct'] = p.loc[m].groupby('signal_date')['liq20'].rank(
        pct=True, method='average', ascending=True
    )

    # Keep the original low-IVOL60 rank untouched as the disciplined core baseline.
    p['rank_ivol60'] = p['ivol60_pct']

    def composite(name: str, cols: list[str], extra_mask=None):
        valid = liq_ok & p[cols].notna().all(axis=1)
        if extra_mask is not None:
            valid &= extra_mask
        tmp = p.loc[valid, cols].mean(axis=1)
        p[name] = np.nan
        p.loc[valid, name] = tmp.groupby(p.loc[valid, 'signal_date']).rank(
            pct=True, method='average', ascending=True
        )

    composite('rank_ivol_price', ['rank_ivol60', 'price_pct'])
    composite('rank_ivol_amount', ['rank_ivol60', 'amount_pct'])
    composite('rank_price_amount', ['price_pct', 'amount_pct'])
    composite('rank_ivol_price_amount', ['rank_ivol60', 'price_pct', 'amount_pct'])

    # Explicitly post-hoc safety diagnostics from the immediately preceding low-price experiment.
    floor3 = np.isfinite(p['raw_price']) & (p['raw_price'] >= 3.0)
    composite('rank_ivol_pricefloor3', ['rank_ivol60'], floor3)
    composite('rank_ivol_amount_pricefloor3', ['rank_ivol60', 'amount_pct'], floor3)

    return p


def rank_column(variant: str) -> str:
    return {
        'ivol60': 'rank_ivol60',
        'ivol_price': 'rank_ivol_price',
        'ivol_amount': 'rank_ivol_amount',
        'price_amount': 'rank_price_amount',
        'ivol_price_amount': 'rank_ivol_price_amount',
        'ivol_pricefloor3': 'rank_ivol_pricefloor3',
        'ivol_amount_pricefloor3': 'rank_ivol_amount_pricefloor3',
    }[variant]


def subset_for_rotation(panel: pd.DataFrame, variant: str, hold_days: int) -> pd.DataFrame:
    if hold_days % 5 != 0:
        raise ValueError('hold_days must be a multiple of the 5-trading-day base signal grid')
    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel['signal_date'].unique())))
    step = hold_days // 5
    chosen = set(all_dates[::step])
    col = rank_column(variant)
    need = ['signal_date', 'trade_date', 'code', 'liq20', 'exec_open', 'exec_high',
            'exec_low', 'exec_volume', 'exec_factor', col]
    q = panel.loc[panel['signal_date'].isin(chosen), need].copy()
    q['ivol60_pct'] = q[col]
    return q.drop(columns=[col]) if col != 'ivol60_pct' else q


def run_one(panel, variant, hold_days, cal, members, benchmark, cost_mult=1.0, daily=True):
    q = subset_for_rotation(panel, variant, hold_days)
    eq, tr, tm, turnover = sim.simulate(q, 'ivol', cal, members, cost_mult, daily_mtm=daily)
    st = sim.perf(eq, tr, turnover, benchmark if daily else None)
    st['variant'] = variant
    st['hold_days'] = hold_days
    st['cost_mult'] = cost_mult
    st['train_2016_2021_return'] = sim.period_return(eq, '2016-07-29', '2021-12-31')
    st['sealed_2022_2026_return'] = sim.period_return(eq, '2022-01-01', '2026-07-29')
    st['positions_max'] = int(eq.positions.max()) if len(eq) else 0
    st['positions_median'] = float(eq.positions.median()) if len(eq) else 0.0
    return st, eq, tr, tm


def main():
    base.START = sim.START; base.WARM = sim.WARM; base.END = sim.END; base.OUT = OUT
    v4.OUT = OUT

    cal, members, ua = base.load_base()
    market_code, market_close, market_cov = v4.pick_market(cal)
    panel = v4.build_panel(cal, members, market_close)
    panel = add_raw_price(panel, cal)
    panel = add_rank_columns(panel)

    benchmark = market_close.loc[sim.START:sim.END].dropna()
    benchmark_return = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)

    # Apples-to-apples 5-day core IVOL reference.
    baseline, base_eq, base_tr, base_tm = run_one(
        panel, 'ivol60', 5, cal, members, benchmark, 1.0, True
    )
    baseline['benchmark_return'] = benchmark_return
    pd.DataFrame([baseline]).to_csv(OUT / 'baseline_5d.csv', index=False)

    rows = []
    eq_cache = {}
    tr_cache = {}
    tm_cache = {}
    for variant in CORE_VARIANTS + POSTHOC_VARIANTS:
        for hold in HOLD_DAYS:
            print('RUN', variant, 'hold', hold, flush=True)
            st, eq, tr, tm = run_one(panel, variant, hold, cal, members, benchmark, 1.0, True)
            st['posthoc'] = variant in POSTHOC_VARIANTS
            st['benchmark_return'] = benchmark_return
            st['excess_total_return'] = st['total_return'] - benchmark_return
            rows.append(st)
            eq_cache[(variant, hold)] = eq
            tr_cache[(variant, hold)] = tr
            tm_cache[(variant, hold)] = tm

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / 'grid.csv', index=False)

    # Winner selection uses TRAIN ONLY and excludes the explicitly post-hoc price-floor variants.
    eligible = grid[~grid.posthoc].copy()
    win_row = eligible.sort_values(
        ['train_2016_2021_return', 'max_drawdown'], ascending=[False, False]
    ).iloc[0]
    winner = str(win_row['variant']); winner_hold = int(win_row['hold_days'])
    print('TRAIN-ONLY WINNER', winner, winner_hold, flush=True)

    winner_eq = eq_cache[(winner, winner_hold)]
    winner_tr = tr_cache[(winner, winner_hold)]
    winner_tm = tm_cache[(winner, winner_hold)]

    cost_rows = []
    for cm in (2.0, 4.0):
        print('COST', winner, winner_hold, cm, flush=True)
        st, _, _, _ = run_one(panel, winner, winner_hold, cal, members, benchmark, cm, True)
        cost_rows.append(st)
    pd.DataFrame(cost_rows).to_csv(OUT / 'winner_cost_stress.csv', index=False)

    robust = sim.robustness(winner_eq, winner_tr)
    robust['variant'] = winner; robust['hold_days'] = winner_hold
    pd.DataFrame([robust]).to_csv(OUT / 'winner_robustness.csv', index=False)

    annual = sim.annual_returns(winner_eq)
    annual['variant'] = winner; annual['hold_days'] = winner_hold
    annual.to_csv(OUT / 'winner_annual.csv', index=False)

    stability = grid.groupby(['variant', 'posthoc']).agg(
        median_total_return=('total_return', 'median'),
        min_total_return=('total_return', 'min'),
        max_total_return=('total_return', 'max'),
        median_train=('train_2016_2021_return', 'median'),
        median_sealed=('sealed_2022_2026_return', 'median'),
        positive_sealed_count=('sealed_2022_2026_return', lambda x: int((x > 0).sum())),
        median_mdd=('max_drawdown', 'median'),
        median_turnover=('turnover_over_initial', 'median'),
    ).reset_index()
    stability.to_csv(OUT / 'variant_stability.csv', index=False)

    all_tm = pd.concat([x.assign(test_variant=k[0], test_hold_days=k[1])
                        for k, x in tm_cache.items() if len(x)], ignore_index=True)
    timing_bad = int((pd.to_datetime(all_tm.signal_date) >= pd.to_datetime(all_tm.trade_date)).sum()) if len(all_tm) else 0
    audit = {
        **ua,
        'market_factor': market_code,
        'market_factor_coverage': float(market_cov.loc[market_cov.code == market_code, 'coverage'].iloc[0]),
        'benchmark_return': benchmark_return,
        'panel_rows': len(panel),
        'signal_dates_5d': int(panel.signal_date.nunique()),
        'core_variants': '|'.join(CORE_VARIANTS),
        'posthoc_variants': '|'.join(POSTHOC_VARIANTS),
        'hold_days_grid': '|'.join(map(str, HOLD_DAYS)),
        'selection_rule': 'highest 2016-2021 return among pre-registered variants only; 2022-2026 sealed not used for selection',
        'weights': 'equal percentile-rank weights only; no fitted weights',
        'portfolio': '30 equal sleeves; enter top10pct; retain through top30pct; next-open execution',
        'timing_violations': timing_bad,
    }
    if timing_bad:
        raise RuntimeError(f'timing violations {timing_bad}')
    pd.DataFrame([audit]).to_csv(OUT / 'audit.csv', index=False)

    print('=== BASELINE 5D ==='); print(pd.DataFrame([baseline]).to_string(index=False), flush=True)
    print('=== GRID ==='); print(grid.sort_values(['posthoc','train_2016_2021_return'], ascending=[True,False]).to_string(index=False), flush=True)
    print('=== STABILITY ==='); print(stability.to_string(index=False), flush=True)
    print('=== WINNER COST ==='); print(pd.DataFrame(cost_rows).to_string(index=False), flush=True)
    print('=== WINNER ROBUSTNESS ==='); print(pd.DataFrame([robust]).to_string(index=False), flush=True)
    print('=== WINNER ANNUAL ==='); print(annual.to_string(index=False), flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
