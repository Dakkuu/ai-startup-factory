from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import run_10y_alpha_stage2 as s2
import run_10y_alpha_composites_qv as comp
import run_10y_stage2_frozen_audit as fa
import run_10y_alpha2f_v2 as sim

OUT = Path('results_phase_sleeve_ensemble')
OUT.mkdir(exist_ok=True)
PHASES = tuple(range(8))


def combine_curves(curves: list[pd.DataFrame], cal: pd.DatetimeIndex, initial_cash: float = 1_000_000.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(cal[(cal >= sim.START) & (cal <= sim.END)])
    cols = []
    for i, eq in enumerate(curves):
        s = eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float)
        # Each standalone phase is simulated from the same initial capital.  An equal-capital
        # 8-sleeve portfolio therefore has equity equal to the arithmetic mean of the eight
        # standalone equity curves.  Before a phase begins trading, its sleeve remains cash.
        s = s.reindex(idx).ffill().fillna(initial_cash)
        cols.append(s.rename(f'phase_{i}'))
    m = pd.concat(cols, axis=1)
    out = pd.DataFrame({'trade_date': idx, 'equity': m.mean(axis=1).to_numpy()})
    return out


def eq_stats(eq: pd.DataFrame, bench: pd.Series) -> dict:
    empty_tr = pd.DataFrame(columns=['net_pnl', 'net_return'])
    st = sim.perf(eq, empty_tr, 0.0, bench)
    st['train_return'] = sim.period_return(eq, '2016-07-29', '2021-12-31')
    st['pseudo_oos_return'] = sim.period_return(eq, '2022-01-01', '2026-07-29')
    return st


def run_ensemble(rq, cal, members, bm, cost: float = 1.0):
    curves = []
    phase_rows = []
    for ph in PHASES:
        st, eq, _, _ = fa.run(rq, cal, members, bm, phase=ph, cost=cost)
        curves.append(eq)
        phase_rows.append(st)
    ens = combine_curves(curves, cal)
    st = eq_stats(ens, bm)
    st['cost_mult'] = cost
    st['phase_count'] = len(PHASES)
    st['construction'] = 'equal_weight_8_phase_sleeves_no_phase_selection'
    return st, ens, pd.DataFrame(phase_rows)


def main():
    p, cal, members, ua, market_code, bm = s2.build_panel(OUT)
    p = comp.add_composite_scores(p)
    rq = comp.ranked(p, 'anti_lottery_momentum')

    base, eq, phases = run_ensemble(rq, cal, members, bm, 1.0)
    pd.DataFrame([base]).to_csv(OUT / 'ensemble.csv', index=False)
    phases.to_csv(OUT / 'phase_components.csv', index=False)
    eq.to_csv(OUT / 'ensemble_equity.csv', index=False)
    sim.annual_returns(eq).to_csv(OUT / 'annual.csv', index=False)
    pd.DataFrame([sim.robustness(eq, pd.DataFrame(columns=['net_pnl','net_return']))]).to_csv(OUT / 'robust.csv', index=False)

    costs = []
    for cm in (2.0, 4.0, 8.0):
        st, _, _ = run_ensemble(rq, cal, members, bm, cm)
        costs.append(st)
    pd.DataFrame(costs).to_csv(OUT / 'costs.csv', index=False)

    # Structural gates are intentionally stricter than "all phases positive": the ensemble
    # must beat the frozen diversified baseline on total return without worsening drawdown
    # materially, and must retain positive pseudo-OOS and 8x-cost performance.
    frozen_baseline_total = 1.749407
    frozen_baseline_mdd = -0.2505
    c8 = pd.DataFrame(costs).query('cost_mult == 8.0').iloc[0]
    gates = pd.DataFrame([
        {'gate':'ensemble_positive', 'pass':int(base['total_return'] > 0)},
        {'gate':'pseudo_oos_positive', 'pass':int(base['pseudo_oos_return'] > 0)},
        {'gate':'beats_frozen_baseline_total_return', 'pass':int(base['total_return'] > frozen_baseline_total)},
        {'gate':'mdd_not_worse_than_baseline_by_10pp', 'pass':int(base['max_drawdown'] >= frozen_baseline_mdd - 0.10)},
        {'gate':'cost8_positive', 'pass':int(c8['total_return'] > 0)},
    ])
    gates.to_csv(OUT / 'gates.csv', index=False)

    verdict = {
        **ua,
        'market_factor': market_code,
        'rule': 'anti_lottery_momentum; N8; hold40; entry10%; keep30%; equal 8 phase sleeves; next-open',
        'ensemble_total_return': base['total_return'],
        'ensemble_cagr': base['cagr'],
        'ensemble_mdd': base['max_drawdown'],
        'ensemble_sharpe': base['sharpe'],
        'train_return': base['train_return'],
        'pseudo_oos_return': base['pseudo_oos_return'],
        'phase_total_return_min': float(phases.total_return.min()),
        'phase_total_return_median': float(phases.total_return.median()),
        'phase_total_return_max': float(phases.total_return.max()),
        'gates_passed': int(gates['pass'].sum()),
        'gates_total': int(len(gates)),
        'hard_pass': int(gates['pass'].all()),
        'signal_universe': 'signal-pure T-only',
        'volume_source_unit_shares': 100,
    }
    pd.DataFrame([verdict]).to_csv(OUT / 'verdict.csv', index=False)
    print('=== PHASE-SLEEVE ENSEMBLE ===')
    print(pd.DataFrame([verdict]).to_string(index=False), flush=True)
    print('=== COSTS ===')
    print(pd.DataFrame(costs)[['cost_mult','total_return','cagr','max_drawdown','sharpe','pseudo_oos_return']].to_string(index=False), flush=True)
    print('=== GATES ===')
    print(gates.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
