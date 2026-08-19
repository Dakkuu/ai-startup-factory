from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import run_10y_era_backtest as base
import run_10y_era_backtest_fast as fast  # patches no-future panel builder + execution-safe selection

TRADE_START = pd.Timestamp('2016-07-29')
TRAIN_START = pd.Timestamp('2014-07-29')
END = pd.Timestamp('2026-07-29')
OUT = Path('results_10y_ridge')
OUT.mkdir(exist_ok=True)

# Build enough pre-history so Ridge is genuinely used from the first live trade date.
base.START = TRAIN_START
base.WARM = TRAIN_START - pd.Timedelta(days=260)
base.END = END
base.OUT = OUT

class AlwaysRidge:
    def __init__(self, panel: pd.DataFrame):
        self.p = panel.copy()
        self.cache = {}
        self.audit = []

    def score(self, d, current):
        d = pd.Timestamp(d)
        key = (d.year, (d.month - 1) // 3)
        model = self.cache.get(key)
        if model is None:
            train = self.p[
                (self.p.signal_date >= d - pd.DateOffset(years=2)) &
                (self.p.label_exit_date < d) &
                self.p.label.notna()
            ].copy()
            train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=base.FEATURES + ['label'])
            if len(train) > 120000:
                train = train.sample(120000, random_state=d.year * 100 + d.month)
            if len(train) < 5000:
                raise RuntimeError(f'insufficient matured Ridge training data {d.date()} rows={len(train)}')
            X = train[base.FEATURES].to_numpy()
            y = train.label.clip(-0.30, 0.30).to_numpy()
            model = Ridge(alpha=8.0).fit(X, y)
            self.cache[key] = model
            max_exit = pd.to_datetime(train.label_exit_date).max()
            violation = not (max_exit < d)
            self.audit.append({
                'train_asof': d,
                'train_rows': len(train),
                'train_window_years': 2,
                'max_label_exit_date': max_exit,
                'timing_violation': int(violation),
            })
            if violation:
                raise RuntimeError('LOOKAHEAD VIOLATION in Ridge training')
        return model.predict(current[base.FEATURES].to_numpy())


def main():
    cal, members, universe_audit = base.load_base()
    panel_all, _ = fast.build_weekly_panel_fast(cal, members)
    quant = AlwaysRidge(panel_all)

    # Live portfolio begins only at 2016-07-29; 2014-2016 data is training history only.
    panel = panel_all[pd.to_datetime(panel_all.signal_date) >= TRADE_START].copy()
    states = base.benchmark_state(cal, panel)
    agent = {'strategy':'04_era_appropriate_quant','risk':'mid','max_names':30,'invest':0.90}

    eq, tr, tm = base.simulate(panel, states, agent, quant)
    st = base.stats(eq, tr)
    st.update({'strategy':'always_ridge_2y_rolling','risk':'mid'})

    bench = base.qb.read_bin('SH000985','close',cal).loc[TRADE_START:END].dropna()
    benchmark_return = float(bench.iloc[-1] / bench.iloc[0] - 1)
    st['benchmark_return'] = benchmark_return
    st['excess_return'] = st['total_return'] - benchmark_return

    annual = []
    for y, z in eq.groupby(pd.to_datetime(eq.trade_date).dt.year):
        annual.append({'year':int(y), 'return':float(z.equity.iloc[-1] / z.equity.iloc[0] - 1)})

    completed_pnl = float(tr.net_pnl.sum()) if len(tr) else 0.0
    best10 = float(tr.nlargest(min(10, len(tr)), 'net_pnl').net_pnl.sum()) if len(tr) else 0.0
    robustness = {
        'completed_trade_pnl': completed_pnl,
        'best10_pnl': best10,
        'pnl_without_best10': completed_pnl - best10,
    }

    timing_viol = int((pd.to_datetime(tm.signal_date) >= pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    model_viol = int(sum(x['timing_violation'] for x in quant.audit))
    min_lag = float((pd.to_datetime(tm.trade_date) - pd.to_datetime(tm.signal_date)).dt.total_seconds().min() / 86400) if len(tm) else np.nan
    audit = {
        **universe_audit,
        'training_history_start': str(TRAIN_START.date()),
        'live_trade_start': str(TRADE_START.date()),
        'live_trade_end': str(END.date()),
        'weekly_panel_rows_all': len(panel_all),
        'weekly_panel_rows_live': len(panel),
        'signal_dates_live': int(panel.signal_date.nunique()),
        'trade_timing_violations': timing_viol,
        'model_timing_violations': model_viol,
        'min_trade_lag_days': min_lag,
        'model_rule': 'Always rolling Ridge(alpha=8), trailing 2 years only; quarterly refit; labels must fully mature before training; 12 cross-sectional price/volume features',
        'execution_rule': 'weekly close signal -> next exchange open; no next-day availability used for ranking; failed suspension/limit fills remain cash; historical fees; 10bp slippage; 5% volume participation',
    }
    if timing_viol or model_viol:
        raise RuntimeError(f'audit failed trade={timing_viol} model={model_viol}')

    pd.DataFrame([audit]).to_csv(OUT/'audit.csv', index=False)
    pd.DataFrame([st]).to_csv(OUT/'summary.csv', index=False)
    eq.to_csv(OUT/'equity.csv', index=False)
    tr.to_csv(OUT/'trades.csv', index=False)
    tm.to_csv(OUT/'timing_events.csv', index=False)
    pd.DataFrame(quant.audit).to_csv(OUT/'model_training_audit.csv', index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual_returns.csv', index=False)
    pd.DataFrame([robustness]).to_csv(OUT/'robustness.csv', index=False)

    print('=== AUDIT ===')
    print(pd.DataFrame([audit]).to_string(index=False), flush=True)
    print('=== SUMMARY ===')
    print(pd.DataFrame([st]).to_string(index=False), flush=True)
    print('=== ANNUAL ===')
    print(pd.DataFrame(annual).to_string(index=False), flush=True)
    print('=== ROBUSTNESS ===')
    print(pd.DataFrame([robustness]).to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
