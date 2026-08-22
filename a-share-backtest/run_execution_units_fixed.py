from __future__ import annotations
"""Execution-unit compatibility layer for chenditc/Tushare Qlib data.

The upstream Qlib `volume` field is sourced from Tushare `vol`, whose unit is lots
(手, 100 shares). The legacy simulators treated reconstructed raw volume as shares,
so the ADV participation cap was ~100x too restrictive. We keep every execution
rule unchanged and only convert lots -> shares before calling the frozen legacy
simulators.
"""
import numpy as np
import pandas as pd

import run_10y_skewfilter_hard as hard
import run_10y_grand_opt as grand

_ORIG_HARD_SIMULATE = hard.hard_simulate
_ORIG_FAST_SIMULATE = grand.fast_simulate
LOTS_TO_SHARES = 100.0


def _volume_to_shares(panel: pd.DataFrame) -> pd.DataFrame:
    z = panel.copy()
    if 'exec_volume' not in z.columns:
        raise RuntimeError('exec_volume missing from execution panel')
    v = pd.to_numeric(z['exec_volume'], errors='coerce').to_numpy(float, copy=True)
    finite = np.isfinite(v)
    v[finite] *= LOTS_TO_SHARES
    z['exec_volume'] = v
    return z


def hard_simulate(panel, cal, members, cost_mult=1.0):
    return _ORIG_HARD_SIMULATE(_volume_to_shares(panel), cal, members, cost_mult)


def fast_simulate(panel, cal, members, cost_mult=1.0):
    return _ORIG_FAST_SIMULATE(_volume_to_shares(panel), cal, members, cost_mult)


def install():
    """Patch only the two execution entry points used by new research scripts."""
    hard.hard_simulate = hard_simulate
    grand.fast_simulate = fast_simulate


def unit_audit() -> dict:
    x = pd.DataFrame({'exec_volume':[1.0, 12.5, np.nan]})
    y = _volume_to_shares(x)
    ok = bool(y.exec_volume.iloc[0] == 100.0 and y.exec_volume.iloc[1] == 1250.0 and np.isnan(y.exec_volume.iloc[2]))
    return {
        'lots_to_shares': LOTS_TO_SHARES,
        'one_lot_input': float(x.exec_volume.iloc[0]),
        'one_lot_output_shares': float(y.exec_volume.iloc[0]),
        'unit_conversion_ok': int(ok),
        'scope': 'volume participation only; prices, fees, slippage, lots, ranking, timing unchanged',
    }
