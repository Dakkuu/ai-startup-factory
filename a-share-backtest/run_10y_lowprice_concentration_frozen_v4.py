from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

import run_10y_lowprice_strict_validation_v2 as sv

# FROZEN BEFORE THIS STRESS AUDIT.
# Selected on 2016-07-29..2021-12-31 only by the predeclared concentration/horizon
# search with all-phase exact RMB1m split, both train halves positive, and MDD gate.
sv.WEIGHTS = {'price': .35, 'iv': .18, 'ef': .15, 'rmom': .20, 'tstat': .12}
sv.CFG = {'liq': .55, 'floor': 2.0, 'hold': 60, 'n': 4, 'entry': .15, 'keep': .40}
sv.NPHASE = 12


def main(mode: str):
    out = Path(f'results_lowprice_concentration_frozen_v4_{mode}')
    out.mkdir(exist_ok=True)
    p, q, cal, members, ua, market_code, bm = sv.build(out)
    if mode == 'core':
        sv.core(out, p, q, cal, members, ua, market_code, bm)
    elif mode == 'capacity':
        sv.capacity(out, q, cal, members, bm)
    elif mode == 'delay':
        sv.delay(out, q, cal, members, bm)
    elif mode == 'noise':
        sv.noise(out, q, cal, members, bm)
    elif mode == 'delete':
        sv.deletion(out, q, cal, members, bm)
    elif mode == 'placebo':
        sv.placebo(out, q, cal, members, bm)
    else:
        raise ValueError(mode)

    pd.DataFrame([{
        'candidate': 'LowPrice-Concentration-Frozen-V4',
        'weights': str(sv.WEIGHTS),
        'config': str(sv.CFG),
        'nphase': sv.NPHASE,
        'selection_period': '2016-07-29..2021-12-31',
        'selection_method': 'predeclared concentration/horizon search; all-phase exact split; both train halves positive; MDD gate; then max train CAGR',
        'validation_2022_2026_used_in_selection': 0,
        'parameter_lock': 1,
        'executor': 'hard_v3; T-close signal to later open; 100-share lots; board-limit block; no replacement',
    }]).to_csv(out / 'frozen_provenance.csv', index=False)
    print('DONE', mode, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('core','capacity','delay','noise','delete','placebo'))
    a = ap.parse_args()
    main(a.mode)
