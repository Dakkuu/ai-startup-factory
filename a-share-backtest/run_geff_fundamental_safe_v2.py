from __future__ import annotations
import sys
import requests

# Transport-only safeguard: do not change factor definitions or backtest rules.
_orig_get = requests.get

def _safe_get(*args, **kwargs):
    kwargs.setdefault('timeout', 20)
    return _orig_get(*args, **kwargs)

requests.get = _safe_get

import run_geff_fundamental_fastpit_v1 as fp

# Strategy starts in 2016. 2015 gives >550 days of pre-start accounting history,
# so older quarters cannot affect any non-stale signal observation.
def _needed_quarters(y0=2015, y1=2026):
    out=[]
    for y in range(2015, 2027):
        for md in ('0331','0630','0930','1231'):
            d=f'{y}{md}'
            if d <= '20260630': out.append(d)
    return out
fp.quarter_ends = _needed_quarters

mode = sys.argv[1] if len(sys.argv) > 1 else 'fast'
if mode == 'fast':
    fp.main()
elif mode == '3stmt':
    import run_geff_fundamental_3stmt_pit_v1 as m
    m.main()
elif mode == 'value':
    import run_geff_fundamental_value_pit_v1 as m
    m.main()
else:
    raise SystemExit(f'unknown mode {mode}')
