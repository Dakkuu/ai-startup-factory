from pathlib import Path
import os
import run_10y_ultra_opt as u

idx=int(os.environ.get('ULTRA_LIQ_SHARD','0'))
if idx<0 or idx>=len(u.LIQ_LEVELS): raise ValueError(idx)
liq=u.LIQ_LEVELS[idx]
u.LIQ_LEVELS=(liq,)
u.OUT=Path(f'results_ultra_opt_liq{int(round(liq*100))}')
u.OUT.mkdir(exist_ok=True)
print('FROZEN LIQ SHARD',idx,liq,flush=True)
u.main()
