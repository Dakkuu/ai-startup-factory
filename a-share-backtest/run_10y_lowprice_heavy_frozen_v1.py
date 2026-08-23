from __future__ import annotations
import argparse
import run_10y_lowprice_strict_validation_v2 as v

# FROZEN before any 2022-2026 validation: selected solely by the 2016-2021
# all-phase exact-split objective in run_10y_lowprice_phase_robust_v2.
v.WEIGHTS={'price':.35,'iv':.18,'ef':.15,'rmom':.20,'tstat':.12}
v.CFG={'liq':.55,'floor':2.0,'hold':60,'n':8,'entry':.10,'keep':.30}
v.NPHASE=12

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=('core','capacity','delay','noise','delete','placebo'))
    a=ap.parse_args()
    v.main(a.mode)
