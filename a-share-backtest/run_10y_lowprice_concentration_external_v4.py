from pathlib import Path
import run_10y_lowprice_external_audit_v2 as ext

# Frozen concentration V4 candidate; no parameter changes from stress outcomes.
ext.OUT = Path('results_lowprice_concentration_external_v4')
ext.OUT.mkdir(exist_ok=True)
ext.WEIGHTS = {'price':.35,'iv':.18,'ef':.15,'rmom':.20,'tstat':.12}
ext.CFG = {'liq':.55,'floor':2.0,'hold':60,'n':4,'entry':.15,'keep':.40}
ext.NPHASE = 12

if __name__ == '__main__':
    ext.main()
