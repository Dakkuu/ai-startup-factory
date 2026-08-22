from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from arch.bootstrap import SPA, StepM

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_grand_opt as grand
import run_10y_balanced_exact as be
import run_10y_max_audit as ma

OUT=Path('results_spa_audit'); OUT.mkdir(exist_ok=True)
SEED=20260822
BASELINE='w0.60_n20_h60_e0.10_k0.30'


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members)
    bm=market_close.loc[sim.START:sim.END].dropna()
    R,cands=ma.candidate_family(p,cal,members,bm)
    bmret=bm.pct_change(fill_method=None).rename('benchmark')
    aligned=pd.concat([bmret,R],axis=1,join='inner').dropna()
    b_loss=-aligned.benchmark
    m_loss=-aligned.drop(columns='benchmark')
    rows=[]
    superior_rows=[]
    for block in (20,60):
        for boot in ('stationary','circular'):
            print('SPA',block,boot,flush=True)
            spa=SPA(b_loss,m_loss,block_size=block,reps=5000,bootstrap=boot,studentize=True,nested=False,seed=SEED+block+(0 if boot=='stationary' else 1000))
            spa.compute(); pv=spa.pvalues
            better=list(spa.better_models(.05,'consistent'))
            rows.append({'block':block,'bootstrap':boot,'p_lower':float(pv['lower']),'p_consistent':float(pv['consistent']),'p_upper':float(pv['upper']),'n_superior_5pct':len(better),'baseline_superior_5pct':int(BASELINE in better),'baseline_name':BASELINE,'T':len(aligned),'K':m_loss.shape[1]})
            for x in better: superior_rows.append({'block':block,'bootstrap':boot,'model':x})
    spa_df=pd.DataFrame(rows); spa_df.to_csv(OUT/'spa_results.csv',index=False)
    pd.DataFrame(superior_rows).to_csv(OUT/'spa_superior_models.csv',index=False)

    # StepM controls family-wise error and identifies all strategies superior to CSI300.
    step_rows=[]
    for block in (20,60):
        print('STEPM',block,flush=True)
        s=StepM(b_loss,m_loss,size=.05,block_size=block,reps=3000,bootstrap='stationary',studentize=True,nested=False,seed=SEED+5000+block)
        s.compute(); sup=list(s.superior_models)
        step_rows.append({'block':block,'n_superior':len(sup),'baseline_superior':int(BASELINE in sup),'models':'|'.join(map(str,sup))})
    step_df=pd.DataFrame(step_rows); step_df.to_csv(OUT/'stepm_results.csv',index=False)

    # DSR sensitivity to much larger assumed research counts. These are sensitivity checks,
    # not a post-hoc gate replacement for the original 1000-trial preregistered test.
    q=be.anchor_weighted(p,'liq70',.60); bst,beq,_,_=ma.run_q(q,60,0,cal,members,bm,n=20,entry=.10,keep=.30)
    sharpes=cands.sharpe.to_numpy(float); br=ma.eqret(beq)
    dsr=[]
    for n in (1000,2500,5000,10000,25000):
        d=ma.dsr_one(br,sharpes,n); dsr.append(d)
    dsr_df=pd.DataFrame(dsr); dsr_df.to_csv(OUT/'dsr_sensitivity.csv',index=False)

    gates={
      'spa_consistent_stationary_block20_le_5pct':int(float(spa_df[(spa_df.block==20)&(spa_df.bootstrap=='stationary')].p_consistent.iloc[0])<=.05),
      'spa_consistent_stationary_block60_le_5pct':int(float(spa_df[(spa_df.block==60)&(spa_df.bootstrap=='stationary')].p_consistent.iloc[0])<=.05),
      'spa_consistent_circular_block20_le_5pct':int(float(spa_df[(spa_df.block==20)&(spa_df.bootstrap=='circular')].p_consistent.iloc[0])<=.05),
      'spa_consistent_circular_block60_le_5pct':int(float(spa_df[(spa_df.block==60)&(spa_df.bootstrap=='circular')].p_consistent.iloc[0])<=.05),
      'baseline_in_spa_superior_all_four':int((spa_df.baseline_superior_5pct==1).all()),
      'baseline_in_stepm_superior_both_blocks':int((step_df.baseline_superior==1).all()),
    }
    gd=pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]); gd.to_csv(OUT/'gates.csv',index=False)
    verdict={**ua,'market_factor':market_code,'baseline':BASELINE,'candidate_family':81,'T':len(aligned),'spa_hard_pass':int(gd['pass'].all()),'gates_passed':int(gd['pass'].sum()),'gates_total':len(gd),'note':'SPA/StepM uses losses=-daily returns; benchmark=CSI300; 5000 SPA reps, block 20/60, stationary+circular; no strategy parameter changed.'}
    pd.DataFrame([verdict]).to_csv(OUT/'verdict.csv',index=False)
    print('=== SPA ==='); print(spa_df.to_string(index=False),flush=True)
    print('=== STEPM ==='); print(step_df.to_string(index=False),flush=True)
    print('=== DSR SENSITIVITY ==='); print(dsr_df.to_string(index=False),flush=True)
    print('=== GATES ==='); print(gd.to_string(index=False),flush=True)
    print('=== VERDICT ==='); print(pd.DataFrame([verdict]).to_string(index=False),flush=True)

if __name__=='__main__': main()
