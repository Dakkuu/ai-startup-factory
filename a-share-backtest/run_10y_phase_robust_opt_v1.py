from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_alpha2f_v2 as sim

OUT=Path('results_phase_robust_opt_v1'); OUT.mkdir(exist_ok=True)
HOLDS=(60,90,120); NS=(8,10,15,20); BUFFERS=((.05,.20),(.10,.30))
RISK={s['name']:s for s in mega.specs_risktrend() if s['name'] in ('trend_core','rt_mom','anti_lottery2','quiet_trend2')}
SPECIAL={
 'v4_linear':{'name':'v4_linear','kind':'mo','spec':{'name':'v4_linear','kind':'linear','liq':.55,'skew':.80,'w':.60}},
 'risk_meanmax':{'name':'risk_meanmax','kind':'mo','spec':{'name':'risk_meanmax','kind':'risk_meanmax','liq':.70,'skew':.90,'requires':['dsemi60','max20']}},
}
SIGNALS=list(RISK)+list(SPECIAL)


def makeq(p,name):
    if name in RISK:return mega.make_rank(p,RISK[name])
    return mo.rerank(p,SPECIAL[name]['spec'])


def phase_set(hold):
    step=max(1,round(hold/5)); return sorted(set([0,step//4,step//2,(3*step)//4]))


def train_phase(q,h,n,e,k,ph,cal,members,bm):
    st,eq,tr,tm=ma.run_q(q,h,ph,cal,members,bm,n=n,entry=e,keep=k,start=mo.START,end=mo.TRAIN_END)
    return st


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False)
    rows=[]
    for name in SIGNALS:
        print('SIGNAL',name,flush=True); q=makeq(p,name)
        for h in HOLDS:
          phs=phase_set(h)
          for n in NS:
           for e,k in BUFFERS:
            vals=[]
            for ph in phs:
                st=train_phase(q,h,n,e,k,ph,cal,members,bm); vals.append(st)
            rs=np.array([x['total_return'] for x in vals],float); dds=np.array([x['max_drawdown'] for x in vals],float)
            rows.append({'signal':name,'hold':h,'n_hold':n,'entry_pct':e,'keep_pct':k,'train_phase_count':len(phs),'train_phase_min':float(rs.min()),'train_phase_median':float(np.median(rs)),'train_phase_mean':float(rs.mean()),'train_phase_max':float(rs.max()),'train_phase_all_positive':int((rs>0).all()),'worst_train_mdd':float(dds.min()),'key':f'{name}|h{h}|n{n}|e{e}|k{k}'})
        del q
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'train_phase_grid.csv',index=False)
    z=grid[(grid.train_phase_all_positive==1)&(grid.worst_train_mdd>-0.50)].copy(); z=z if len(z) else grid.copy()
    # three pre-defined selection objectives: median growth, maximin, and median-minus-dispersion.
    med=z.sort_values(['train_phase_median','train_phase_min'],ascending=[False,False]).iloc[0]
    worst=z.sort_values(['train_phase_min','train_phase_median'],ascending=[False,False]).iloc[0]
    z=z.copy(); z['robust_score']=z.train_phase_median-.5*(z.train_phase_max-z.train_phase_min)
    stable=z.sort_values(['robust_score','train_phase_min'],ascending=[False,False]).iloc[0]
    fins=pd.DataFrame([med,worst,stable]).drop_duplicates('key'); fins.to_csv(OUT/'train_selected.csv',index=False)
    full=[]; phases=[]; costs=[]; annual=[]; tails=[]
    for r in fins.itertuples(index=False):
        q=makeq(p,r.signal)
        st,eq,tr,tm=mo.full_run(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm); st.update(key=r.key,signal=r.signal,selection='four evenly spaced phases in 2016-2021 only',hold=int(r.hold),n_hold=int(r.n_hold),entry_pct=float(r.entry_pct),keep_pct=float(r.keep_pct)); full.append(st)
        a=mega.annual(eq); a['key']=r.key; annual.append(a); rr=sim.robustness(eq,tr); rr['key']=r.key; tails.append(rr)
        for cm in (2.,4.,8.):
            x,_,_,_=mo.full_run(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm,cost=cm); x.update(key=r.key,cost_mult_test=cm); costs.append(x)
        step=max(1,round(int(r.hold)/5))
        for ph in range(step):
            x,_,_,_=mo.full_run(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm,phase=ph); x.update(key=r.key,phase=ph); phases.append(x)
    pd.DataFrame(full).to_csv(OUT/'full_finalists.csv',index=False); pd.DataFrame(costs).to_csv(OUT/'costs.csv',index=False); pd.DataFrame(phases).to_csv(OUT/'all_phases.csv',index=False); pd.DataFrame(tails).to_csv(OUT/'tails.csv',index=False)
    if annual:pd.concat(annual,ignore_index=True).to_csv(OUT/'annual.csv',index=False)
    ps=[]
    for key,g in pd.DataFrame(phases).groupby('key'):
        ps.append({'key':key,'phase_count':len(g),'full_phase_min':g.total_return.min(),'full_phase_median':g.total_return.median(),'full_phase_mean':g.total_return.mean(),'full_phase_max':g.total_return.max(),'all_phases_positive':int((g.total_return>0).all()),'pseudo_min':g.pseudo_oos_2022_2026_return.min(),'pseudo_median':g.pseudo_oos_2022_2026_return.median(),'pseudo_mean':g.pseudo_oos_2022_2026_return.mean(),'all_pseudo_positive':int((g.pseudo_oos_2022_2026_return>0).all())})
    pd.DataFrame(ps).to_csv(OUT/'phase_summary.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'signals':len(SIGNALS),'configs':len(grid),'train_exact_runs':int(sum(4 for _ in range(len(grid)))),'selection':'four fixed evenly spaced train phases, 2016-2021 only; 2022-2026 never used in selection','signal_universe':'T-only signal-pure','volume_unit_shares':100}]).to_csv(OUT/'audit.csv',index=False)
    print('=== TRAIN ROBUST TOP ===');print(z.sort_values('train_phase_median',ascending=False).head(30).to_string(index=False),flush=True)
    print('=== FULL ===');print(pd.DataFrame(full).to_string(index=False),flush=True)
    print('=== PHASE ===');print(pd.DataFrame(ps).to_string(index=False),flush=True)

if __name__=='__main__':main()
