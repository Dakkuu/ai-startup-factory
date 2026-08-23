from __future__ import annotations
from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_alpha2f_v2 as sim
import run_10y_max_audit as ma

START=mo.START; TRAIN_END=mo.TRAIN_END; HALF1_END=mo.HALF1_END; HALF2_START=mo.HALF2_START; PSEUDO_START=mo.PSEUDO_START; END=mo.END
HOLDS=(60,90,120,150)
NS=(8,10,12,15,20)
BUFFERS=((.05,.20),(.10,.30))

FACTOR_COL={'iv':'ivol60','ef':'eff120','down':'dsemi60','amax':'max20','askew':'skew60','rmom':'rmom126','tstat':'tstat120','dd':'dd120','beta':'beta252','capture':'capture120','mom':'mom120','volshock':'volshock'}


def specs_risktrend():
    return [
      {'name':'rt_bal','kind':'weighted','w':{'iv':.24,'ef':.20,'down':.16,'amax':.10,'rmom':.18,'tstat':.12}},
      {'name':'rt_def','kind':'weighted','w':{'iv':.30,'down':.24,'amax':.14,'beta':.10,'ef':.12,'rmom':.10}},
      {'name':'rt_mom','kind':'weighted','w':{'iv':.20,'down':.14,'ef':.20,'rmom':.26,'tstat':.12,'amax':.08}},
      {'name':'rt_quality','kind':'weighted','w':{'iv':.24,'ef':.24,'down':.14,'tstat':.14,'capture':.10,'rmom':.14}},
      {'name':'anti_lottery2','kind':'weighted','w':{'iv':.24,'ef':.16,'down':.14,'amax':.14,'askew':.10,'rmom':.22}},
      {'name':'lowrisk_capture2','kind':'weighted','w':{'iv':.26,'down':.24,'beta':.14,'capture':.14,'ef':.10,'rmom':.12}},
      {'name':'quiet_trend2','kind':'weighted','w':{'iv':.20,'ef':.24,'rmom':.22,'tstat':.18,'volshock':.16}},
      {'name':'drawdown_trend2','kind':'weighted','w':{'iv':.22,'ef':.20,'down':.18,'dd':.12,'rmom':.18,'tstat':.10}},
      {'name':'lowbeta_mom2','kind':'weighted','w':{'iv':.22,'beta':.18,'down':.16,'ef':.16,'rmom':.20,'tstat':.08}},
      {'name':'capture_mom2','kind':'weighted','w':{'iv':.20,'ef':.18,'down':.14,'capture':.16,'rmom':.22,'amax':.10}},
      {'name':'risk_core','kind':'weighted','w':{'iv':.34,'down':.26,'amax':.16,'beta':.14,'ef':.10}},
      {'name':'trend_core','kind':'weighted','w':{'iv':.16,'ef':.26,'rmom':.28,'tstat':.18,'dd':.12}},
    ]


def specs_twostage():
    out=[]
    out += [
      {'name':'g_lowrisk_45','kind':'gate','g':{'iv':.45,'down':.45},'w':{'ef':.45,'rmom':.35,'tstat':.20}},
      {'name':'g_lowrisk_55','kind':'gate','g':{'iv':.55,'down':.55},'w':{'ef':.40,'rmom':.35,'tstat':.15,'amax':.10}},
      {'name':'g_lottery_50','kind':'gate','g':{'amax':.50,'askew':.50},'w':{'iv':.35,'ef':.30,'rmom':.25,'down':.10}},
      {'name':'g_lottery_65','kind':'gate','g':{'amax':.65,'askew':.65},'w':{'iv':.30,'ef':.25,'rmom':.30,'down':.15}},
      {'name':'g_eff_40','kind':'gate','g':{'ef':.40},'w':{'iv':.35,'down':.25,'rmom':.25,'amax':.15}},
      {'name':'g_eff_55','kind':'gate','g':{'ef':.55},'w':{'iv':.30,'down':.20,'rmom':.30,'tstat':.20}},
      {'name':'g_mom_45','kind':'gate','g':{'rmom':.45},'w':{'iv':.35,'ef':.30,'down':.20,'amax':.15}},
      {'name':'g_mom_60','kind':'gate','g':{'rmom':.60},'w':{'iv':.30,'ef':.30,'down':.20,'tstat':.20}},
      {'name':'g_trend_50','kind':'gate','g':{'ef':.55,'rmom':.55,'tstat':.60},'w':{'iv':.45,'down':.30,'amax':.25}},
      {'name':'g_deftrend_60','kind':'gate','g':{'iv':.60,'down':.60,'amax':.65},'w':{'ef':.35,'rmom':.40,'tstat':.25}},
      {'name':'g_lowbeta_55','kind':'gate','g':{'beta':.55,'iv':.60},'w':{'ef':.30,'rmom':.35,'down':.20,'capture':.15}},
      {'name':'g_quiet_60','kind':'gate','g':{'volshock':.60,'amax':.65},'w':{'iv':.25,'ef':.30,'rmom':.30,'tstat':.15}},
    ]
    return out


def specs_nonlinear():
    return [
      {'name':'nl_risktrend_geo','kind':'groups','risk':{'iv':.35,'down':.30,'amax':.20,'beta':.15},'trend':{'ef':.35,'rmom':.35,'tstat':.20,'dd':.10},'combine':'geo'},
      {'name':'nl_risktrend_max','kind':'groups','risk':{'iv':.35,'down':.30,'amax':.20,'beta':.15},'trend':{'ef':.35,'rmom':.35,'tstat':.20,'dd':.10},'combine':'max'},
      {'name':'nl_risktrend_meanmax','kind':'groups','risk':{'iv':.35,'down':.30,'amax':.20,'beta':.15},'trend':{'ef':.35,'rmom':.35,'tstat':.20,'dd':.10},'combine':'meanmax'},
      {'name':'nl_lotterytrend_geo','kind':'groups','risk':{'iv':.35,'down':.25,'amax':.25,'askew':.15},'trend':{'ef':.35,'rmom':.40,'tstat':.15,'capture':.10},'combine':'geo'},
      {'name':'nl_lotterytrend_max','kind':'groups','risk':{'iv':.35,'down':.25,'amax':.25,'askew':.15},'trend':{'ef':.35,'rmom':.40,'tstat':.15,'capture':.10},'combine':'max'},
      {'name':'nl_iv_eff_power2','kind':'power','f':('iv','ef'),'p':2.0},
      {'name':'nl_iv_eff_power4','kind':'power','f':('iv','ef'),'p':4.0},
      {'name':'nl_iv_down_ef_geo','kind':'geo3','f':('iv','down','ef')},
      {'name':'nl_iv_amax_rmom_geo','kind':'geo3','f':('iv','amax','rmom')},
      {'name':'nl_iv_eff_rmom_geo','kind':'geo3','f':('iv','ef','rmom')},
      {'name':'nl_bottleneck4','kind':'bottleneck','f':('iv','ef','down','rmom')},
      {'name':'nl_meanmax4','kind':'meanmax4','f':('iv','ef','down','rmom')},
    ]


def required(spec):
    ks=[]
    if spec['kind'] in ('weighted','gate'):
        ks += list(spec.get('w',{})); ks += list(spec.get('g',{}))
    elif spec['kind']=='groups': ks += list(spec['risk'])+list(spec['trend'])
    else: ks += list(spec.get('f',()))
    return sorted(set(FACTOR_COL[k] for k in ks if k in FACTOR_COL))


def weighted(R,w):
    keys=[k for k in w if k in R]
    sw=sum(float(w[k]) for k in keys)
    if not keys or sw<=0:return None
    z=None
    for k in keys:
        v=R[k]*float(w[k])/sw
        z=v if z is None else z+v
    return z


def make_rank(p,spec,liq=.70,skew=.80):
    x=p.copy(); x['rank_test']=np.nan
    m=mo.eligible_mask(x,liq,skew)
    for c in required(spec): m &= np.isfinite(x[c])
    if not m.any(): return x
    R=mo.component_ranks(x,m); kind=spec['kind']
    if kind=='weighted': raw=weighted(R,spec['w'])
    elif kind=='gate':
        keep=pd.Series(True,index=R[next(iter(R))].index)
        for k,q in spec['g'].items():
            if k not in R: keep &= False
            else: keep &= R[k] <= float(q)
        mm=pd.Series(False,index=x.index); mm.loc[keep.index]=keep; m &= mm
        if not m.any(): return x
        R=mo.component_ranks(x,m); raw=weighted(R,spec['w'])
    elif kind=='groups':
        a=weighted(R,spec['risk']); b=weighted(R,spec['trend'])
        if spec['combine']=='geo': raw=np.sqrt((a*b).clip(lower=1e-12))
        elif spec['combine']=='max': raw=pd.concat([a,b],axis=1).max(axis=1)
        else: raw=.5*((a+b)/2)+.5*pd.concat([a,b],axis=1).max(axis=1)
    elif kind=='power':
        a,b=(R[k] for k in spec['f']); pw=float(spec['p']); raw=((a.pow(pw)+b.pow(pw))/2).pow(1/pw)
    elif kind=='geo3':
        a,b,c=(R[k] for k in spec['f']); raw=((a*b*c).clip(lower=1e-12)).pow(1/3)
    elif kind=='bottleneck': raw=pd.concat([R[k] for k in spec['f']],axis=1).max(axis=1)
    elif kind=='meanmax4':
        z=pd.concat([R[k] for k in spec['f']],axis=1); raw=.5*z.mean(axis=1)+.5*z.max(axis=1)
    else: raise ValueError(kind)
    if raw is None:return x
    x.loc[m,'rank_test']=raw.groupby(x.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return x


def annual(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]
    rows=[]
    for y,g in s.groupby(s.index.year):
        b=s[s.index<pd.Timestamp(f'{y}-01-01')]; st=float(b.iloc[-1]) if len(b) else float(g.iloc[0])
        rows.append({'year':int(y),'return':float(g.iloc[-1]/st-1)})
    return pd.DataFrame(rows)


def train_point(q,h,n,e,k,cal,members,bm):
    st,eq,tr,tm=ma.run_q(q,int(h),0,cal,members,bm,n=int(n),entry=float(e),keep=float(k),start=START,end=TRAIN_END)
    st['half1_return']=mo.period_return(eq,START,HALF1_END); st['half2_return']=mo.period_return(eq,HALF2_START,TRAIN_END)
    st['min_half_return']=min(st['half1_return'],st['half2_return']) if np.isfinite(st['half1_return']) and np.isfinite(st['half2_return']) else np.nan
    return st


def full_point(q,h,n,e,k,cal,members,bm,cost=1.,phase=0):
    return mo.full_run(q,h,n,e,k,cal,members,bm,cost=cost,phase=phase)


def select_finalists(grid):
    z=grid[(grid.half1_return>0)&(grid.half2_return>0)&(grid.max_drawdown>-0.45)].copy()
    if len(z)==0:z=grid.copy()
    growth=z.sort_values(['total_return','min_half_return','max_drawdown'],ascending=[False,False,False]).iloc[0]
    robust=z.sort_values(['min_half_return','total_return','max_drawdown'],ascending=[False,False,False]).iloc[0]
    riskadj=z.assign(score=z.total_return/(1+z.max_drawdown.abs())).sort_values(['score','min_half_return'],ascending=[False,False]).iloc[0]
    out=pd.DataFrame([growth,robust,riskadj]).drop_duplicates(['signal','hold','n_hold','entry_pct','keep_pct'])
    return out


def main(lane):
    out=Path(f'results_megaopt_v1_{lane}'); out.mkdir(exist_ok=True)
    p,cal,members,ua,market_code,bm=mo.build_panel(out,need_fwd=False)
    specs={'risktrend':specs_risktrend(),'twostage':specs_twostage(),'nonlinear':specs_nonlinear()}[lane]
    rows=[]
    for si,spec in enumerate(specs,1):
        print('SIGNAL',lane,si,len(specs),spec['name'],flush=True)
        q=make_rank(p,spec)
        for h in HOLDS:
          for n in NS:
           for e,k in BUFFERS:
            st=train_point(q,h,n,e,k,cal,members,bm)
            st.update(signal=spec['name'],spec=json.dumps(spec,sort_keys=True),hold=h,n_hold=n,entry_pct=e,keep_pct=k)
            rows.append(st)
        del q
    grid=pd.DataFrame(rows); grid.to_csv(out/'train_grid.csv',index=False)
    fins=select_finalists(grid); fins.to_csv(out/'train_selected.csv',index=False)

    full=[]; costs=[]; phases=[]; anns=[]; tails=[]
    for r in fins.itertuples(index=False):
        spec=next(s for s in specs if s['name']==r.signal); q=make_rank(p,spec)
        st,eq,tr,tm=full_point(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm)
        st.update(signal=r.signal,selection='2016-2021 only',hold=int(r.hold),n_hold=int(r.n_hold),entry_pct=float(r.entry_pct),keep_pct=float(r.keep_pct)); full.append(st)
        a=annual(eq); a['signal']=r.signal; a['hold']=int(r.hold); a['n_hold']=int(r.n_hold); anns.append(a)
        rr=sim.robustness(eq,tr); rr.update(signal=r.signal,hold=int(r.hold),n_hold=int(r.n_hold)); tails.append(rr)
        for cm in (2.,4.,8.):
            x,_,_,_=full_point(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm,cost=cm); x.update(signal=r.signal,cost_mult_test=cm); costs.append(x)
        step=max(1,round(int(r.hold)/5))
        for ph in range(step):
            x,_,_,_=full_point(q,int(r.hold),int(r.n_hold),float(r.entry_pct),float(r.keep_pct),cal,members,bm,phase=ph); x.update(signal=r.signal,phase=ph); phases.append(x)
        del q
    fdf=pd.DataFrame(full).sort_values(['train_2016_2021_return','total_return'],ascending=[False,False]); fdf.to_csv(out/'full_finalists.csv',index=False)
    pd.DataFrame(costs).to_csv(out/'costs.csv',index=False); pd.DataFrame(phases).to_csv(out/'phases.csv',index=False); pd.DataFrame(tails).to_csv(out/'tails.csv',index=False)
    if anns: pd.concat(anns,ignore_index=True).to_csv(out/'annual.csv',index=False)
    ph=pd.DataFrame(phases)
    phase_summary=[]
    for sig,g in ph.groupby('signal'):
        phase_summary.append({'signal':sig,'phase_count':len(g),'phase_min_return':g.total_return.min(),'phase_median_return':g.total_return.median(),'phase_max_return':g.total_return.max(),'all_phases_positive':int((g.total_return>0).all()),'phase_min_pseudo_oos':g.pseudo_oos_2022_2026_return.min(),'phase_median_pseudo_oos':g.pseudo_oos_2022_2026_return.median()})
    pd.DataFrame(phase_summary).to_csv(out/'phase_summary.csv',index=False)
    audit={**ua,'market_factor':market_code,'lane':lane,'signal_count':len(specs),'train_grid_points':len(grid),'finalists':len(fdf),'selection':'all signal/config selection uses 2016-2021 only; 2022-2026 pseudo-OOS only','signal_universe':'T-only signal-pure','volume_unit_shares':100,'target500_hits_finalists':int((fdf.total_return>=5.0).sum()) if len(fdf) else 0}
    pd.DataFrame([audit]).to_csv(out/'audit.csv',index=False)
    print('=== TRAIN TOP ==='); print(grid.sort_values('total_return',ascending=False).head(25).to_string(index=False),flush=True)
    print('=== FULL FINALISTS ==='); print(fdf.to_string(index=False),flush=True)
    print('=== PHASE SUMMARY ==='); print(pd.DataFrame(phase_summary).to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('lane',choices=('risktrend','twostage','nonlinear')); args=ap.parse_args(); main(args.lane)
