from __future__ import annotations
from pathlib import Path
import argparse, json, numpy as np, pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_lowprice_signalpure_v1 as lp
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_max_audit as ma
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_hard_executor_v3 as hv3
hv3.patch()

START=mo.START; TRAIN_END=mo.TRAIN_END
VARIANTS={
 'base':{'price':.25,'iv':.20,'ef':.20,'rmom':.22,'tstat':.13},
 'price_heavy':{'price':.35,'iv':.18,'ef':.15,'rmom':.20,'tstat':.12},
 'price_light':{'price':.15,'iv':.25,'ef':.20,'rmom':.25,'tstat':.15},
 'defensive':{'price':.20,'iv':.30,'ef':.20,'rmom':.18,'tstat':.12},
 'momentum':{'price':.20,'iv':.15,'ef':.15,'rmom':.32,'tstat':.18},
 'balanced':{'price':.20,'iv':.20,'ef':.20,'rmom':.25,'tstat':.15},
}
SHARDS={'a':['base','price_heavy'],'b':['price_light','defensive'],'c':['momentum','balanced']}
LIQS=(.45,.55,.65); FLOORS=(1.5,2.0,3.0); HOLDS=(60,90,120); NS=(8,10,12,15); BUFFERS=((.05,.20),(.10,.30))
BASECOLS=strict.BASECOLS+['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy']


def phase_count(h): return max(1,round(int(h)/5))
def screen_phases(h):
    n=phase_count(h); return sorted(set(int(round(x))%n for x in np.linspace(0,n-1,4)))

def subset(q,h,ph):
    n=phase_count(h); dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[int(ph)::n]); cols=[c for c in BASECOLS if c in q.columns]; z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')

def run_one(q,h,ph,n,e,k,cal,members,bm,cash=1e6):
    return ma.run_panel(subset(q,h,ph),cal,members,bm,n=int(n),entry=float(e),keep=float(k),initial_cash=float(cash),start=START,end=TRAIN_END)

def combine_abs(eqs,initials):
    start=pd.Timestamp(START); idx={start}; ser=[]
    for e,init in zip(eqs,initials):
        s=e.set_index(pd.to_datetime(e.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]; s=pd.concat([pd.Series({start:float(init)}),s]); s=s[~s.index.duplicated(keep='last')].sort_index(); ser.append(s); idx.update(s.index)
    idx=pd.DatetimeIndex(sorted(idx)); arr=[s.reindex(idx).ffill().fillna(float(init)) for s,init in zip(ser,initials)]; tot=pd.concat(arr,axis=1).sum(axis=1); return pd.DataFrame({'trade_date':idx,'equity':tot.to_numpy(float)})
def eq_cagr(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); yrs=max((s.index[-1]-s.index[0]).days/365.25,1e-9); return float((s.iloc[-1]/s.iloc[0])**(1/yrs)-1)
def eq_return(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); return float(s.iloc[-1]/s.iloc[0]-1)
def eq_mdd(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); return float((s/s.cummax()-1).min())


def main(shard):
    out=Path(f'results_lowprice_phase_robust_v2_{shard}'); out.mkdir(exist_ok=True)
    p,cal,members,ua,market_code,bm=mo.build_panel(out,need_fwd=False); p=lp.attach_price(p,cal); p=strict.attach_gap_flags(p,cal,'board')
    rows=[]; qcache={}
    for vname in SHARDS[shard]:
      w=VARIANTS[vname]
      for liq in LIQS:
       for floor in FLOORS:
        print('SIGNAL',shard,vname,liq,floor,flush=True); q=lp.rank_signal(p,w,liq,floor); qcache[(vname,liq,floor)]=q
        for h in HOLDS:
         phs=screen_phases(h)
         for n in NS:
          for e,k in BUFFERS:
            vals=[]
            for ph in phs:
                st,eq,tr,tm=run_one(q,h,ph,n,e,k,cal,members,bm,1e6); vals.append((st,eq))
            rets=np.array([x[0]['total_return'] for x in vals],float); cgs=np.array([x[0]['cagr'] for x in vals],float); mdds=np.array([x[0]['max_drawdown'] for x in vals],float)
            score=float(np.median(cgs)+.50*np.min(cgs)-.25*np.std(cgs))
            rows.append({'shard':shard,'variant':vname,'weights':json.dumps(w,sort_keys=True),'liq':liq,'floor':floor,'hold':h,'n_hold':n,'entry':e,'keep':k,'screen_phases':'|'.join(map(str,phs)),'screen_min_return':rets.min(),'screen_median_return':np.median(rets),'screen_min_cagr':cgs.min(),'screen_median_cagr':np.median(cgs),'screen_std_cagr':cgs.std(),'screen_worst_mdd':mdds.min(),'screen_all_positive':int((rets>0).all()),'screen_score':score})
    grid=pd.DataFrame(rows); grid.to_csv(out/'screen_grid.csv',index=False)
    eligible=grid[(grid.screen_all_positive==1)&(grid.screen_worst_mdd>-0.55)].copy(); eligible=eligible if len(eligible) else grid.copy()
    top=eligible.sort_values(['screen_score','screen_min_cagr'],ascending=False).head(24).copy(); top.to_csv(out/'stage2_candidates.csv',index=False)
    finals=[]
    for r in top.itertuples(index=False):
        q=qcache[(r.variant,float(r.liq),float(r.floor))]; pc=phase_count(r.hold); per=1e6/pc; eqs=[]; phase_cg=[]; phase_ret=[]; phase_mdd=[]
        for ph in range(pc):
            st,eq,tr,tm=run_one(q,r.hold,ph,r.n_hold,r.entry,r.keep,cal,members,bm,per); eqs.append(eq); phase_cg.append(st['cagr']); phase_ret.append(st['total_return']); phase_mdd.append(st['max_drawdown'])
        ens=combine_abs(eqs,[per]*pc); ec=eq_cagr(ens); er=eq_return(ens); em=eq_mdd(ens); cg=np.array(phase_cg,float); rr=np.array(phase_ret,float)
        robust=float(ec+.50*np.nanmin(cg)-.25*np.nanstd(cg))
        finals.append({**r._asdict(),'allphase_count':pc,'exact_split_train_return':er,'exact_split_train_cagr':ec,'exact_split_train_mdd':em,'allphase_min_return':np.nanmin(rr),'allphase_median_return':np.nanmedian(rr),'allphase_min_cagr':np.nanmin(cg),'allphase_median_cagr':np.nanmedian(cg),'allphase_std_cagr':np.nanstd(cg),'allphase_all_positive':int((rr>0).all()),'robust_score_final':robust})
    f=pd.DataFrame(finals).sort_values(['robust_score_final','allphase_min_cagr'],ascending=False); f.to_csv(out/'stage2_exact_train.csv',index=False); f.head(1).to_csv(out/'train_only_winner.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'shard':shard,'variants':'|'.join(SHARDS[shard]),'screen_configs':len(grid),'stage2_exact_candidates':len(f),'selection_period':'2016-07-29..2021-12-31','validation_2022_2026_accesses':0,'stage1':'4 fixed phases per config','stage2':'all phases, exact total 1m split cash, board-limit hard_v3','objective':'ensemble CAGR + .50 worst-phase CAGR - .25 phase CAGR std','volume_unit_shares':100}]).to_csv(out/'audit.csv',index=False)
    print('WINNER',f.head(1).to_string(index=False),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('shard',choices=('a','b','c')); a=ap.parse_args(); main(a.shard)
