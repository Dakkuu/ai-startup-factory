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

START=mo.START; TRAIN_END=mo.TRAIN_END; PSEUDO=mo.PSEUDO_START; END=mo.END
SEED=20260823
VARIANTS={
 'cur_secondary':{'iv':.267,'ef':.267,'rmom':.293,'tstat':.173},
 'mom_eff':{'iv':.20,'ef':.20,'rmom':.35,'tstat':.25},
 'down_mom':{'iv':.20,'down':.20,'rmom':.35,'tstat':.25},
 'amax_mom':{'iv':.20,'amax':.15,'rmom':.40,'tstat':.25},
 'balanced':{'iv':.20,'ef':.15,'down':.15,'amax':.10,'rmom':.25,'tstat':.15},
 'lotteryfree':{'iv':.25,'down':.20,'amax':.15,'askew':.10,'rmom':.20,'tstat':.10},
 'trend_heavy':{'iv':.15,'ef':.15,'rmom':.45,'tstat':.25},
 'defensive':{'iv':.30,'ef':.15,'down':.25,'amax':.15,'rmom':.15},
 'iv_rmom':{'iv':.30,'rmom':.50,'tstat':.20},
 'eff_rmom':{'ef':.25,'rmom':.50,'tstat':.25},
 'dd_mom':{'iv':.20,'dd':.15,'rmom':.40,'tstat':.25},
 'beta_mom':{'iv':.20,'beta':.15,'rmom':.40,'tstat':.25},
}
SHARDS={'a':['cur_secondary','mom_eff','down_mom','amax_mom'],
        'b':['balanced','lotteryfree','trend_heavy','defensive'],
        'c':['iv_rmom','eff_rmom','dd_mom','beta_mom']}
LIQS=(.45,.55,.65); FLOORS=(1.5,2.0); PRICE_CAPS=(.20,.30,.40,.50); BLENDS=(0.,.25)
HOLDS=(60,90,120); NS=(8,10,12,15); BUFFERS=((.05,.20),(.10,.30))
BASECOLS=strict.BASECOLS+['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy']
REQ={'down':'dsemi60','amax':'max20','askew':'skew60','rmom':'rmom126','tstat':'tstat120','dd':'dd120','beta':'beta252'}


def rank_gate(p,w,liq,floor,price_cap,blend):
    x=p.copy(); x['rank_test']=np.nan
    m=mo.eligible_mask(x,float(liq),.80)&np.isfinite(x.raw_price)&(x.raw_price>=float(floor))
    for k in w:
        if k in REQ: m &= np.isfinite(x[REQ[k]])
    if not m.any(): return x
    pr=x.loc[m].groupby('signal_date').raw_price.rank(pct=True,method='average',ascending=True)
    keep=pr<=float(price_cap); m2=pd.Series(False,index=x.index); m2.loc[keep.index]=keep; m &= m2
    if not m.any(): return x
    R=mo.component_ranks(x,m); parts=[]; raw=None; sw=0.
    for k,v in w.items():
        if k not in R: continue
        vv=float(v); parts.append(R[k]); raw=R[k]*vv if raw is None else raw+R[k]*vv; sw+=vv
    if raw is None or sw<=0: return x
    raw=raw/sw
    if float(blend)>0 and parts:
        worst=pd.concat(parts,axis=1).max(axis=1)
        raw=(1-float(blend))*raw+float(blend)*worst
    x.loc[m,'rank_test']=raw.groupby(x.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return x


def phase_count(h): return max(1,round(int(h)/5))
def screen_phases(h):
    n=phase_count(h); return sorted(set(int(round(v))%n for v in np.linspace(0,n-1,4)))

def subset(q,h,ph):
    pc=phase_count(h); dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[int(ph)::pc])
    cols=[c for c in BASECOLS if c in q.columns]; z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def run_one(q,h,ph,n,e,k,cal,members,bm,cash=1e6,train=True,cost=1.):
    kw={'n':int(n),'entry':float(e),'keep':float(k),'initial_cash':float(cash),'cost':float(cost)}
    if train: kw.update(start=START,end=TRAIN_END)
    return ma.run_panel(subset(q,h,ph),cal,members,bm,**kw)

def combine_abs(eqs,initials,start):
    start=pd.Timestamp(start); idx={start}; ser=[]
    for e,init in zip(eqs,initials):
        s=e.set_index(pd.to_datetime(e.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]
        s=pd.concat([pd.Series({start:float(init)}),s]); s=s[~s.index.duplicated(keep='last')].sort_index(); ser.append(s); idx.update(s.index)
    idx=pd.DatetimeIndex(sorted(idx)); arr=[s.reindex(idx).ffill().fillna(float(init)) for s,init in zip(ser,initials)]
    tot=pd.concat(arr,axis=1).sum(axis=1); return pd.DataFrame({'trade_date':idx,'equity':tot.to_numpy(float)})
def eq_return(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); return float(s.iloc[-1]/s.iloc[0]-1)
def eq_cagr(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); y=max((s.index[-1]-s.index[0]).days/365.25,1e-9); return float((s.iloc[-1]/s.iloc[0])**(1/y)-1)
def eq_mdd(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); return float((s/s.cummax()-1).min())
def period(eq,a,b):
    z=eq[(pd.to_datetime(eq.trade_date)>=pd.Timestamp(a))&(pd.to_datetime(eq.trade_date)<=pd.Timestamp(b))]
    return float(z.equity.iloc[-1]/z.equity.iloc[0]-1) if len(z)>1 else np.nan


def eval_phases(q,h,n,e,k,cal,members,bm,phs,train=True,total_cash=None,cost=1.):
    eqs=[]; rets=[]; cgs=[]; mdds=[]; pcash=(float(total_cash)/len(phs) if total_cash is not None else 1e6)
    for ph in phs:
        st,eq,tr,tm=run_one(q,h,ph,n,e,k,cal,members,bm,cash=pcash,train=train,cost=cost)
        eqs.append(eq); rets.append(float(st['total_return'])); cgs.append(float(st['cagr'])); mdds.append(float(st['max_drawdown']))
    out={'min_return':float(np.min(rets)),'median_return':float(np.median(rets)),'min_cagr':float(np.min(cgs)),'median_cagr':float(np.median(cgs)),'std_cagr':float(np.std(cgs)),'worst_mdd':float(np.min(mdds)),'all_positive':int(np.min(rets)>0)}
    if total_cash is not None:
        ens=combine_abs(eqs,[pcash]*len(phs),START); out.update(ensemble_return=eq_return(ens),ensemble_cagr=eq_cagr(ens),ensemble_mdd=eq_mdd(ens)); return out,ens
    return out,None


def main(shard):
    out=Path(f'results_lowprice_gate_v3_{shard}'); out.mkdir(exist_ok=True)
    p,cal,members,ua,market_code,bm=mo.build_panel(out,need_fwd=False); p=lp.attach_price(p,cal); p=strict.attach_gap_flags(p,cal,'board')
    sigrows=[]; qcache={}
    for name in SHARDS[shard]:
      w=VARIANTS[name]
      for liq in LIQS:
       for floor in FLOORS:
        for cap in PRICE_CAPS:
         for blend in BLENDS:
          key=(name,liq,floor,cap,blend); print('SIGNAL',shard,key,flush=True); q=rank_gate(p,w,liq,floor,cap,blend); qcache[key]=q
          r,_=eval_phases(q,90,10,.10,.30,cal,members,bm,screen_phases(90),train=True)
          score=r['median_cagr']+.65*r['min_cagr']-.35*r['std_cagr']+.05*r['worst_mdd']
          sigrows.append({'shard':shard,'variant':name,'weights':json.dumps(w,sort_keys=True),'liq':liq,'floor':floor,'price_cap':cap,'blendmax':blend,**r,'signal_score':score})
    sig=pd.DataFrame(sigrows); sig.to_csv(out/'stage1_signals.csv',index=False)
    z=sig[(sig.all_positive==1)&(sig.worst_mdd>-0.55)].copy(); z=z if len(z) else sig.copy(); top=z.sort_values(['signal_score','min_cagr'],ascending=False).head(12)
    rows=[]
    for s in top.itertuples(index=False):
      q=qcache[(s.variant,float(s.liq),float(s.floor),float(s.price_cap),float(s.blendmax))]
      for h in HOLDS:
       for n in NS:
        for e,k in BUFFERS:
          r,_=eval_phases(q,h,n,e,k,cal,members,bm,screen_phases(h),train=True)
          score=r['median_cagr']+.65*r['min_cagr']-.35*r['std_cagr']+.05*r['worst_mdd']
          rows.append({'shard':shard,'variant':s.variant,'weights':s.weights,'liq':s.liq,'floor':s.floor,'price_cap':s.price_cap,'blendmax':s.blendmax,'hold':h,'n_hold':n,'entry':e,'keep':k,**r,'screen_score':score})
    grid=pd.DataFrame(rows); grid.to_csv(out/'stage2_grid.csv',index=False)
    g=grid[(grid.all_positive==1)&(grid.worst_mdd>-0.55)].copy(); g=g if len(g) else grid.copy(); top2=g.sort_values(['screen_score','min_cagr'],ascending=False).head(8)
    ex=[]
    for r in top2.itertuples(index=False):
        q=qcache[(r.variant,float(r.liq),float(r.floor),float(r.price_cap),float(r.blendmax))]; pc=phase_count(r.hold); phs=list(range(pc))
        x,ens=eval_phases(q,r.hold,r.n_hold,r.entry,r.keep,cal,members,bm,phs,train=True,total_cash=1e6)
        robust=x['ensemble_cagr']+.65*x['min_cagr']-.35*x['std_cagr']+.05*x['ensemble_mdd']
        ex.append({**r._asdict(),**{f'exact_{k}':v for k,v in x.items()},'robust_score_final':robust})
    exdf=pd.DataFrame(ex).sort_values(['robust_score_final','exact_min_cagr'],ascending=False); exdf.to_csv(out/'stage3_exact_train.csv',index=False); winner=exdf.iloc[0].to_dict(); pd.DataFrame([winner]).to_csv(out/'train_winner.csv',index=False)

    q=qcache[(winner['variant'],float(winner['liq']),float(winner['floor']),float(winner['price_cap']),float(winner['blendmax']))]; pc=phase_count(int(winner['hold'])); phs=list(range(pc))
    full,eq=eval_phases(q,int(winner['hold']),int(winner['n_hold']),float(winner['entry']),float(winner['keep']),cal,members,bm,phs,train=False,total_cash=1e6)
    full.update(train_selected_robust_score=winner['robust_score_final'],train_only_selection=1,train_2016_2021_return=period(eq,START,TRAIN_END),pseudo_oos_2022_2026_return=period(eq,PSEUDO,END),variant=winner['variant'],weights=winner['weights'],liq=winner['liq'],floor=winner['floor'],price_cap=winner['price_cap'],blendmax=winner['blendmax'],hold=winner['hold'],n_hold=winner['n_hold'],entry=winner['entry'],keep=winner['keep'])
    pd.DataFrame([full]).to_csv(out/'shard_winner_full_validation.csv',index=False); fa.annual(eq).to_csv(out/'shard_winner_annual.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'shard':shard,'selection_period':'2016-07-29..2021-12-31','validation_period_not_used_in_selection':1,'design':'low nominal price percentile gate -> secondary T-only quality/momentum rank','hard_executor':'v3; 100-share lots; board-limit blocked execution; no replacement','total_cash_exact_split_train':1_000_000,'candidate_signals':len(sig),'stage2_configs':len(grid),'stage3_exact':len(exdf)}]).to_csv(out/'audit.csv',index=False)
    print('TRAIN WINNER',pd.DataFrame([winner]).to_string(index=False),flush=True); print('FULL VALIDATION',pd.DataFrame([full]).to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('shard',choices=('a','b','c')); a=ap.parse_args(); main(a.shard)
