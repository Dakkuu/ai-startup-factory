from __future__ import annotations
from pathlib import Path
import argparse, numpy as np, pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_lowprice_signalpure_v1 as lp
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_max_audit as ma
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_hard_executor_v3 as hv3
import run_10y_alpha2f_v2 as sim
hv3.patch()

# FROZEN before this audit: winner selected on 2016-2021 only in lowprice_signalpure_v1.
WEIGHTS={'price':.25,'iv':.20,'ef':.20,'rmom':.22,'tstat':.13}
CFG={'liq':.55,'floor':2.0,'hold':90,'n':8,'entry':.10,'keep':.30}
NPHASE=18; SEED=20260823
BASECOLS=strict.BASECOLS+['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy']


def build(out):
    p,cal,members,ua,market_code,bm=mo.build_panel(out,need_fwd=False)
    p=lp.attach_price(p,cal)
    # Signal is created BEFORE adding execution-day limit flags; flags are execution-only.
    q0=lp.rank_signal(p,WEIGHTS,CFG['liq'],CFG['floor'])
    p2=strict.attach_gap_flags(q0,cal,'board')
    keep=[c for c in BASECOLS if c in p2.columns]
    q=p2[keep].copy()
    return p,q,cal,members,ua,market_code,bm


def subset(q,ph):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
    chosen=set(dates[ph::NPHASE])
    z=q[q.signal_date.isin(chosen)].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')


def run_phase(q,ph,cal,members,bm,cash,cost=1.,vp=.05):
    return ma.run_panel(subset(q,ph),cal,members,bm,n=CFG['n'],entry=CFG['entry'],keep=CFG['keep'],cost=cost,initial_cash=float(cash),vol_part=float(vp))


def combine_abs(eqs,initials):
    start=pd.Timestamp(mo.START); idx={start}; ser=[]
    for e,init in zip(eqs,initials):
        s=e.set_index(pd.to_datetime(e.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]
        s=pd.concat([pd.Series({start:float(init)}),s]); s=s[~s.index.duplicated(keep='last')].sort_index(); ser.append(s); idx.update(s.index)
    idx=pd.DatetimeIndex(sorted(idx)); arr=[s.reindex(idx).ffill().fillna(float(init)) for s,init in zip(ser,initials)]
    total=pd.concat(arr,axis=1).sum(axis=1)
    return pd.DataFrame({'trade_date':idx,'equity':total.to_numpy(float)})


def trade_metrics(trades):
    if not trades:return {'trades':0,'win_rate':np.nan,'profit_factor':np.nan,'payoff_ratio':np.nan,'completed_pnl':0.}
    t=pd.concat(trades,ignore_index=True) if trades else pd.DataFrame()
    if t.empty or 'net_pnl' not in t:return {'trades':len(t),'win_rate':np.nan,'profit_factor':np.nan,'payoff_ratio':np.nan,'completed_pnl':0.}
    p=pd.to_numeric(t.net_pnl,errors='coerce').dropna(); w=p[p>0]; l=p[p<0]
    return {'trades':len(t),'win_rate':float((p>0).mean()) if len(p) else np.nan,'profit_factor':float(w.sum()/abs(l.sum())) if len(l) and abs(l.sum())>0 else np.nan,'payoff_ratio':float(w.mean()/abs(l.mean())) if len(w) and len(l) and abs(l.mean())>0 else np.nan,'completed_pnl':float(p.sum())}


def summarize(eq,bm,tradeframes=None):
    s=fa.perf_eq(eq,bm); s['train_2016_2021_return']=fa.period_return(eq,mo.START,mo.TRAIN_END); s['pseudo_oos_2022_2026_return']=fa.period_return(eq,mo.PSEUDO_START,mo.END); s.update(trade_metrics(tradeframes or [])); return s


def run_ensemble(q,phases,cal,members,bm,total_cash=1e6,cost=1.,vp=.05):
    per=float(total_cash)/len(phases); eqs=[]; trs=[]; sts=[]; tms=[]
    for ph in phases:
        st,e,tr,tm=run_phase(q,ph,cal,members,bm,per,cost,vp); eqs.append(e); trs.append(tr.assign(phase=ph) if len(tr) else tr); sts.append(st); tms.append(tm.assign(phase=ph) if len(tm) else tm)
    eq=combine_abs(eqs,[per]*len(phases)); s=summarize(eq,bm,trs); s.update(total_initial_cash=total_cash,sleeves=len(phases),cash_per_sleeve=per,cost_mult_test=cost,volume_participation=vp)
    return s,eq,trs,tms,sts


def tail(eq,trs):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); r=s.pct_change().dropna()
    r5=r.copy(); r5.loc[r5.nlargest(min(5,len(r5))).index]=0.; n=max(1,int(np.ceil(.01*len(r)))); r1=r.copy(); r1.loc[r1.nlargest(n).index]=0.
    t=pd.concat(trs,ignore_index=True) if trs else pd.DataFrame(); pnl=pd.to_numeric(t.net_pnl,errors='coerce').dropna() if len(t) and 'net_pnl' in t else pd.Series(dtype=float)
    return {'base_return':float(s.iloc[-1]/s.iloc[0]-1),'without_best5_days':float((1+r5).prod()-1),'without_best_1pct_days':float((1+r1).prod()-1),'completed_pnl':float(pnl.sum()) if len(pnl) else 0.,'pnl_without_best5_trades':float(pnl.sum()-pnl.nlargest(min(5,len(pnl))).sum()) if len(pnl) else 0.}


def causal_scramble(p):
    # Ranking must not depend on any T+1 execution field.
    rng=np.random.default_rng(SEED); q0=lp.rank_signal(p,WEIGHTS,CFG['liq'],CFG['floor']); p2=p.copy(deep=False)
    for c in ['exec_open','exec_high','exec_low','exec_volume','exec_factor']:
        a=p[c].to_numpy(copy=True); rng.shuffle(a); p2[c]=a
    q1=lp.rank_signal(p2,WEIGHTS,CFG['liq'],CFG['floor']); a=q0.rank_test.to_numpy(float); b=q1.rank_test.to_numpy(float); same_nan=np.array_equal(np.isnan(a),np.isnan(b)); m=np.isfinite(a)&np.isfinite(b); md=float(np.max(np.abs(a[m]-b[m]))) if m.any() else np.nan
    return {'same_nan_pattern':int(same_nan),'max_abs_rank_diff_after_exec_scramble':md,'pass':int(same_nan and md==0.0)}


def noise_q(q,sigma,rng):
    x=q.copy(); v=x.rank_test.to_numpy(float,copy=True); m=np.isfinite(v); v[m]=np.clip(v[m]+rng.normal(0,float(sigma),int(m.sum())),0,1); x['rank_test']=v; x.loc[m,'rank_test']=x.loc[m].groupby('signal_date').rank_test.rank(pct=True,method='average'); return x

def placebo_q(q,rng):
    x=q.copy(); v=x.rank_test.to_numpy(float,copy=True); m=np.isfinite(v); z=np.full(len(v),np.nan); z[m]=rng.random(int(m.sum())); x['rank_test']=z; x.loc[m,'rank_test']=x.loc[m].groupby('signal_date').rank_test.rank(pct=True,method='average'); return x


def core(out,p,q,cal,members,ua,market_code,bm):
    scr=causal_scramble(p); pd.DataFrame([scr]).to_csv(out/'causal_exec_scramble.csv',index=False)
    diag=[]
    for ph in range(NPHASE):
        st,e,tr,tm=run_phase(q,ph,cal,members,bm,1e6); st['phase']=ph; st['train_2016_2021_return']=fa.period_return(e,mo.START,mo.TRAIN_END); st['pseudo_oos_2022_2026_return']=fa.period_return(e,mo.PSEUDO_START,mo.END); diag.append(st)
    pd.DataFrame(diag).to_csv(out/'phase_diagnostic_1m_each.csv',index=False)
    rows=[]; saved={}
    for cash in (1e6,1e7):
        s,e,tr,tm,st=run_ensemble(q,list(range(NPHASE)),cal,members,bm,cash); s['implementation']='all18_exact_split_cash'; rows.append(s); saved[cash]=(e,tr)
    groups={'g0':[0,3,6,9,12,15],'g1':[1,4,7,10,13,16],'g2':[2,5,8,11,14,17]}
    for name,phs in groups.items():
        s,e,tr,tm,st=run_ensemble(q,phs,cal,members,bm,1e6); s.update(implementation='six_sleeve',anchor=name); rows.append(s)
    pd.DataFrame(rows).to_csv(out/'implementations.csv',index=False)
    e,tr=saved[1e6]; fa.annual(e).to_csv(out/'annual_exact_1m_all18.csv',index=False); pd.DataFrame([tail(e,tr)]).to_csv(out/'tail_exact_1m_all18.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        s,e,tr,tm,st=run_ensemble(q,list(range(NPHASE)),cal,members,bm,1e6,cost=cm); costs.append(s)
    pd.DataFrame(costs).to_csv(out/'costs_exact_1m_all18.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'candidate':'LowPrice-Trend frozen phase0 winner','weights':str(WEIGHTS),'config':str(CFG),'selection_lock':'selected on 2016-2021 in prior lowprice research; frozen before this validation','validation_2022_2026_used_in_this_selection':0,'execution':'hard_v3 board-limit proxy; signal-pure; actual split cash; 100-share lots; blocked trade no replacement','causal_exec_scramble_pass':scr['pass'],'timing':'T-close signal to later-open execution'}]).to_csv(out/'audit.csv',index=False)


def capacity(out,q,cal,members,bm):
    rows=[]
    for cash in (1e6,5e6,1e7,5e7,1e8,2e8):
      for vp in (.01,.05):
        s,e,tr,tm,st=run_ensemble(q,list(range(NPHASE)),cal,members,bm,cash,vp=vp); rows.append(s)
    pd.DataFrame(rows).to_csv(out/'capacity_exact_split18.csv',index=False)


def delayed_phase(q,ph,delay,cal,members,bm,cash):
    # rank/signal stays frozen; only execution date/quotes move.
    z=fa.delay_panel(q,CFG['hold'],ph,delay,cal,members); z=strict.attach_gap_flags(z,cal,'board')
    return ma.run_panel(z,cal,members,bm,n=CFG['n'],entry=CFG['entry'],keep=CFG['keep'],initial_cash=float(cash))

def delay(out,q,cal,members,bm):
    rows=[]
    for d in (1,3,5):
        per=1e6/NPHASE; eqs=[]; trs=[]
        for ph in range(NPHASE):
            st,e,tr,tm=delayed_phase(q,ph,d,cal,members,bm,per); eqs.append(e); trs.append(tr.assign(phase=ph) if len(tr) else tr)
        e=combine_abs(eqs,[per]*NPHASE); s=summarize(e,bm,trs); s.update(delay_sessions=d,total_initial_cash=1e6,sleeves=NPHASE); rows.append(s)
    pd.DataFrame(rows).to_csv(out/'delay_exact_split18.csv',index=False)

def noise(out,q,cal,members,bm):
    rng=np.random.default_rng(SEED+11); rows=[]
    for sig in (.02,.05,.10):
      for rep in range(10):
        x=noise_q(q,sig,rng); s,e,tr,tm,st=run_ensemble(x,list(range(NPHASE)),cal,members,bm,1e6); s.update(noise_sigma=sig,rep=rep); rows.append(s)
    pd.DataFrame(rows).to_csv(out/'noise_exact_split18.csv',index=False)

def deletion(out,q,cal,members,bm):
    rng=np.random.default_rng(SEED+12); codes=np.array(sorted(q.code.unique())); rows=[]
    for rep in range(20):
        drop=set(rng.choice(codes,size=int(.20*len(codes)),replace=False).tolist()); x=q[~q.code.isin(drop)].copy(); s,e,tr,tm,st=run_ensemble(x,list(range(NPHASE)),cal,members,bm,1e6); s.update(deleted_share=.20,rep=rep); rows.append(s)
    pd.DataFrame(rows).to_csv(out/'delete20_exact_split18.csv',index=False)

def placebo(out,q,cal,members,bm):
    rng=np.random.default_rng(SEED+13); rows=[]
    for rep in range(60):
        x=placebo_q(q,rng); s,e,tr,tm,st=run_ensemble(x,list(range(NPHASE)),cal,members,bm,1e6); s['rep']=rep; rows.append(s)
    pd.DataFrame(rows).to_csv(out/'placebo_exact_split18.csv',index=False)


def main(mode):
    out=Path(f'results_lowprice_strict_v2_{mode}'); out.mkdir(exist_ok=True); p,q,cal,members,ua,market_code,bm=build(out)
    {'core':lambda:core(out,p,q,cal,members,ua,market_code,bm),'capacity':lambda:capacity(out,q,cal,members,bm),'delay':lambda:delay(out,q,cal,members,bm),'noise':lambda:noise(out,q,cal,members,bm),'delete':lambda:deletion(out,q,cal,members,bm),'placebo':lambda:placebo(out,q,cal,members,bm)}[mode]()
    print('DONE',mode,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('core','capacity','delay','noise','delete','placebo')); a=ap.parse_args(); main(a.mode)
