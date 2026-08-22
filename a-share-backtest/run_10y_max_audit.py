from __future__ import annotations
from pathlib import Path
import hashlib, inspect, itertools, math
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_skewfilter_hard as hard
import run_10y_grand_opt as grand
import run_10y_balanced_exact as be

OUT=Path("results_max_audit"); OUT.mkdir(exist_ok=True)
SEED=20260822
BASELINE=dict(weight=.60,n=20,hold=60,entry=.10,keep=.30,pool="liq70")
HOLD_GRID=(45,50,55,60,65,70,75)
NOISE_SIGMAS=(.02,.05,.10)
N_NOISE=10
N_DELETE=20
N_PLACEBO=100
BOOT_REPS=2000
RC_REPS=800

def sha(obj):
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()

def minimal(q):
    return q[['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy()

def subset_phase(q,hold,phase=0):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
    step=max(1,round(hold/5))
    chosen=set(dates[phase::step])
    z=minimal(q[q.signal_date.isin(chosen)])
    z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def run_panel(z,cal,members,bm,n=20,entry=.10,keep=.30,cost=1.0,
              initial_cash=1_000_000.,vol_part=.05,slip=.001,start=None,end=None):
    old=(hard.N_HOLD,sim.ENTRY_PCT,sim.KEEP_PCT,sim.INITIAL_CASH,
         sim.VOLUME_PARTICIPATION,sim.SLIPPAGE,sim.START,sim.END)
    start=pd.Timestamp(start if start is not None else old[-2])
    end=pd.Timestamp(end if end is not None else old[-1])
    hard.N_HOLD=n; sim.ENTRY_PCT=entry; sim.KEEP_PCT=keep
    sim.INITIAL_CASH=float(initial_cash); sim.VOLUME_PARTICIPATION=float(vol_part); sim.SLIPPAGE=float(slip)
    sim.START=start; sim.END=end
    try:
        zz=z[(pd.to_datetime(z.signal_date)>=start)&(pd.to_datetime(z.trade_date)<=end)].copy()
        b=bm.loc[start:end]
        eq,tr,tm,to=hard.hard_simulate(zz,cal,members,cost)
        st=sim.perf(eq,tr,to,b)
        st.update(dict(n_hold=n,entry_pct=entry,keep_pct=keep,cost_mult=cost,
                       initial_cash=initial_cash,volume_participation=vol_part,slippage=slip,
                       positions_max=int(eq.positions.max()),positions_median=float(eq.positions.median())))
        return st,eq,tr,tm
    finally:
        hard.N_HOLD,sim.ENTRY_PCT,sim.KEEP_PCT,sim.INITIAL_CASH,sim.VOLUME_PARTICIPATION,sim.SLIPPAGE,sim.START,sim.END=old

def run_q(q,hold,phase,cal,members,bm,**kw):
    return run_panel(subset_phase(q,hold,phase),cal,members,bm,**kw)

def eqret(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index()
    return s.pct_change().dropna()

def simple_stats(r):
    if len(r)<20: return dict(cagr=np.nan,sharpe=np.nan,total=np.nan)
    years=len(r)/252.0
    total=float((1+r).prod()-1)
    cagr=float((1+total)**(1/years)-1) if total>-1 else -1.
    sd=float(r.std())
    sh=float(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan
    return dict(total=total,cagr=cagr,sharpe=sh)

def phase_audit(q,cal,members,bm):
    rows=[]; step=round(BASELINE['hold']/5)
    for ph in range(step):
        print("PHASE",ph,flush=True)
        st,_,_,_=run_q(q,BASELINE['hold'],ph,cal,members,bm,n=20,entry=.10,keep=.30)
        rows.append({**st,'phase':ph})
    z=pd.DataFrame(rows); z.to_csv(OUT/'phase_offsets.csv',index=False); return z

def hold_audit(q,cal,members,bm):
    rows=[]
    for h in HOLD_GRID:
        print("HOLD",h,flush=True)
        st,_,_,_=run_q(q,h,0,cal,members,bm,n=20,entry=.10,keep=.30)
        rows.append({**st,'hold_days_nominal':h})
    z=pd.DataFrame(rows); z.to_csv(OUT/'hold_grid.csv',index=False); return z

def noise_audit(q,cal,members,bm):
    rng=np.random.default_rng(SEED); rows=[]; q0=minimal(q); finite=np.isfinite(q0.rank_test.to_numpy(float))
    for sig in NOISE_SIGMAS:
        for k in range(N_NOISE):
            x=q0.copy(); vals=x.rank_test.to_numpy(float)
            vals[finite]=np.clip(vals[finite]+rng.normal(0,sig,finite.sum()),0,1); x['rank_test']=vals
            x.loc[finite,'rank_test']=x.loc[finite].groupby('signal_date').rank_test.rank(pct=True,method='average')
            st,_,_,_=run_panel(subset_phase(x,60,0),cal,members,bm); rows.append({**st,'noise_sigma':sig,'seed':k})
    z=pd.DataFrame(rows); z.to_csv(OUT/'rank_noise.csv',index=False); return z

def deletion_audit(q,cal,members,bm):
    rng=np.random.default_rng(SEED+1); codes=np.array(sorted(q.code.unique())); rows=[]
    for k in range(N_DELETE):
        drop=set(rng.choice(codes,size=int(.20*len(codes)),replace=False).tolist())
        st,_,_,_=run_q(q[~q.code.isin(drop)],60,0,cal,members,bm)
        rows.append({**st,'seed':k,'deleted_share':.20})
    z=pd.DataFrame(rows); z.to_csv(OUT/'random_delete20.csv',index=False); return z

def placebo_audit(q,cal,members,bm):
    rng=np.random.default_rng(SEED+2); rows=[]; q0=minimal(q); mask=np.isfinite(q0.rank_test.to_numpy(float))
    for k in range(N_PLACEBO):
        x=q0.copy(); a=np.full(len(x),np.nan); a[mask]=rng.random(mask.sum()); x['rank_test']=a
        st,_,_,_=run_panel(subset_phase(x,60,0),cal,members,bm); rows.append({**st,'seed':k})
        if k%20==0: print("PLACEBO",k,flush=True)
    z=pd.DataFrame(rows); z.to_csv(OUT/'placebo_random_rank.csv',index=False); return z

def reverse_audit(q,cal,members,bm):
    x=minimal(q); m=np.isfinite(x.rank_test); x.loc[m,'rank_test']=1-x.loc[m,'rank_test']
    st,_,_,_=run_panel(subset_phase(x,60,0),cal,members,bm)
    pd.DataFrame([{**st,'test':'reverse_rank'}]).to_csv(OUT/'reverse_rank.csv',index=False); return st

def delay_panel(q,delay,cal,members):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[::12]); z=minimal(q[q.signal_date.isin(chosen)]).copy()
    target={}
    for d in chosen:
        k=cal.searchsorted(pd.Timestamp(d),side='right')+(delay-1); target[pd.Timestamp(d)]=cal[k] if k<len(cal) else pd.NaT
    z['trade_date']=pd.to_datetime(z.signal_date).map(target)
    for c in ['exec_open','exec_high','exec_low','exec_volume','exec_factor']: z[c]=np.nan
    first=members.groupby('code').start.min().to_dict(); last=members.groupby('code').end.max().to_dict()
    for i,(code,idx) in enumerate(z.groupby('code').groups.items(),1):
        idx=np.asarray(list(idx)); ds=pd.DatetimeIndex(z.loc[idx,'trade_date'])
        for fld,out in [('open','exec_open'),('high','exec_high'),('low','exec_low'),('volume','exec_volume'),('factor','exec_factor')]:
            s=base.qb.read_bin(code,fld,cal)
            if fld=='factor' and s.empty: z.loc[idx,out]=1.0
            elif not s.empty: z.loc[idx,out]=s.reindex(ds).to_numpy(float)
        active=(ds>=pd.Timestamp(first.get(code,'1900-01-01')))&(ds<=pd.Timestamp(last.get(code,'2100-01-01')))
        bad=np.array(~active)|pd.isna(ds)
        if bad.any(): z.loc[idx[bad],['exec_open','exec_high','exec_low','exec_volume']]=np.nan
        if i%1000==0: print("DELAY BUILD",delay,i,flush=True)
    z['exec_factor']=z.exec_factor.replace(0,np.nan).fillna(1.0)
    z=z[np.isfinite(z[['exec_open','exec_high','exec_low','exec_volume']]).all(axis=1)]
    z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')

def delay_audit(q,cal,members,bm):
    rows=[]
    for d in (1,2,3,5):
        print("DELAY",d,flush=True); z=delay_panel(q,d,cal,members)
        st,_,_,_=run_panel(z,cal,members,bm); rows.append({**st,'delay_sessions':d})
    x=pd.DataFrame(rows); x.to_csv(OUT/'execution_delay.csv',index=False); return x

def capacity_audit(q,cal,members,bm):
    rows=[]; z=subset_phase(q,60,0)
    for cash in (1e6,5e6,1e7,5e7,1e8):
        for vp in (.01,.05):
            print("CAP",cash,vp,flush=True); st,_,_,_=run_panel(z,cal,members,bm,initial_cash=cash,vol_part=vp)
            rows.append({**st,'cash_test':cash,'vp_test':vp})
    x=pd.DataFrame(rows); x.to_csv(OUT/'capacity.csv',index=False); return x

def date_window_audit(q,cal,members,bm):
    rows=[]
    starts=['2016-07-29','2016-10-31','2017-01-31','2017-04-28','2017-07-31','2018-01-31']
    for s in starts:
        st,_,_,_=run_q(q,60,0,cal,members,bm,start=s,end='2026-07-29'); rows.append({**st,'kind':'start_shift','window_start':s,'window_end':'2026-07-29'})
    windows=[('2016-07-29','2021-07-30'),('2017-07-31','2022-07-29'),('2018-07-31','2023-07-31'),('2019-07-31','2024-07-31'),('2020-07-31','2025-07-31'),('2021-07-30','2026-07-29')]
    for s,e in windows:
        st,_,_,_=run_q(q,60,0,cal,members,bm,start=s,end=e); rows.append({**st,'kind':'rolling_5y','window_start':s,'window_end':e})
    x=pd.DataFrame(rows); x.to_csv(OUT/'date_windows.csv',index=False); return x

def moving_block_bootstrap(r,reps=BOOT_REPS,block=20):
    rng=np.random.default_rng(SEED+5); arr=np.asarray(r,float); n=len(arr); out=[]
    for _ in range(reps):
        idx=[]
        while len(idx)<n:
            s=int(rng.integers(0,n)); idx.extend([(s+j)%n for j in range(block)])
        out.append(simple_stats(pd.Series(arr[np.array(idx[:n])])))
    return pd.DataFrame(out)

def bootstrap_audit(eq):
    r=eqret(eq); rows=[]
    for block in (20,60):
        z=moving_block_bootstrap(r,BOOT_REPS,block)
        for metric in ('total','cagr','sharpe'):
            rows.append({'block':block,'metric':metric,'p2_5':z[metric].quantile(.025),'median':z[metric].median(),'p97_5':z[metric].quantile(.975)})
    x=pd.DataFrame(rows); x.to_csv(OUT/'bootstrap_ci.csv',index=False); return x

def candidate_family(p,cal,members,bm):
    returns={}; rows=[]
    for w in (.55,.60,.65):
        q=be.anchor_weighted(p,'liq70',w)
        for n in (15,20,25):
            for h in (50,60,70):
                for e,k in ((.05,.20),(.10,.30),(.15,.40)):
                    name=f'w{w:.2f}_n{n}_h{h}_e{e:.2f}_k{k:.2f}'
                    st,eq,_,_=run_q(q,h,0,cal,members,bm,n=n,entry=e,keep=k)
                    rows.append({**st,'candidate':name,'weight':w,'hold':h}); returns[name]=eqret(eq).rename(name)
    pd.DataFrame(rows).to_csv(OUT/'candidate_family.csv',index=False)
    R=pd.concat(returns.values(),axis=1,join='inner').dropna(); R.to_csv(OUT/'candidate_daily_returns.csv')
    return R,pd.DataFrame(rows)

def pbo_test(R,nblocks=10):
    cuts=np.array_split(np.arange(len(R)),nblocks); vals=[]
    for train_blocks in itertools.combinations(range(nblocks),nblocks//2):
        tr=np.concatenate([cuts[i] for i in train_blocks]); te=np.concatenate([cuts[i] for i in range(nblocks) if i not in train_blocks])
        atr=R.iloc[tr].mean()/R.iloc[tr].std(); ate=R.iloc[te].mean()/R.iloc[te].std(); win=atr.idxmax(); vals.append(float(ate.rank(pct=True,method='average')[win]))
    a=np.array(vals); return {'pbo':float(np.mean(a<.5)),'median_oos_percentile':float(np.median(a)),'splits':len(a),'p10_oos_percentile':float(np.quantile(a,.10))}

def dsr_one(r,candidate_sharpes,ntrials):
    x=np.asarray(r,float); T=len(x); sr=float(np.mean(x)/np.std(x,ddof=1)); srs=np.asarray(candidate_sharpes,float)/np.sqrt(252.0); sigma_sr=float(np.std(srs,ddof=1))
    gamma=0.5772156649015329; emax=(1-gamma)*norm.ppf(1-1/ntrials)+gamma*norm.ppf(1-1/(ntrials*math.e)); sr0=sigma_sr*emax
    sk=float(skew(x,bias=False)); ku=float(kurtosis(x,fisher=False,bias=False)); den=math.sqrt(max(1e-12,1-sk*sr+(ku-1)*sr*sr/4)); z=(sr-sr0)*math.sqrt(max(1,T-1))/den
    return {'n_trials':ntrials,'sr_hat_ann':sr*np.sqrt(252),'sr0_ann':sr0*np.sqrt(252),'dsr_prob':float(norm.cdf(z)),'z':float(z),'skew':sk,'kurtosis':ku,'T':T}

def reality_check(R,bm,reps=RC_REPS,block=20):
    B=bm.pct_change(fill_method=None).reindex(R.index).fillna(0.0); X=R.sub(B,axis=0); mu=X.mean(); sd=X.std().replace(0,np.nan); obs=float((np.sqrt(len(X))*mu/sd).max()); centered=X-mu
    A=centered.to_numpy(float); n=len(A); rng=np.random.default_rng(SEED+6); mx=[]
    for _ in range(reps):
        idx=[]
        while len(idx)<n:
            s=int(rng.integers(0,n)); idx.extend([(s+j)%n for j in range(block)])
        Z=A[np.array(idx[:n])]; m=Z.mean(axis=0); d=Z.std(axis=0,ddof=1); t=np.sqrt(n)*m/np.where(d>0,d,np.nan); mx.append(np.nanmax(t))
    return {'observed_max_t':obs,'bootstrap_p':float(np.mean(np.asarray(mx)>=obs)),'reps':reps,'block':block,'candidates':X.shape[1]}

def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members)
    bm=market_close.loc[sim.START:sim.END].dropna(); q=be.anchor_weighted(p,'liq70',.60)
    base_st,base_eq,base_tr,base_tm=run_q(q,60,0,cal,members,bm); pd.DataFrame([base_st]).to_csv(OUT/'baseline.csv',index=False); sim.annual_returns(base_eq).to_csv(OUT/'annual.csv',index=False)
    phase=phase_audit(q,cal,members,bm); hold=hold_audit(q,cal,members,bm); delay=delay_audit(q,cal,members,bm); noise=noise_audit(q,cal,members,bm); dele=deletion_audit(q,cal,members,bm); plac=placebo_audit(q,cal,members,bm); reverse_audit(q,cal,members,bm); cap=capacity_audit(q,cal,members,bm); windows=date_window_audit(q,cal,members,bm); boot=bootstrap_audit(base_eq)
    R,cands=candidate_family(p,cal,members,bm); pbo=pbo_test(R); sharpes=cands.sharpe.to_numpy(float); dsr=[dsr_one(eqret(base_eq),sharpes,n) for n in (len(sharpes),500,1000)]; pd.DataFrame(dsr).to_csv(OUT/'deflated_sharpe.csv',index=False); pd.DataFrame([pbo]).to_csv(OUT/'pbo.csv',index=False); rc=reality_check(R,bm); pd.DataFrame([rc]).to_csv(OUT/'reality_check.csv',index=False)
    placebo_p=float((plac.total_return>=base_st['total_return']).mean()); placebo_sharpe_p=float((plac.sharpe>=base_st['sharpe']).mean())
    gates={'timing_zero':int(len(base_tm)==0 or (pd.to_datetime(base_tm.signal_date)<pd.to_datetime(base_tm.trade_date)).all()),'position_cap':int(base_st['positions_max']<=20),'all_12_phases_positive':int((phase.total_return>0).all()),'phase_median_cagr_ge_8pct':int(phase.cagr.median()>=.08),'hold_45_75_all_positive':int((hold.total_return>0).all()),'delay_3d_positive':int(float(delay.loc[delay.delay_sessions==3,'total_return'].iloc[0])>0),'random_delete_all_positive':int((dele.total_return>0).all()),'noise10_median_positive':int(noise.loc[noise.noise_sigma==.10,'total_return'].median()>0),'placebo_return_p_le_5pct':int(placebo_p<=.05),'placebo_sharpe_p_le_5pct':int(placebo_sharpe_p<=.05),'pbo_le_25pct':int(pbo['pbo']<=.25),'dsr_1000_ge_95pct':int(dsr[-1]['dsr_prob']>=.95),'reality_check_p_le_5pct':int(rc['bootstrap_p']<=.05),'bootstrap60_cagr_lower_gt_0':int(float(boot[(boot.block==60)&(boot.metric=='cagr')].p2_5.iloc[0])>0),'rolling5y_all_positive':int((windows.loc[windows.kind=='rolling_5y','total_return']>0).all()),'capacity_10m_1pct_positive':int(float(cap[(cap.cash_test==1e7)&(cap.vp_test==.01)].total_return.iloc[0])>0)}
    gate_df=pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]); gate_df.to_csv(OUT/'gates.csv',index=False)
    manifest={**ua,'market_factor':market_code,'strategy':'Anti-Lottery Low-IVOL Liq70','frozen_rule':'liq top70%; remove highest 20% skew40; score=.60 low-IVOL60 rank + .40 efficiency120 rank; N20; 60d; entry10 keep30; next-open','code_hash_pool':sha(grand.pool_mask),'code_hash_rank':sha(be.anchor_weighted),'code_hash_choose':sha(hard.choose_det),'code_hash_exec':sha(hard.hard_simulate),'code_hash_fee':sha(sim.fee),'baseline_total_return':base_st['total_return'],'baseline_cagr':base_st['cagr'],'baseline_mdd':base_st['max_drawdown'],'placebo_return_p':placebo_p,'placebo_sharpe_p':placebo_sharpe_p,'pbo':pbo['pbo'],'dsr1000':dsr[-1]['dsr_prob'],'reality_check_p':rc['bootstrap_p'],'internal_hard_pass':int(gate_df['pass'].all()),'gates_passed':int(gate_df['pass'].sum()),'gates_total':len(gate_df)}
    pd.DataFrame([manifest]).to_csv(OUT/'manifest_and_verdict.csv',index=False)
    print('=== MANIFEST ==='); print(pd.DataFrame([manifest]).to_string(index=False),flush=True); print('=== GATES ==='); print(gate_df.to_string(index=False),flush=True); print('=== PHASE ==='); print(phase[['phase','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True); print('=== HOLD ==='); print(hold[['hold_days_nominal','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True); print('=== DELAY ==='); print(delay[['delay_sessions','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True); print('=== DSR ==='); print(pd.DataFrame(dsr).to_string(index=False),flush=True); print('=== PBO ==='); print(pd.DataFrame([pbo]).to_string(index=False),flush=True); print('=== REALITY CHECK ==='); print(pd.DataFrame([rc]).to_string(index=False),flush=True)
    if not gates['timing_zero'] or not gates['position_cap']: raise RuntimeError('FATAL execution audit failure')

if __name__=='__main__': main()
