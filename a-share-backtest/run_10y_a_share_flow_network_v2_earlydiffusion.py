from __future__ import annotations
import warnings, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import run_10y_china_behavior_daily as base
import run_10y_a_share_flow_network as v1
import run_10y_a_share_flow_network_strict as strict

warnings.filterwarnings('ignore')
OUT=Path('results_10y_flow_network_v2_earlydiffusion'); OUT.mkdir(exist_ok=True)
strict.OUT=OUT; v1.OUT=OUT
START=pd.Timestamp('2016-07-29'); END=pd.Timestamp('2026-07-29'); INITIAL=1_000_000.0
SLIPPAGE=.0020; PARTICIPATION=.02; MAX_NAMES=2
# V2 rules are frozen before this run. 04 is the designated primary; 01-03 are ablations only.
VARIANTS=[
 '01_own_attention_early_ablation',
 '02_early_theme_diffusion_ablation',
 '03_early_theme_diffusion_lhb_ablation',
 '04_V2_PRIMARY_early_diffusion_regime_lhb',
]
PERIODS={
 'development_2016H2_2021':(pd.Timestamp('2016-07-29'),pd.Timestamp('2021-12-31')),
 'validation_2022_2023':(pd.Timestamp('2022-01-01'),pd.Timestamp('2023-12-31')),
 'holdout_2024_2026H1':(pd.Timestamp('2024-01-01'),pd.Timestamp('2026-07-29')),
}

def raw_arrays(adj_close,adj_open,adj_high,adj_volume,factor):
    ok=np.isfinite(factor)&(factor>0)
    rp=np.divide(adj_close,factor,out=np.full_like(adj_close,np.nan,float),where=ok)
    ro=np.divide(adj_open,factor,out=np.full_like(adj_open,np.nan,float),where=ok)
    rh=np.divide(adj_high,factor,out=np.full_like(adj_high,np.nan,float),where=ok)
    rv=np.multiply(adj_volume,factor,out=np.full_like(adj_volume,np.nan,float),where=ok)
    return rp,ro,rh,rv

def rolling_st_like(raw_ret,member,window=60):
    # Past-only proxy for historical 5% ST-style trading bands. Conservative exclusion, not a true ST label.
    n,p=raw_ret.shape; out=np.zeros((n,p),bool)
    near=((np.abs(raw_ret)>=.046)&(np.abs(raw_ret)<=.054)).astype(np.int8)
    big=(np.abs(raw_ret)>.065).astype(np.int8)
    csn=np.cumsum(near,axis=0);csb=np.cumsum(big,axis=0)
    for t in range(1,n):
        a=max(0,t-window); nn=csn[t-1]-(csn[a-1] if a>0 else 0); bb=csb[t-1]-(csb[a-1] if a>0 else 0)
        out[t]=(nn>=1)&(bb==0)&member[t]
    return out

def cs_rank_on(ids,arr):
    return v1.pct_rank(np.asarray(arr)[ids])

def compute_early_candidates(t,dates,codes,member,adj_close,raw_close,raw_volume,ret1,mom5,volratio,amount_rank,limitup,st_like,lhb_map):
    if t<65:return pd.DataFrame()
    valid=member[t]&(~st_like[t])&np.isfinite(raw_close[t])&np.isfinite(ret1[t])&np.isfinite(mom5[t])&np.isfinite(volratio[t])&np.isfinite(amount_rank[t])
    valid &= (raw_close[t]>=5.0)&(amount_rank[t]>=.45)&(volratio[t]>=.8)
    ids=np.where(valid)[0]
    if len(ids)<80:return pd.DataFrame()
    # Attention universe is broad; relation graph itself is built from T-60..T-1 only.
    att=.30*v1.pct_rank(np.maximum(ret1[t,ids],-.03))+.25*v1.pct_rank(np.log1p(np.maximum(volratio[t,ids],0)))+.25*v1.pct_rank(amount_rank[t,ids])+.20*v1.pct_rank(np.maximum(mom5[t,ids],-.10))
    take=ids[np.argsort(np.nan_to_num(att,nan=-9))[-120:]]; k=len(take)
    H=ret1[t-60:t,take].copy()
    for r in range(H.shape[0]):
        dt=t-60+r; mm=member[dt]&np.isfinite(ret1[dt]); med=np.nanmedian(ret1[dt,mm]) if mm.any() else 0.0; H[r]-=med
    H=np.nan_to_num(H,nan=0,posinf=0,neginf=0);H-=H.mean(axis=0,keepdims=True);sd=H.std(axis=0,keepdims=True);sd[sd<1e-6]=1;Z=H/sd
    corr=(Z.T@Z)/max(1,H.shape[0]-1);np.fill_diagonal(corr,0);adj=corr>.35;pc=adj.sum(axis=1)
    tr=np.nan_to_num(ret1[t,take],nan=0); vr=np.nan_to_num(volratio[t,take],nan=0); m5=np.nan_to_num(mom5[t,take],nan=0)
    # Seed = today's true attention shock/leader. Target itself should not be a seed.
    seed=limitup[t,take] | ((tr>=.06)&(vr>=1.5))
    strong=((tr>=.035)&(vr>=1.25)).astype(float)
    seed_count=adj@seed.astype(float)
    peer_strength=np.divide(adj@tr,pc,out=np.zeros(k),where=pc>0)
    peer_strong_share=np.divide(adj@strong,pc,out=np.zeros(k),where=pc>0)
    peer_vol=np.divide(adj@np.log1p(np.maximum(vr,0)),pc,out=np.zeros(k),where=pc>0)
    prev3=np.nanmean(ret1[t-3:t,take],axis=0);prev3=np.nan_to_num(prev3,nan=0)
    peer_prev3=np.divide(adj@prev3,pc,out=np.zeros(k),where=pc>0)
    accel=peer_strength-peer_prev3
    lag=peer_strength-tr
    # Early capacity core: theme has visibly moved, stock is participating but has not become the obvious lottery leader yet.
    target=(~seed)&(tr>=.002)&(tr<=.055)&(m5>=-.04)&(m5<=.16)&(vr>=1.15)&(vr<=4.5)&(amount_rank[t,take]>=.60)&(pc>=2)&(seed_count>=1)&(peer_strength>=.022)&(peer_strong_share>=.12)&(accel>=.006)&(lag>=.003)&(lag<=.08)
    cand=np.where(target)[0]
    if len(cand)==0:return pd.DataFrame()
    theme_r=v1.pct_rank(peer_strength[cand]);acc_r=v1.pct_rank(accel[cand]);lag_r=v1.pct_rank(lag[cand]);liq_r=v1.pct_rank(amount_rank[t,take[cand]]);ret_r=v1.pct_rank(tr[cand]);vol_r=v1.pct_rank(vr[cand])
    d=pd.Timestamp(dates[t]).normalize();rows=[]
    for z,j in enumerate(cand):
        idx=int(take[j]); l=lhb_map.get((d,codes[idx])); lhb=0.0
        if l is not None:
            nr=l.get('net_ratio',np.nan)
            if np.isfinite(nr):
                if 0<nr<=10:lhb=.12
                elif nr>15:lhb=-.12
                elif nr< -5:lhb=-.10
            if np.isfinite(l.get('net',np.nan)) and l['net']<0:lhb-=.04
        crowd=max(0,m5[j]-.12)/.12+max(0,vr[j]-3.5)/3.5+max(0,tr[j]-.045)/.045
        crowd=min(crowd,2.0)/2.0
        own=.55*ret_r[z]+.45*vol_r[z]
        network=.34*theme_r[z]+.23*acc_r[z]+.16*lag_r[z]+.17*liq_r[z]+.10*own-.18*crowd
        rows.append({'idx':idx,'code':codes[idx],'own':float(own),'network':float(network),'lhb':float(lhb),'theme_strength':float(peer_strength[j]),'theme_accel':float(accel[j]),'lag':float(lag[j]),'peer_count':int(pc[j]),'seed_count':float(seed_count[j]),'ret1':float(tr[j]),'mom5':float(m5[j]),'volratio':float(vr[j]),'amount_rank':float(amount_rank[t,idx]),'crowd':float(crowd)})
    return pd.DataFrame(rows)

def rank_variant(df,variant):
    if df.empty:return df
    x=df.copy()
    if variant.startswith('01_'):x['score']=x.own-.15*x.crowd
    elif variant.startswith('02_'):x['score']=x.network
    elif variant.startswith('03_'):x['score']=x.network+.08*x.lhb
    else:x['score']=x.network+.08*x.lhb
    return x.sort_values('score',ascending=False)

def period_stats(eq,trades,name,a,b):
    z=eq[(pd.to_datetime(eq.date)>=a)&(pd.to_datetime(eq.date)<=b)].copy()
    if len(z)<2:return {'strategy':name,'period':None}
    s=pd.Series(z.equity.values,index=pd.to_datetime(z.date));r=s.pct_change().fillna(0);total=s.iloc[-1]/s.iloc[0]-1;yrs=max((s.index[-1]-s.index[0]).days/365.25,1/365.25);cagr=(1+total)**(1/yrs)-1 if total>-1 else np.nan;dd=(s/s.cummax()-1).min();sd=r.std();sh=(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan
    tt=trades[(pd.to_datetime(trades.entry_date)>=a)&(pd.to_datetime(trades.entry_date)<=b)] if len(trades) else trades
    return {'strategy':name,'period_return':float(total),'period_cagr':float(cagr),'period_mdd':float(dd),'period_sharpe':float(sh),'period_trades':int(len(tt)),'period_win':float((tt.net_return>0).mean()) if len(tt) else np.nan,'period_pf':float(tt[tt.net_pnl>0].net_pnl.sum()/(-tt[tt.net_pnl<0].net_pnl.sum())) if len(tt) and (tt.net_pnl<0).any() else np.nan}

def main():
    dates,codes,close,open_,high,volume,factor,member,load_audit=base.load_data();dates=pd.DatetimeIndex(dates);codes=list(codes);member=member.astype(bool)
    stock_mask=np.array([bool(base.STOCK_RE.match(c)) for c in codes]);member[:,~stock_mask]=False
    raw_close,raw_open,raw_high,raw_volume=raw_arrays(close,open_,high,volume,factor)
    n,p=close.shape;ret1=np.full_like(close,np.nan,float);ret1[1:]=close[1:]/close[:-1]-1;raw_ret=np.full_like(raw_close,np.nan,float);raw_ret[1:]=raw_close[1:]/raw_close[:-1]-1;mom5=np.full_like(close,np.nan,float);mom5[5:]=close[5:]/close[:-5]-1
    vma20=v1.rolling_mean_mat(raw_volume,20);volratio=np.divide(raw_volume,vma20,out=np.full_like(raw_volume,np.nan,float),where=vma20>0)
    amount=np.maximum(raw_volume,0)*np.maximum(raw_close,0);amount_rank=np.full_like(close,np.nan,float)
    for t in range(n):
        ids=np.where(member[t]&np.isfinite(amount[t]))[0];amount_rank[t,ids]=v1.pct_rank(np.log1p(amount[t,ids]))
    limitup=np.zeros((n,p),bool);limitdn=np.zeros((n,p),bool)
    for t in range(1,n):
        for idx in np.where(member[t]&np.isfinite(raw_ret[t]))[0]:
            lim=v1.get_limit_pct(codes[idx],dates[t]);limitup[t,idx]=raw_ret[t,idx]>=lim*.985;limitdn[t,idx]=raw_ret[t,idx]<=-lim*.985
    st_like=rolling_st_like(raw_ret,member,60)
    market=v1.build_regime(dates,member,ret1,limitup,limitdn,close);market.to_csv(OUT/'market_regime.csv',index=False)
    lhb=strict.fetch_lhb_parallel();lhb_map={}
    if len(lhb):
        for r in lhb.itertuples(index=False):lhb_map[(pd.Timestamp(r.date).normalize(),r.code)]={'net':float(r.lhb_net) if pd.notna(r.lhb_net) else np.nan,'net_ratio':float(r.lhb_net_ratio) if pd.notna(r.lhb_net_ratio) else np.nan}
        lhb.to_csv(OUT/'lhb_raw_safe_fields.csv',index=False)
    start_i=int(np.searchsorted(dates.values,START.to_datetime64()));end_i=int(np.searchsorted(dates.values,END.to_datetime64(),side='right')-1)
    cache={}
    for t in range(max(start_i,65),end_i+1):
        cache[t]=compute_early_candidates(t,dates,codes,member,close,raw_close,raw_volume,ret1,mom5,volratio,amount_rank,limitup,st_like,lhb_map)
        if t%250==0:print('V2 candidate',t,len(cache[t]),flush=True)
    summaries=[];period_rows=[];alltr=[];alleq=[];failed=[];timing=[]
    for variant in VARIANTS:
        print('SIM',variant,flush=True);cash=INITIAL;pos={};pending=[];pending_exits=set();trades=[];eq=[];last_signal={}
        for t in range(start_i,end_i+1):
            d=pd.Timestamp(dates[t]);state=str(market.iloc[t].state)
            # exits decided at previous close
            for idx in list(pending_exits):
                if idx not in pos:continue
                ro=raw_open[t,idx];ao=open_[t,idx]
                if not np.isfinite(ro) or not np.isfinite(ao) or not np.isfinite(raw_close[t-1,idx]) or base.open_locked(codes[idx],d,ro,raw_close[t-1,idx],'sell'):
                    failed.append([variant,d,codes[idx],'sell','open_limit_or_missing']);continue
                P=pos.pop(idx);adj_exec=ao*(1-SLIPPAGE);gross=P.units*adj_exec;f=v1.fee(gross,'sell',d);cash+=gross-f;pnl=(gross-f)-(P.entry_value+P.entry_fee)
                trades.append({'strategy':variant,'code':codes[idx],'signal_date':P.signal_date,'entry_date':P.entry_date,'exit_date':d,'entry_adj_px':P.entry_px,'exit_adj_px':adj_exec,'entry_value':P.entry_value,'net_pnl':pnl,'net_return':pnl/(P.entry_value+P.entry_fee)})
            pending_exits=set()
            # T signal -> T+1 opening auction. A buy limit of +4% vs signal close is predetermined at T close.
            if pending and len(pos)<MAX_NAMES:
                slots=MAX_NAMES-len(pos);equity_ref=cash+sum(P.units*(close[t-1,i] if t>0 and np.isfinite(close[t-1,i]) else P.entry_px) for i,P in pos.items());target_each=equity_ref*(.42 if variant.startswith('04_') else .38)
                accepted=0
                for rec in pending:
                    if accepted>=slots:break
                    idx=int(rec['idx'])
                    if idx in pos or not member[t,idx]:continue
                    ro=raw_open[t,idx];ao=open_[t,idx];prev=raw_close[t-1,idx]
                    if not np.isfinite(ro) or not np.isfinite(ao) or not np.isfinite(prev):failed.append([variant,d,codes[idx],'buy','missing']);continue
                    limit_px=prev*1.04
                    if ro>limit_px:failed.append([variant,d,codes[idx],'buy','predetermined_limit_not_filled']);continue
                    if base.open_locked(codes[idx],d,ro,prev,'buy'):failed.append([variant,d,codes[idx],'buy','open_limit_locked']);continue
                    raw_exec=ro*(1+SLIPPAGE);adj_exec=ao*(1+SLIPPAGE);cap=PARTICIPATION*max(raw_volume[t-1,idx],0)*max(raw_close[t-1,idx],0);budget=min(target_each,cap,cash*.98);shares=int(budget/raw_exec//100*100)
                    if shares<100:continue
                    gross=shares*raw_exec;f=v1.fee(gross,'buy',d)
                    if gross+f>cash:continue
                    econ_units=gross/adj_exec;cash-=gross+f;pos[idx]=v1.Position(econ_units,adj_exec,d,pd.Timestamp(rec['signal_date']),adj_exec,gross,f);timing.append([variant,pd.Timestamp(rec['signal_date']),d,codes[idx],t-int(rec['signal_t'])]);accepted+=1
            pending=[]
            holdings=0
            for idx,P in pos.items():
                cp=close[t,idx]
                if np.isfinite(cp):P.peak=max(P.peak,cp);holdings+=P.units*cp
                else:holdings+=P.units*P.entry_px
            eq.append({'strategy':variant,'date':d,'equity':cash+holdings,'cash':cash,'positions':len(pos)})
            # exits: short information-diffusion horizon, not a long-term trend model
            ctab=cache.get(t,pd.DataFrame())
            for idx,P in list(pos.items()):
                cp=close[t,idx]
                if not np.isfinite(cp):continue
                entry_i=int(np.searchsorted(dates.values,P.entry_date.to_datetime64()));held=t-entry_i;row=ctab[ctab.idx==idx] if len(ctab) else pd.DataFrame();theme_now=float(row.theme_strength.iloc[0]) if len(row) else np.nan
                rnow=cp/P.entry_px-1
                exit_flag=(held>=3 or rnow<=-.06 or rnow>=.12 or cp/max(P.peak,1e-9)-1<=-.05)
                if variant.startswith('04_') and held>=1 and state in ('retreat','climax','ice'):exit_flag=True
                if held>=1 and np.isfinite(theme_now) and theme_now<.005:exit_flag=True
                if exit_flag:pending_exits.add(idx)
            ranked=rank_variant(cache.get(t,pd.DataFrame()),variant)
            # Primary uses only favorable A-share sentiment regimes; ablations relax pieces to show contribution.
            if variant.startswith('04_') and state not in ('repair','main'):ranked=ranked.iloc[0:0]
            if not ranked.empty:
                ranked=ranked[(ranked.network>=.45) if not variant.startswith('01_') else (ranked.own>=.45)]
                picks=[]
                for r in ranked.head(8).to_dict('records'):
                    idx=int(r['idx']);last=last_signal.get(idx,-999)
                    if idx in pos or t-last<5:continue
                    r['signal_date']=d;r['signal_t']=t;picks.append(r);last_signal[idx]=t
                    if len(picks)>=1:break  # one new theme-core order per day maximum
                pending=picks
        eqdf=pd.DataFrame(eq);tdf=pd.DataFrame(trades);st=v1.stats(eqdf,trades);st['strategy']=variant;summaries.append(st);alltr.extend(trades);alleq.append(eqdf)
        for label,(a,b) in PERIODS.items():
            z=period_stats(eqdf,tdf,variant,a,b);z['period']=label;period_rows.append(z)
        print('RESULT',variant,st,flush=True)
    summary=pd.DataFrame(summaries).sort_values('total_return',ascending=False)
    # Dynamic all-stock equal-weight benchmark for context, using prior-day and current-day membership only.
    ew_daily=[];ew_dates=[]
    for t in range(start_i+1,end_i+1):
        mm=member[t-1]&member[t]&np.isfinite(ret1[t]);ew_daily.append(np.nanmean(ret1[t,mm]) if mm.any() else 0);ew_dates.append(dates[t])
    ew=pd.Series(ew_daily,index=pd.DatetimeIndex(ew_dates));summary['dynamic_equal_weight_return']=float((1+ew).prod()-1)
    per=pd.DataFrame(period_rows)
    bench=[]
    for label,(a,b) in PERIODS.items():
        z=ew[(ew.index>=a)&(ew.index<=b)];bench.append((label,float((1+z).prod()-1) if len(z) else np.nan))
    bmap=dict(bench);per['benchmark_equal_weight_return']=per.period.map(bmap)
    summary.to_csv(OUT/'summary.csv',index=False);per.to_csv(OUT/'period_metrics.csv',index=False);pd.concat(alleq,ignore_index=True).to_csv(OUT/'equity.csv',index=False);tdf_all=pd.DataFrame(alltr);tdf_all.to_csv(OUT/'trades.csv',index=False)
    pd.DataFrame(failed,columns=['strategy','date','code','side','reason']).to_csv(OUT/'failed_fills.csv',index=False);tim=pd.DataFrame(timing,columns=['strategy','signal_date','trade_date','code','trade_session_lag']);tim.to_csv(OUT/'timing_audit.csv',index=False)
    # gross pre-friction diagnostics by variant
    gross=[]
    if len(tdf_all):
        sl=SLIPPAGE;tdf_all['gross_pre_friction']=(tdf_all.exit_adj_px/(1-sl))/(tdf_all.entry_adj_px/(1+sl))-1
        for name,g in tdf_all.groupby('strategy'):
            posret=g[g.gross_pre_friction>0].gross_pre_friction.sum();neg=-g[g.gross_pre_friction<0].gross_pre_friction.sum();gross.append([name,len(g),g.gross_pre_friction.mean(),g.gross_pre_friction.median(),(g.gross_pre_friction>0).mean(),posret/neg if neg>0 else np.nan])
    pd.DataFrame(gross,columns=['strategy','trades','gross_mean','gross_median','gross_win_rate','gross_pf_sum_return']).to_csv(OUT/'gross_signal_diagnostics.csv',index=False)
    rb=[]
    if len(tdf_all):
        for name,g in tdf_all.groupby('strategy'):
            g=g.sort_values('net_pnl',ascending=False);pnl=g.net_pnl.sum();rb.append([name,pnl,g.head(5).net_pnl.sum(),pnl-g.head(5).net_pnl.sum(),g.head(10).net_pnl.sum(),pnl-g.head(10).net_pnl.sum()])
    pd.DataFrame(rb,columns=['strategy','completed_pnl','best5_pnl','pnl_without_best5','best10_pnl','pnl_without_best10']).to_csv(OUT/'robustness.csv',index=False)
    audit={'version':'V2 frozen early-diffusion','primary':'04_V2_PRIMARY_early_diffusion_regime_lhb','start':str(START.date()),'end':str(END.date()),'stock_union':int(member[start_i:end_i+1].any(axis=0).sum()),'future_return_fields_used':0,'trade_timing_violations':int((pd.to_datetime(tim.trade_date)<=pd.to_datetime(tim.signal_date)).sum()) if len(tim) else 0,'min_trade_session_lag':int(tim.trade_session_lag.min()) if len(tim) else -1,'same_open_filter_and_fill':0,'entry_rule':'T close signal; precommitted buy limit <=1.04*T raw close; T+1 open fill if crossed','theme_relation_window':'T-60..T-1 residual-return correlation only','theme_shock':'T-day peer strength/seed/acceleration','target':'non-seed moderate-gain liquid capacity core','max_names':MAX_NAMES,'max_new_orders_per_day':1,'max_hold_sessions':3,'slippage_each_side':SLIPPAGE,'news_used':0,'lhb_future_fields_used':0,'st_status':'past-only conservative 5pct-band proxy; not authoritative historical ST label','holdout_note':'V2 parameters were frozen before this run; however broad V1 behavior for 2016-2025 had already been observed, so 2024-2026 is a holdout-like check, not pristine research OOS.'}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    print('=== AUDIT ===');print(pd.DataFrame([audit]).to_string(index=False));print('=== SUMMARY ===');print(summary.to_string(index=False));print('=== PERIODS ===');print(per.to_string(index=False));print('=== GROSS ===');print(pd.read_csv(OUT/'gross_signal_diagnostics.csv').to_string(index=False))

if __name__=='__main__':main()
