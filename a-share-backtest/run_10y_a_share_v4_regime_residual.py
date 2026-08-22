from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import run_10y_china_behavior_daily as base
import run_10y_a_share_flow_network as v1
import run_10y_a_share_flow_network_strict as strict
import run_10y_a_share_flow_network_v2_earlydiffusion as v2

warnings.filterwarnings('ignore')
OUT=Path('results_10y_v4_regime_residual'); OUT.mkdir(exist_ok=True)
strict.OUT=OUT; v1.OUT=OUT
START=pd.Timestamp('2016-07-29'); DEV_END=pd.Timestamp('2021-12-31'); VAL_END=pd.Timestamp('2023-12-31'); END=pd.Timestamp('2026-07-29')
INITIAL=1_000_000.; HOLD=2; ENTRY_SLIP=.003; EXIT_SLIP=.002; PARTICIPATION=.02; MAX_NAMES=4; MAX_NEW=2
# Frozen V4 before validation/holdout results. No parameter search on 2022+.
FEATURES=['r_ret1','r_mom5','r_mom20','r_volratio','r_amount','r_price','r_vol20','r_high20','limitup','streak','lhb_flag','lhb_net_ratio','auction_gap','r_auction_gap']


def roll_std(a,w):
    df=pd.DataFrame(a); return df.rolling(w,min_periods=max(5,w//2)).std().to_numpy()

def roll_max(a,w):
    df=pd.DataFrame(a); return df.rolling(w,min_periods=max(5,w//2)).max().to_numpy()

def cross_rank_row(arr,valid):
    out=np.full(arr.shape,np.nan,float); ids=np.where(valid & np.isfinite(arr))[0]
    if len(ids): out[ids]=v1.pct_rank(arr[ids])
    return out

def market_eligible(m,mom5_value=None):
    def z(name,default=np.nan):
        x=getattr(m,name); return default if pd.isna(x) else float(x)
    bp=z('board_premium'); med=z('median_ret'); br=z('breadth20'); dn=z('limitdn_share')
    # Frozen from V3 development tree positive leaves, not re-estimated here.
    if np.isfinite(bp) and bp<=-.01: return True
    if np.isfinite(bp) and bp>-.01 and np.isfinite(med) and med<=.01 and np.isfinite(br) and br<=.18 and np.isfinite(dn) and dn<=.01: return True
    if np.isfinite(med) and med>.01 and np.isfinite(dn) and dn<=0 and mom5_value is not None and mom5_value<=.04: return True
    return False

def build_X(t,ids,raw_close,raw_open,ret1,mom5,mom20,volratio,amount_rank,vol20,high20,limitup,streak,lhb_map,dates,codes):
    gap=raw_open[t+1,ids]/raw_close[t,ids]-1
    valid_all=np.isfinite(ret1[t])
    def rvals(arr):
        row=cross_rank_row(arr,valid_all); return row[ids]
    X=np.column_stack([
        rvals(ret1[t]),rvals(mom5[t]),rvals(mom20[t]),rvals(np.log1p(np.maximum(volratio[t],0))),amount_rank[t,ids],rvals(np.log(np.maximum(raw_close[t],.5))),rvals(-vol20[t]),rvals(raw_close[t]/high20[t]-1),limitup[t,ids].astype(float),np.minimum(streak[t,ids],4).astype(float),np.zeros(len(ids)),np.zeros(len(ids)),gap,v1.pct_rank(gap)
    ])
    d=pd.Timestamp(dates[t]).normalize()
    for k,idx in enumerate(ids):
        l=lhb_map.get((d,codes[idx]))
        if l is not None:
            X[k,10]=1.; nr=l.get('net_ratio',np.nan); X[k,11]=0 if not np.isfinite(nr) else float(np.clip(nr,-30,30))/30
    return X

def benchmark_return(a,b,dates,member,close):
    i0=max(1,int(np.searchsorted(dates.values,a.to_datetime64()))); i1=int(np.searchsorted(dates.values,b.to_datetime64(),side='right')-1); wealth=1.
    for t in range(i0,i1+1):
        m=member[t]&member[t-1]&np.isfinite(close[t])&np.isfinite(close[t-1])&(close[t-1]>0)
        if m.any(): wealth*=1+float(np.nanmean(close[t,m]/close[t-1,m]-1))
    return wealth-1

def fit_models(X,y):
    sc=StandardScaler(); Xs=sc.fit_transform(X); ridge=Ridge(alpha=20.0); ridge.fit(Xs,y)
    hgb=HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=500,l2_regularization=10.,random_state=20260819); hgb.fit(X,y)
    return sc,ridge,hgb

def predict(models,X):
    sc,ridge,hgb=models; p1=ridge.predict(sc.transform(X)); p2=hgb.predict(X)
    # Cross-sectional standardization before ensemble to prevent scale dominance.
    def zs(p):
        s=np.std(p); return (p-np.mean(p))/(s if s>1e-9 else 1)
    return .5*zs(p1)+.5*zs(p2),p1,p2

def run_period(name,a,b,models,dates,codes,member,close,open_,raw_close,raw_open,raw_volume,ret1,mom5,mom20,volratio,amount_rank,vol20,high20,limitup,streak,lhb_map,market):
    i0=max(65,int(np.searchsorted(dates.values,a.to_datetime64()))); i1=min(int(np.searchsorted(dates.values,b.to_datetime64(),side='right')-1),len(dates)-HOLD-2)
    cash=INITIAL; pos={}; eq=[]; trades=[]; failed=[]
    for t in range(i0,i1+HOLD+2):
        d=pd.Timestamp(dates[t])
        # exits
        for idx,P in list(pos.items()):
            if t<P['exit_idx']: continue
            ro=raw_open[t,idx]; ao=open_[t,idx]; prev=raw_close[t-1,idx]
            if not np.isfinite(ro) or not np.isfinite(ao) or not np.isfinite(prev) or base.open_locked(codes[idx],d,ro,prev,'sell'):
                P['exit_idx']=t+1; failed.append((name,d,codes[idx],'sell','locked_or_missing')); continue
            adj_exec=ao*(1-EXIT_SLIP); gross=P['econ_units']*adj_exec; f=v1.fee(gross,'sell',d); cash+=gross-f; pnl=(gross-f)-(P['entry_value']+P['entry_fee'])
            trades.append((name,codes[idx],P['signal_date'],P['entry_date'],d,P['entry_value'],pnl,pnl/(P['entry_value']+P['entry_fee']),P['score']))
            del pos[idx]
        if t>i1:
            mtm=cash+sum(P['econ_units']*(close[t,idx] if np.isfinite(close[t,idx]) else P['adj_entry']) for idx,P in pos.items()); eq.append((d,mtm)); continue
        sig=t-1
        if sig>=65 and len(pos)<MAX_NAMES:
            mkt=market.iloc[sig]
            valid=member[sig]&member[t]&np.isfinite(raw_close[sig])&np.isfinite(raw_open[t])&np.isfinite(open_[t])&np.isfinite(ret1[sig])&np.isfinite(mom5[sig])&np.isfinite(mom20[sig])&np.isfinite(volratio[sig])&np.isfinite(amount_rank[sig])&np.isfinite(vol20[sig])&np.isfinite(high20[sig])
            valid &= (raw_close[sig]>=3)&(amount_rank[sig]>=.25)&(volratio[sig]>=.5)
            ids=np.where(valid)[0]
            if len(ids):
                # V3 market gate; third leaf is stock-specific via mom5.
                elig=np.array([market_eligible(mkt,float(mom5[sig,idx])) for idx in ids])
                ids=ids[elig]
            if len(ids):
                X=build_X(sig,ids,raw_close,raw_open,ret1,mom5,mom20,volratio,amount_rank,vol20,high20,limitup,streak,lhb_map,dates,codes); ok=np.isfinite(X).all(axis=1); ids=ids[ok]; X=X[ok]
                if len(ids):
                    score,pr,ph=predict(models,X); # do not chase extreme opening gaps
                    ok=(X[:,12]>-.08)&(X[:,12]<.095)&(~limitup[sig,ids] | (X[:,12]<.03))
                    cand=np.where(ok)[0]
                    if len(cand):
                        # top 2% cross-sectional residual score; no absolute threshold tuning
                        cut=np.quantile(score[cand],.98) if len(cand)>=50 else np.max(score[cand])
                        cand=cand[score[cand]>=cut]; order=cand[np.argsort(score[cand])[::-1]]
                        slots=min(MAX_NEW,MAX_NAMES-len(pos)); added=0; equity_ref=cash+sum(P['econ_units']*(close[t-1,idx] if np.isfinite(close[t-1,idx]) else P['adj_entry']) for idx,P in pos.items()); target=equity_ref*.24
                        for q in order:
                            if added>=slots: break
                            idx=int(ids[q]); ro=raw_open[t,idx]; ao=open_[t,idx]; prev=raw_close[t-1,idx]
                            if idx in pos or base.open_locked(codes[idx],d,ro,prev,'buy'): failed.append((name,d,codes[idx],'buy','locked_or_existing')); continue
                            raw_exec=ro*(1+ENTRY_SLIP); adj_exec=ao*(1+ENTRY_SLIP); cap=PARTICIPATION*max(raw_volume[t-1,idx],0)*max(raw_close[t-1,idx],0); budget=min(target,cap,cash*.98); shares=int(budget/raw_exec//100*100)
                            if shares<100: continue
                            raw_gross=shares*raw_exec; f=v1.fee(raw_gross,'buy',d)
                            if raw_gross+f>cash: continue
                            econ_units=raw_gross/adj_exec; cash-=raw_gross+f; pos[idx]=dict(econ_units=econ_units,adj_entry=adj_exec,entry_date=d,exit_idx=t+HOLD,signal_date=pd.Timestamp(dates[sig]),entry_value=raw_gross,entry_fee=f,score=float(score[q])); added+=1
        mtm=cash+sum(P['econ_units']*(close[t,idx] if np.isfinite(close[t,idx]) else P['adj_entry']) for idx,P in pos.items()); eq.append((d,mtm))
    eq=pd.DataFrame(eq,columns=['date','equity']); tr=pd.DataFrame(trades,columns=['period','code','signal_date','entry_date','exit_date','entry_value','net_pnl','net_return','score']); ff=pd.DataFrame(failed,columns=['period','date','code','side','reason']); return eq,tr,ff

def stats(eq,tr):
    s=pd.Series(eq.equity.values,index=pd.to_datetime(eq.date)); r=s.pct_change().fillna(0); total=s.iloc[-1]/s.iloc[0]-1; yrs=max((s.index[-1]-s.index[0]).days/365.25,1/365.25); cagr=(1+total)**(1/yrs)-1 if total>-1 else np.nan; dd=(s/s.cummax()-1).min(); sd=r.std(); sh=r.mean()/sd*np.sqrt(252) if sd>0 else np.nan
    if len(tr): win=(tr.net_return>0).mean(); neg=-tr.loc[tr.net_pnl<0,'net_pnl'].sum(); pf=tr.loc[tr.net_pnl>0,'net_pnl'].sum()/neg if neg>0 else np.nan
    else: win=pf=np.nan
    return dict(final_asset=float(s.iloc[-1]),total_return=float(total),cagr=float(cagr),max_drawdown=float(dd),sharpe=float(sh),trades=len(tr),win_rate=float(win) if pd.notna(win) else np.nan,profit_factor=float(pf) if pd.notna(pf) else np.nan)

def main():
    dates,codes,close,open_,high,volume,factor,member,_=base.load_data(); dates=pd.DatetimeIndex(dates); codes=list(codes); member=member.astype(bool); stock_mask=np.array([bool(base.STOCK_RE.match(c)) for c in codes]); member[:,~stock_mask]=False
    raw_close,raw_open,raw_high,raw_volume=v2.raw_arrays(close,open_,high,volume,factor); n,p=close.shape
    ret1=np.full_like(close,np.nan);ret1[1:]=close[1:]/close[:-1]-1; mom5=np.full_like(close,np.nan);mom5[5:]=close[5:]/close[:-5]-1; mom20=np.full_like(close,np.nan);mom20[20:]=close[20:]/close[:-20]-1
    vol20=roll_std(ret1,20); high20=roll_max(raw_close,20); vma20=v1.rolling_mean_mat(raw_volume,20); volratio=np.divide(raw_volume,vma20,out=np.full_like(raw_volume,np.nan),where=vma20>0); amount=raw_volume*raw_close; amount_rank=np.full_like(close,np.nan)
    for t in range(n):
        ids=np.where(member[t]&np.isfinite(amount[t]))[0]; amount_rank[t,ids]=v1.pct_rank(np.log1p(np.maximum(amount[t,ids],0)))
    raw_ret=np.full_like(raw_close,np.nan);raw_ret[1:]=raw_close[1:]/raw_close[:-1]-1; limitup=np.zeros((n,p),bool);limitdn=np.zeros((n,p),bool);streak=np.zeros((n,p),np.int8)
    for t in range(1,n):
        for idx in np.where(member[t]&np.isfinite(raw_ret[t]))[0]:
            lim=v1.get_limit_pct(codes[idx],dates[t]);limitup[t,idx]=raw_ret[t,idx]>=lim*.985;limitdn[t,idx]=raw_ret[t,idx]<=-lim*.985;streak[t,idx]=min(4,int(streak[t-1,idx])+1) if limitup[t,idx] else 0
    market=v1.build_regime(dates,member,ret1,limitup,limitdn,close);market.to_csv(OUT/'market_regime.csv',index=False)
    lhb=strict.fetch_lhb_parallel();lhb_map={}
    if len(lhb):
        lhb.to_csv(OUT/'lhb_raw_safe_fields.csv',index=False)
        for r in lhb.itertuples(index=False):lhb_map[(pd.Timestamp(r.date).normalize(),r.code)]={'net':float(r.lhb_net) if pd.notna(r.lhb_net) else np.nan,'net_ratio':float(r.lhb_net_ratio) if pd.notna(r.lhb_net_ratio) else np.nan}
    # Development-only residual training
    rng=np.random.default_rng(20260819); Xs=[];ys=[]; s0=max(65,int(np.searchsorted(dates.values,START.to_datetime64())));s1=min(int(np.searchsorted(dates.values,DEV_END.to_datetime64(),side='right')-1),n-HOLD-2)
    for t in range(s0,s1+1):
        valid=member[t]&member[t+1]&np.isfinite(raw_close[t])&np.isfinite(raw_open[t+1])&np.isfinite(open_[t+1])&np.isfinite(open_[t+1+HOLD])&np.isfinite(ret1[t])&np.isfinite(mom5[t])&np.isfinite(mom20[t])&np.isfinite(volratio[t])&np.isfinite(amount_rank[t])&np.isfinite(vol20[t])&np.isfinite(high20[t]);valid&=(raw_close[t]>=3)&(amount_rank[t]>=.20)&(volratio[t]>=.5)
        ids=np.where(valid)[0]
        if len(ids)>400:ids=rng.choice(ids,400,replace=False)
        if not len(ids):continue
        X=build_X(t,ids,raw_close,raw_open,ret1,mom5,mom20,volratio,amount_rank,vol20,high20,limitup,streak,lhb_map,dates,codes); fut=open_[t+1+HOLD,ids]/open_[t+1,ids]-1
        # cross-sectional residual label removes market beta/day effect
        y=fut-np.nanmedian(fut); ok=np.isfinite(X).all(axis=1)&np.isfinite(y)&(np.abs(y)<.7)
        if ok.any():Xs.append(X[ok]);ys.append(y[ok])
    Xdev=np.vstack(Xs);ydev=np.concatenate(ys); models=fit_models(Xdev,ydev); print('DEV_ROWS',len(ydev),'Y_MEAN',np.mean(ydev),'Y_STD',np.std(ydev),flush=True)
    periods=[('validation_2022_2023',pd.Timestamp('2022-01-01'),VAL_END),('holdout_2024_2026H1',pd.Timestamp('2024-01-01'),END)];summ=[];alltr=[];alleq=[];allff=[]
    for name,a,b in periods:
        eq,tr,ff=run_period(name,a,b,models,dates,codes,member,close,open_,raw_close,raw_open,raw_volume,ret1,mom5,mom20,volratio,amount_rank,vol20,high20,limitup,streak,lhb_map,market);st=stats(eq,tr);st['period']=name;st['dynamic_equal_weight_return']=benchmark_return(a,b,dates,member,close);summ.append(st);eq['period']=name;alltr.append(tr);alleq.append(eq);allff.append(ff);print('RESULT',name,st,flush=True)
    pd.DataFrame(summ).to_csv(OUT/'summary.csv',index=False);pd.concat(alltr,ignore_index=True).to_csv(OUT/'trades.csv',index=False);pd.concat(alleq,ignore_index=True).to_csv(OUT/'equity.csv',index=False);pd.concat(allff,ignore_index=True).to_csv(OUT/'failed_fills.csv',index=False)
    audit=pd.DataFrame([{'version':'V4 frozen regime-gated cross-sectional residual ensemble','development':'2016-07-29..2021-12-31','validation':'2022-01-01..2023-12-31','holdout':'2024-01-01..2026-07-29','future_features':0,'future_labels_outside_development':0,'market_gate_source':'V3 development positive leaves only','stock_model':'50% Ridge(alpha20)+50% HistGradientBoosting fixed','entry':'T+1 auction observed; fill open+30bp','exit':'T+3 open-20bp','daily_selection':'top 2% predicted cross-sectional residual; max2 new, max4 names'}]);audit.to_csv(OUT/'audit.csv',index=False);print('AUDIT');print(audit.to_string(index=False));print('SUMMARY');print(pd.DataFrame(summ).to_string(index=False))
if __name__=='__main__':main()
