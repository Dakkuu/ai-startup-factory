from __future__ import annotations
import warnings, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
import run_10y_china_behavior_daily as base
import run_10y_a_share_flow_network as v1
import run_10y_a_share_flow_network_strict as strict
import run_10y_a_share_flow_network_v2_earlydiffusion as v2

warnings.filterwarnings('ignore')
OUT=Path('results_10y_v3_behavior_tree'); OUT.mkdir(exist_ok=True)
strict.OUT=OUT; v1.OUT=OUT
START=pd.Timestamp('2016-07-29'); END=pd.Timestamp('2026-07-29')
DEV_END=pd.Timestamp('2021-12-31'); VAL_END=pd.Timestamp('2023-12-31')
INITIAL=1_000_000.0
ENTRY_SLIP=.0030; EXIT_SLIP=.0020; PARTICIPATION=.02; MAX_NAMES=4; MAX_NEW=2; HOLD=2
# Frozen before run. Coarse tree, large leaves, no hyperparameter search.
TREE_DEPTH=4; MIN_LEAF=12000; SAMPLE_PER_DAY=400; TARGET_COST_PROXY=.006
FEATURES=['ret1','mom5','log_volratio','amount_rank','log_price','limitup','streak','lhb_flag','lhb_net_ratio','mkt_limitup_share','mkt_limitdn_share','mkt_median_ret','mkt_breadth20','mkt_board_premium','mkt_reseal_rate','auction_gap']


def make_features(t, ids, dates, codes, member, raw_close, raw_open, ret1, mom5, volratio, amount_rank, limitup, streak, lhb_map, market):
    # Features are known by T+1 open: T-close state + T LHB (published after close) + T+1 auction gap.
    nxt=t+1
    gap=raw_open[nxt,ids]/raw_close[t,ids]-1
    m=market.iloc[t]
    rows=np.zeros((len(ids),len(FEATURES)),float)
    rows[:,0]=ret1[t,ids]
    rows[:,1]=mom5[t,ids]
    rows[:,2]=np.log1p(np.maximum(volratio[t,ids],0))
    rows[:,3]=amount_rank[t,ids]
    rows[:,4]=np.log(np.maximum(raw_close[t,ids],0.5))
    rows[:,5]=limitup[t,ids].astype(float)
    rows[:,6]=np.minimum(streak[t,ids],4)
    for k,idx in enumerate(ids):
        l=lhb_map.get((pd.Timestamp(dates[t]).normalize(),codes[idx]))
        if l is not None:
            rows[k,7]=1.0
            nr=l.get('net_ratio',np.nan)
            rows[k,8]=0.0 if not np.isfinite(nr) else float(np.clip(nr,-30,30))/30.0
    def mv(name, default=0.0):
        x=getattr(m,name)
        return default if pd.isna(x) else float(x)
    rows[:,9]=mv('limitup_share'); rows[:,10]=mv('limitdn_share'); rows[:,11]=mv('median_ret')
    rows[:,12]=mv('breadth20',.5); rows[:,13]=mv('board_premium'); rows[:,14]=mv('reseal_rate')
    rows[:,15]=gap
    return rows


def leaf_table(model, X, y):
    leaf=model.apply(X); out=[]
    for lf in np.unique(leaf):
        z=y[leaf==lf]; out.append((int(lf),len(z),float(np.mean(z)),float(np.median(z)),float(np.mean(z>0))))
    return pd.DataFrame(out,columns=['leaf','n','mean_target_netproxy','median_target_netproxy','win_rate']).sort_values('mean_target_netproxy',ascending=False)


def stats(eq,trades):
    s=pd.Series(eq.equity.values,index=pd.to_datetime(eq.date)); r=s.pct_change().fillna(0)
    total=s.iloc[-1]/s.iloc[0]-1; yrs=max((s.index[-1]-s.index[0]).days/365.25,1/365.25); cagr=(1+total)**(1/yrs)-1 if total>-1 else np.nan
    dd=(s/s.cummax()-1).min(); sd=r.std(); sharpe=(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan
    if len(trades):
        win=(trades.net_return>0).mean(); neg=-trades.loc[trades.net_pnl<0,'net_pnl'].sum(); pf=trades.loc[trades.net_pnl>0,'net_pnl'].sum()/neg if neg>0 else np.nan
    else: win=pf=np.nan
    return dict(final_asset=float(s.iloc[-1]),total_return=float(total),cagr=float(cagr),max_drawdown=float(dd),sharpe=float(sharpe),trades=int(len(trades)),win_rate=float(win) if pd.notna(win) else np.nan,profit_factor=float(pf) if pd.notna(pf) else np.nan)


def run_period(name,a,b,model,good_leaves,dates,codes,member,close,open_,raw_close,raw_open,raw_volume,ret1,mom5,volratio,amount_rank,limitup,streak,lhb_map,market):
    i0=max(int(np.searchsorted(dates.values,a.to_datetime64())),65); i1=min(int(np.searchsorted(dates.values,b.to_datetime64(),side='right')-1),len(dates)-HOLD-2)
    cash=INITIAL; pos={}; eq=[]; trades=[]; failed=[]
    # pos: idx -> dict with econ_units, raw_cost, adj_entry, entry_date, exit_idx, signal_date, entry_fee
    for t in range(i0,i1+HOLD+2):
        d=pd.Timestamp(dates[t])
        # scheduled exits at today's open
        for idx,P in list(pos.items()):
            if t<P['exit_idx']: continue
            ro=raw_open[t,idx]; ao=open_[t,idx]; prev=raw_close[t-1,idx]
            if not np.isfinite(ro) or not np.isfinite(ao) or not np.isfinite(prev) or base.open_locked(codes[idx],d,ro,prev,'sell'):
                failed.append((name,d,codes[idx],'sell','locked_or_missing')); P['exit_idx']=t+1; continue
            raw_exec=ro*(1-EXIT_SLIP); adj_exec=ao*(1-EXIT_SLIP); gross=P['econ_units']*adj_exec; f=v1.fee(gross,'sell',d); cash+=gross-f
            pnl=(gross-f)-(P['entry_value']+P['entry_fee'])
            trades.append((name,codes[idx],P['signal_date'],P['entry_date'],d,P['entry_value'],pnl,pnl/(P['entry_value']+P['entry_fee'])))
            del pos[idx]
        if t>i1: 
            mtm=cash+sum(P['econ_units']*(close[t,idx] if np.isfinite(close[t,idx]) else P['adj_entry']) for idx,P in pos.items()); eq.append((d,mtm)); continue
        # signal from previous close + today's auction, then trade immediately after open with 30bp execution penalty
        sig=t-1
        if sig<i0-1 or len(pos)>=MAX_NAMES: 
            mtm=cash+sum(P['econ_units']*(close[t,idx] if np.isfinite(close[t,idx]) else P['adj_entry']) for idx,P in pos.items()); eq.append((d,mtm)); continue
        valid=member[sig]&member[t]&np.isfinite(raw_close[sig])&np.isfinite(raw_open[t])&np.isfinite(open_[t])&np.isfinite(ret1[sig])&np.isfinite(mom5[sig])&np.isfinite(volratio[sig])&np.isfinite(amount_rank[sig])
        valid &= (raw_close[sig]>=3)&(amount_rank[sig]>=.20)&(volratio[sig]>=.5)
        ids=np.where(valid)[0]
        if len(ids):
            X=make_features(sig,ids,dates,codes,member,raw_close,raw_open,ret1,mom5,volratio,amount_rank,limitup,streak,lhb_map,market)
            finite=np.isfinite(X).all(axis=1); ids=ids[finite]; X=X[finite]
            if len(ids):
                leaves=model.apply(X); pred=model.predict(X); ok=np.array([lf in good_leaves for lf in leaves])
                # Do not chase locked opens or extreme gaps. Gap itself is known at auction and included in tree.
                ok &= (X[:,15]>-.08)&(X[:,15]<.095)
                cand=np.where(ok)[0]
                if len(cand):
                    order=cand[np.argsort(pred[cand])[::-1]]
                    slots=min(MAX_NEW,MAX_NAMES-len(pos)); added=0
                    equity_ref=cash+sum(P['econ_units']*(close[t-1,idx] if np.isfinite(close[t-1,idx]) else P['adj_entry']) for idx,P in pos.items())
                    target=equity_ref*.24
                    for q in order:
                        if added>=slots: break
                        idx=int(ids[q])
                        if idx in pos: continue
                        ro=raw_open[t,idx]; ao=open_[t,idx]; prev=raw_close[t-1,idx]
                        if base.open_locked(codes[idx],d,ro,prev,'buy'):
                            failed.append((name,d,codes[idx],'buy','open_locked')); continue
                        raw_exec=ro*(1+ENTRY_SLIP); adj_exec=ao*(1+ENTRY_SLIP)
                        cap=PARTICIPATION*max(raw_volume[t-1,idx],0)*max(raw_close[t-1,idx],0); budget=min(target,cap,cash*.98)
                        shares=int(budget/raw_exec//100*100)
                        if shares<100: continue
                        raw_gross=shares*raw_exec; f=v1.fee(raw_gross,'buy',d)
                        if raw_gross+f>cash: continue
                        # economic adjusted units let corporate actions flow through adjusted price path
                        econ_units=raw_gross/adj_exec
                        cash-=raw_gross+f
                        pos[idx]=dict(econ_units=econ_units,raw_cost=raw_gross,adj_entry=adj_exec,entry_date=d,exit_idx=t+HOLD,signal_date=pd.Timestamp(dates[sig]),entry_value=raw_gross,entry_fee=f)
                        added+=1
        mtm=cash+sum(P['econ_units']*(close[t,idx] if np.isfinite(close[t,idx]) else P['adj_entry']) for idx,P in pos.items()); eq.append((d,mtm))
    eq=pd.DataFrame(eq,columns=['date','equity']); tr=pd.DataFrame(trades,columns=['period','code','signal_date','entry_date','exit_date','entry_value','net_pnl','net_return']); ff=pd.DataFrame(failed,columns=['period','date','code','side','reason'])
    return eq,tr,ff


def main():
    dates,codes,close,open_,high,volume,factor,member,_=base.load_data(); dates=pd.DatetimeIndex(dates); codes=list(codes); member=member.astype(bool)
    stock_mask=np.array([bool(base.STOCK_RE.match(c)) for c in codes]); member[:,~stock_mask]=False
    raw_close,raw_open,raw_high,raw_volume=v2.raw_arrays(close,open_,high,volume,factor)
    n,p=close.shape; ret1=np.full_like(close,np.nan,float); ret1[1:]=close[1:]/close[:-1]-1; mom5=np.full_like(close,np.nan,float); mom5[5:]=close[5:]/close[:-5]-1
    vma20=v1.rolling_mean_mat(raw_volume,20); volratio=np.divide(raw_volume,vma20,out=np.full_like(raw_volume,np.nan,float),where=vma20>0)
    amount=np.maximum(raw_volume,0)*np.maximum(raw_close,0); amount_rank=np.full_like(close,np.nan,float)
    for t in range(n):
        ids=np.where(member[t]&np.isfinite(amount[t]))[0]; amount_rank[t,ids]=v1.pct_rank(np.log1p(amount[t,ids]))
    raw_ret=np.full_like(raw_close,np.nan,float); raw_ret[1:]=raw_close[1:]/raw_close[:-1]-1
    limitup=np.zeros((n,p),bool); limitdn=np.zeros((n,p),bool); streak=np.zeros((n,p),np.int8)
    for t in range(1,n):
        ids=np.where(member[t]&np.isfinite(raw_ret[t]))[0]
        for idx in ids:
            lim=v1.get_limit_pct(codes[idx],dates[t]); limitup[t,idx]=raw_ret[t,idx]>=lim*.985; limitdn[t,idx]=raw_ret[t,idx]<=-lim*.985
            streak[t,idx]=(min(4,int(streak[t-1,idx])+1) if limitup[t,idx] else 0)
    market=v1.build_regime(dates,member,ret1,limitup,limitdn,close); market.to_csv(OUT/'market_regime.csv',index=False)
    lhb=strict.fetch_lhb_parallel(); lhb_map={}
    if len(lhb):
        lhb.to_csv(OUT/'lhb_raw_safe_fields.csv',index=False)
        for r in lhb.itertuples(index=False): lhb_map[(pd.Timestamp(r.date).normalize(),r.code)]={'net':float(r.lhb_net) if pd.notna(r.lhb_net) else np.nan,'net_ratio':float(r.lhb_net_ratio) if pd.notna(r.lhb_net_ratio) else np.nan}
    # deterministic development sample only
    rng=np.random.default_rng(20260819); Xs=[]; ys=[]
    s0=max(int(np.searchsorted(dates.values,START.to_datetime64())),65); s1=min(int(np.searchsorted(dates.values,DEV_END.to_datetime64(),side='right')-1),n-HOLD-2)
    for t in range(s0,s1+1):
        valid=member[t]&member[t+1]&np.isfinite(raw_close[t])&np.isfinite(raw_open[t+1])&np.isfinite(open_[t+1])&np.isfinite(open_[t+1+HOLD])&np.isfinite(ret1[t])&np.isfinite(mom5[t])&np.isfinite(volratio[t])&np.isfinite(amount_rank[t])
        valid &= (raw_close[t]>=3)&(amount_rank[t]>=.20)&(volratio[t]>=.5)
        ids=np.where(valid)[0]
        if len(ids)>SAMPLE_PER_DAY: ids=rng.choice(ids,SAMPLE_PER_DAY,replace=False)
        if not len(ids): continue
        X=make_features(t,ids,dates,codes,member,raw_close,raw_open,ret1,mom5,volratio,amount_rank,limitup,streak,lhb_map,market)
        y=open_[t+1+HOLD,ids]/open_[t+1,ids]-1-TARGET_COST_PROXY
        ok=np.isfinite(X).all(axis=1)&np.isfinite(y)&(np.abs(y)<.7)
        if ok.any(): Xs.append(X[ok]); ys.append(y[ok])
        if t%250==0: print('DEV_SAMPLE',t,sum(len(x) for x in ys),flush=True)
    Xdev=np.vstack(Xs); ydev=np.concatenate(ys)
    model=DecisionTreeRegressor(max_depth=TREE_DEPTH,min_samples_leaf=MIN_LEAF,random_state=20260819)
    model.fit(Xdev,ydev)
    lt=leaf_table(model,Xdev,ydev); lt.to_csv(OUT/'dev_leaf_table.csv',index=False)
    (OUT/'tree.txt').write_text(export_text(model,feature_names=FEATURES),encoding='utf-8')
    # Freeze positive leaves. Require >20bp net-proxy and >15k samples; do not force a trade if none.
    good=set(lt.loc[(lt.mean_target_netproxy>.002)&(lt.n>=15000),'leaf'].astype(int))
    pd.DataFrame({'selected_leaf':sorted(good)}).to_csv(OUT/'selected_leaves.csv',index=False)
    print('DEV_ROWS',len(ydev),'GOOD_LEAVES',sorted(good),flush=True); print(lt.head(20).to_string(index=False),flush=True); print(export_text(model,feature_names=FEATURES),flush=True)
    periods=[('validation_2022_2023',pd.Timestamp('2022-01-01'),pd.Timestamp('2023-12-31')),('holdout_2024_2026H1',pd.Timestamp('2024-01-01'),END)]
    summaries=[]; alltr=[]; alleq=[]; allff=[]
    for name,a,b in periods:
        eq,tr,ff=run_period(name,a,b,model,good,dates,codes,member,close,open_,raw_close,raw_open,raw_volume,ret1,mom5,volratio,amount_rank,limitup,streak,lhb_map,market)
        st=stats(eq,tr); st['period']=name; summaries.append(st); eq['period']=name; alltr.append(tr); alleq.append(eq); allff.append(ff); print('RESULT',name,st,flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False); pd.concat(alltr,ignore_index=True).to_csv(OUT/'trades.csv',index=False); pd.concat(alleq,ignore_index=True).to_csv(OUT/'equity.csv',index=False); pd.concat(allff,ignore_index=True).to_csv(OUT/'failed_fills.csv',index=False)
    audit=pd.DataFrame([{'version':'V3 frozen interpretable behavior tree','train_period':'2016-07-29..2021-12-31','validation_period':'2022-01-01..2023-12-31','holdout_period':'2024-01-01..2026-07-29','stock_union':int(member[:,stock_mask].any(axis=0).sum()),'future_fields_in_features':0,'target_used_outside_development':0,'same_open_exact_fill':0,'auction_entry_penalty_bps':int(ENTRY_SLIP*10000),'hold_sessions':HOLD,'tree_depth':TREE_DEPTH,'min_leaf':MIN_LEAF,'selected_leaves':','.join(map(str,sorted(good))),'note':'Tree and leaf selection use development only. T+1 auction gap is observed at open; fill assumes +30bp after open. Validation/holdout never used to fit or select leaves.'}]); audit.to_csv(OUT/'audit.csv',index=False); print('AUDIT'); print(audit.to_string(index=False)); print('SUMMARY'); print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__': main()
