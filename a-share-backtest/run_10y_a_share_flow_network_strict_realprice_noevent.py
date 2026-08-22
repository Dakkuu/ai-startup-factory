from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import run_10y_china_behavior_daily as base
import run_10y_a_share_flow_network as m
import run_10y_a_share_flow_network_strict as s

warnings.filterwarnings('ignore')
OUT=Path('results_10y_flow_network_strict_realprice_noevent'); OUT.mkdir(exist_ok=True)
s.OUT=OUT; m.OUT=OUT
START=pd.Timestamp('2016-07-29'); END=pd.Timestamp('2026-07-29')
INITIAL=1_000_000.0; MAX_NAMES=5; SLIPPAGE=.0020; PARTICIPATION=.02
VARIANTS=[
 '01_leader_only_strict_realprice',
 '02_network_leader_strict_realprice',
 '03_network_lhb_strict_realprice',
 '04_full_regime_network_lhb_strict_realprice',
 '05_full_regime_network_lhb_strict_realprice_auction_T2',
]

def score_variant(df,v):
    if df.empty:return df
    x=df.copy()
    if v.startswith('01_'):x['score']=.55*x.leader+.25*x.base+.20*x.capacity-.18*x.crowd
    elif v.startswith('02_'):x['score']=.35*x.theme+.35*x.leader+.20*x.capacity+.10*x.base-.18*x.crowd
    elif v.startswith('03_'):x['score']=.29*x.theme+.29*x.leader+.18*x.capacity+.08*x.base+.10*x.lhb-.20*x.crowd
    else:x['score']=.27*x.theme+.27*x.leader+.18*x.capacity+.06*x.base+.10*x.lhb-.22*x.crowd
    return x.sort_values('score',ascending=False)

def raw_arrays(adj_close,adj_open,adj_high,adj_volume,factor):
    ok=np.isfinite(factor)&(factor>0)
    raw_close=np.divide(adj_close,factor,out=np.full_like(adj_close,np.nan,float),where=ok)
    raw_open=np.divide(adj_open,factor,out=np.full_like(adj_open,np.nan,float),where=ok)
    raw_high=np.divide(adj_high,factor,out=np.full_like(adj_high,np.nan,float),where=ok)
    raw_volume=np.multiply(adj_volume,factor,out=np.full_like(adj_volume,np.nan,float),where=ok)
    return raw_close,raw_open,raw_high,raw_volume

def main():
    dates,codes,close,open_,high,volume,factor,member,load_audit=base.load_data()
    dates=pd.DatetimeIndex(dates); codes=list(codes); member=member.astype(bool)
    stock_mask=np.array([bool(base.STOCK_RE.match(c)) for c in codes]); member[:,~stock_mask]=False
    raw_close,raw_open,raw_high,raw_volume=raw_arrays(close,open_,high,volume,factor)
    n,p=close.shape
    print('DATA',n,p,'stock_union',int(member.any(axis=0).sum()),flush=True)
    # Signals use adjusted total-return prices; execution constraints use raw RMB/share data.
    ret1=np.full_like(close,np.nan,float); ret1[1:]=close[1:]/close[:-1]-1
    mom5=np.full_like(close,np.nan,float); mom5[5:]=close[5:]/close[:-5]-1
    vma20=m.rolling_mean_mat(raw_volume,20)
    volratio=np.divide(raw_volume,vma20,out=np.full_like(raw_volume,np.nan,float),where=vma20>0)
    amount=np.maximum(raw_volume,0)*np.maximum(raw_close,0)
    amount_rank=np.full_like(close,np.nan,float)
    for t in range(n):
        ids=np.where(member[t]&np.isfinite(amount[t]))[0]; amount_rank[t,ids]=m.pct_rank(np.log1p(amount[t,ids]))
    # Reconstruct price-limit events from raw unadjusted closes.
    raw_ret=np.full_like(raw_close,np.nan,float); raw_ret[1:]=raw_close[1:]/raw_close[:-1]-1
    limitup=np.zeros((n,p),bool); limitdn=np.zeros((n,p),bool)
    for t in range(1,n):
        for idx in np.where(member[t]&np.isfinite(raw_ret[t]))[0]:
            lim=m.get_limit_pct(codes[idx],dates[t]); limitup[t,idx]=raw_ret[t,idx]>=lim*.985; limitdn[t,idx]=raw_ret[t,idx]<=-lim*.985
    streak=np.zeros((n,p),np.int16)
    for t in range(1,n):streak[t]=np.where(limitup[t],streak[t-1]+1,0)
    market=m.build_regime(dates,member,ret1,limitup,limitdn,close); market.to_csv(OUT/'market_regime.csv',index=False)
    lhb=s.fetch_lhb_parallel(); lhb_map={}
    if len(lhb):
        for r in lhb.itertuples(index=False):
            lhb_map[(pd.Timestamp(r.date).normalize(),r.code)]={'net':float(r.lhb_net) if pd.notna(r.lhb_net) else np.nan,'net_ratio':float(r.lhb_net_ratio) if pd.notna(r.lhb_net_ratio) else np.nan}
        lhb.to_csv(OUT/'lhb_raw_safe_fields.csv',index=False)
    event_map={}
    start_i=int(np.searchsorted(dates.values,START.to_datetime64())); end_i=int(np.searchsorted(dates.values,END.to_datetime64(),side='right')-1)
    cache={}
    for t in range(max(start_i,25),end_i+1):
        # raw_close supplies true low-price/crowding status; returns/momentum remain adjusted.
        cache[t]=m.compute_daily_candidates(t,dates,codes,member,raw_close,raw_volume,ret1,mom5,volratio,amount_rank,limitup,streak,lhb_map,event_map)
        if t%250==0:print('candidate',t,len(cache[t]),flush=True)
    summaries=[]; alltr=[]; alleq=[]; failed=[]; timing=[]; confirms=[]
    for v in VARIANTS:
        print('SIM',v,flush=True); cash=INITIAL; pos={}; real_entry_shares={}; pending=[]; pending_auction=[]; confirmed_to_buy=[]; pending_exits=set(); trades=[]; eq=[]
        for t in range(start_i,end_i+1):
            d=pd.Timestamp(dates[t])
            # Previous-close exits fill at today's open. Economic proceeds are marked on adjusted total-return price path.
            for idx in list(pending_exits):
                if idx not in pos:continue
                rop=raw_open[t,idx]; aop=open_[t,idx]
                if (not np.isfinite(rop)) or (not np.isfinite(aop)) or (not np.isfinite(raw_close[t-1,idx])) or base.open_locked(codes[idx],d,rop,raw_close[t-1,idx],'sell'):
                    failed.append([v,d,codes[idx],'sell','unfilled_open_limit_or_missing']);continue
                P=pos.pop(idx); shares=real_entry_shares.pop(idx,0)
                adj_exec=aop*(1-SLIPPAGE); econ_gross=P.units*adj_exec
                # commission base approximated by economic sale value; corporate-action fee difference is immaterial at this horizon.
                f=m.fee(econ_gross,'sell',d); cash+=econ_gross-f; pnl=(econ_gross-f)-(P.entry_value+P.entry_fee)
                trades.append({'strategy':v,'code':codes[idx],'signal_date':P.signal_date,'entry_date':P.entry_date,'exit_date':d,'entry_adj_px':P.entry_px,'exit_adj_px':adj_exec,'entry_raw_shares':shares,'entry_value':P.entry_value,'net_pnl':pnl,'net_return':pnl/(P.entry_value+P.entry_fee)})
            pending_exits=set()
            buylist=confirmed_to_buy if v.startswith('05_') else pending
            if buylist and len(pos)<MAX_NAMES:
                slots=MAX_NAMES-len(pos); state=market.iloc[t-1].state if t>0 else 'divergence'; exposure=m.EXPOSURE[state] if v.startswith(('04_','05_')) else .80
                marked=sum(P.units*(close[t-1,i] if t>0 and np.isfinite(close[t-1,i]) else P.entry_px) for i,P in pos.items())
                target_total=max(0,cash+marked)*exposure; target_each=target_total/MAX_NAMES; accepted=0
                for rec in buylist:
                    if accepted>=slots:break
                    idx=int(rec['idx'])
                    if idx in pos or not member[t,idx]:continue
                    rop=raw_open[t,idx]; aop=open_[t,idx]
                    if (not np.isfinite(rop)) or (not np.isfinite(aop)) or (not np.isfinite(raw_close[t-1,idx])):
                        failed.append([v,d,codes[idx],'buy','missing_or_not_member']);continue
                    if base.open_locked(codes[idx],d,rop,raw_close[t-1,idx],'buy'):
                        failed.append([v,d,codes[idx],'buy','open_limit_locked']);continue
                    raw_exec=rop*(1+SLIPPAGE); adj_exec=aop*(1+SLIPPAGE)
                    cap=PARTICIPATION*max(raw_volume[t-1,idx],0)*max(raw_close[t-1,idx],0); budget=min(target_each,cap,cash*.98)
                    shares=int(budget/raw_exec//100*100)
                    if shares<100:continue
                    gross=shares*raw_exec; f=m.fee(gross,'buy',d)
                    if gross+f>cash:continue
                    # Synthetic adjusted units preserve split/dividend total-return economics; actual shares only enforce A-share lot sizing.
                    econ_units=gross/adj_exec
                    cash-=gross+f; pos[idx]=m.Position(econ_units,adj_exec,d,pd.Timestamp(rec['signal_date']),adj_exec,gross,f); real_entry_shares[idx]=shares
                    timing.append([v,pd.Timestamp(rec['signal_date']),rec.get('confirm_date',pd.NaT),d,codes[idx],t-int(rec['signal_t'])]);accepted+=1
            pending=[]; confirmed_to_buy=[]
            next_confirmed=[]
            if v.startswith('05_') and pending_auction:
                for rec in pending_auction:
                    idx=int(rec['idx'])
                    if not member[t,idx] or not np.isfinite(raw_open[t,idx]) or not np.isfinite(raw_close[t-1,idx]):continue
                    gap=raw_open[t,idx]/raw_close[t-1,idx]-1; maxgap=.08 if rec.get('streak',0)>=1 else .055
                    if gap<-.035 or gap>maxgap:failed.append([v,d,codes[idx],'confirm','auction_gap_reject']);continue
                    q=dict(rec); q['confirm_date']=d; next_confirmed.append(q); confirms.append([v,pd.Timestamp(rec['signal_date']),d,codes[idx],gap])
                pending_auction=[]
            holdings=0
            for idx,P in pos.items():
                cp=close[t,idx]
                if np.isfinite(cp):P.peak=max(P.peak,cp);holdings+=P.units*cp
                else:holdings+=P.units*P.entry_px
            eq.append({'strategy':v,'date':d,'equity':cash+holdings,'cash':cash,'positions':len(pos)})
            state=market.iloc[t].state
            for idx,P in list(pos.items()):
                cp=close[t,idx]
                if not np.isfinite(cp):continue
                held=max(0,t-int(np.searchsorted(dates.values,P.entry_date.to_datetime64()))); ctab=cache.get(t,pd.DataFrame()); row=ctab[ctab.idx==idx] if not ctab.empty else pd.DataFrame(); theme_now=float(row.theme.iloc[0]) if len(row) else 0
                exit_flag=(held>=5 or cp/P.entry_px-1<=-.08 or cp/max(P.peak,1e-9)-1<=-.07)
                if v.startswith(('04_','05_')) and (state=='retreat' or theme_now<.20) and held>=1:exit_flag=True
                if exit_flag:pending_exits.add(idx)
            ranked=score_variant(cache.get(t,pd.DataFrame()),v)
            if not ranked.empty:
                if not v.startswith('01_'):ranked=ranked[(ranked.peer_count>=3)&(ranked.theme>=.45)]
                if v.startswith(('04_','05_')):ranked=ranked.iloc[0:0] if state=='retreat' else ranked[ranked.score>=.42]
                else:ranked=ranked[ranked.score>=.40]
                picks=[]
                for r in ranked.head(12).to_dict('records'):
                    if int(r['idx']) not in pos:r['signal_date']=d;r['signal_t']=t;picks.append(r)
                if v.startswith('05_'):pending_auction=picks
                else:pending=picks
            if v.startswith('05_'):confirmed_to_buy=next_confirmed
        eqdf=pd.DataFrame(eq); st=m.stats(eqdf,trades); st['strategy']=v; summaries.append(st);alltr.extend(trades);alleq.append(eqdf);print('RESULT',v,st,flush=True)
    summary=pd.DataFrame(summaries).sort_values('total_return',ascending=False)
    ew=[]
    for t in range(start_i+1,end_i+1):
        mm=member[t-1]&member[t]&np.isfinite(ret1[t]);ew.append(np.nanmean(ret1[t,mm]) if mm.any() else 0.0)
    ewret=float(np.prod(1+np.asarray(ew))-1) if ew else np.nan;summary['dynamic_equal_weight_return']=ewret
    summary.to_csv(OUT/'summary.csv',index=False);pd.concat(alleq,ignore_index=True).to_csv(OUT/'equity.csv',index=False);tdf=pd.DataFrame(alltr);tdf.to_csv(OUT/'trades.csv',index=False)
    pd.DataFrame(failed,columns=['strategy','date','code','side','reason']).to_csv(OUT/'failed_fills.csv',index=False)
    tim=pd.DataFrame(timing,columns=['strategy','signal_date','confirm_date','trade_date','code','trade_session_lag']);tim.to_csv(OUT/'timing_audit.csv',index=False)
    pd.DataFrame(confirms,columns=['strategy','signal_date','confirm_date','code','observed_open_gap']).to_csv(OUT/'auction_confirmations.csv',index=False)
    ar=[];edf=pd.concat(alleq,ignore_index=True)
    for name,g in edf.groupby('strategy'):
        g=g.sort_values('date')
        for y,yg in g.groupby(pd.to_datetime(g.date).dt.year):ar.append([name,y,yg.equity.iloc[-1]/yg.equity.iloc[0]-1])
    pd.DataFrame(ar,columns=['strategy','year','return']).to_csv(OUT/'annual_returns.csv',index=False)
    rb=[]
    if len(tdf):
        for name,g in tdf.groupby('strategy'):
            g=g.sort_values('net_pnl',ascending=False);pnl=g.net_pnl.sum();rb.append([name,pnl,g.head(5).net_pnl.sum(),pnl-g.head(5).net_pnl.sum(),g.head(10).net_pnl.sum(),pnl-g.head(10).net_pnl.sum()])
    pd.DataFrame(rb,columns=['strategy','completed_pnl','best5_pnl','pnl_without_best5','best10_pnl','pnl_without_best10']).to_csv(OUT/'robustness.csv',index=False)
    auction_tim=tim[tim.strategy.str.startswith('05_')] if len(tim) else pd.DataFrame()
    audit={'release_tag':base.RELEASE_TAG,'start':str(START.date()),'end':str(END.date()),'stock_union':int(member[start_i:end_i+1].any(axis=0).sum()),'trade_timing_violations':int((pd.to_datetime(tim.trade_date)<=pd.to_datetime(tim.signal_date)).sum()) if len(tim) else 0,'min_trade_session_lag':int(tim.trade_session_lag.min()) if len(tim) else -1,'strict_auction_min_session_lag':int(auction_tim.trade_session_lag.min()) if len(auction_tim) else -1,'same_open_used_for_filter_and_fill':0,'qlib_normalized_price_used_as_yuan':0,'raw_price_formula':'raw_price=normalized_adjusted_price/factor','raw_volume_formula':'raw_volume=normalized_adjusted_volume*factor','signal_price':'adjusted total-return price ratios','lot_sizing':'raw RMB open, 100-share lots','limit_detection':'raw close-to-close return','lhb_future_return_fields_used':0,'lhb_vendor_success_rate_used':0,'news_events_used':0}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    print('=== AUDIT ===');print(pd.DataFrame([audit]).to_string(index=False));print('=== SUMMARY ===');print(summary.to_string(index=False));print('=== ROBUSTNESS ===');print(pd.read_csv(OUT/'robustness.csv').to_string(index=False))

if __name__=='__main__':main()
