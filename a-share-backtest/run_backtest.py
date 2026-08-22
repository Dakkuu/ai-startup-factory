from __future__ import annotations

import os, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

START=os.getenv('BT_START','2025-08-01'); END=os.getenv('BT_END','2026-08-18')
INITIAL_CASH=1_000_000.0; COMMISSION=0.00025; MIN_COMMISSION=5.0
STAMP_DUTY_SELL=0.0005; TRANSFER_FEE=0.00001; SLIPPAGE=0.0005
MAX_NAMES=5; MAX_WEIGHT=0.20
OUT=Path('results'); DATA=Path('data'); OUT.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)

def qdf(rs):
    rows=[]
    while rs.error_code=='0' and rs.next(): rows.append(rs.get_row_data())
    if rs.error_code!='0': raise RuntimeError(rs.error_msg)
    return pd.DataFrame(rows,columns=rs.fields)

def fetch_baostock():
    import baostock as bs
    lg=bs.login()
    if lg.error_code!='0': raise RuntimeError('baostock login failed: '+lg.error_msg)
    try:
        warm_start=(pd.Timestamp(START)-pd.Timedelta(days=240)).strftime('%Y-%m-%d')
        cal=qdf(bs.query_trade_dates(start_date=warm_start,end_date=END))
        days=pd.to_datetime(cal.loc[cal.is_trading_day=='1','calendar_date']).sort_values().tolist()
        if not days: raise RuntimeError('No trading days')
        weekly=[]; seen=set()
        for d in days:
            k=d.to_period('W-MON')
            if k not in seen: weekly.append(d); seen.add(k)
        snapshots={}; union=set()
        for i,d in enumerate(weekly):
            df=qdf(bs.query_zz500_stocks(date=d.strftime('%Y-%m-%d')))
            if not df.empty:
                codes=set(df.code.astype(str)); snapshots[pd.Timestamp(d)]=codes; union|=codes
            if i%12==0: print('membership',i+1,'/',len(weekly),'union',len(union))
            time.sleep(.03)
        if len(union)<300: raise RuntimeError(f'Historical universe too small: {len(union)}')
        fields='date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST'
        frames=[]; failed=[]
        for i,code in enumerate(sorted(union)):
            rs=bs.query_history_k_data_plus(code,fields,start_date=warm_start,end_date=END,frequency='d',adjustflag='2')
            df=qdf(rs)
            if df.empty: failed.append(code)
            else: frames.append(df)
            if (i+1)%50==0: print('price fetch',i+1,'/',len(union))
            time.sleep(.01)
        if not frames: raise RuntimeError('No bars fetched')
        prices=pd.concat(frames,ignore_index=True)
        idx=qdf(bs.query_history_k_data_plus('sh.000905','date,code,open,high,low,close,preclose,volume,amount,pctChg',start_date=warm_start,end_date=END,frequency='d',adjustflag='3'))
        prices.to_parquet(DATA/'prices.parquet',index=False); idx.to_parquet(DATA/'index.parquet',index=False)
        snap=[{'snapshot_date':d,'code':c} for d,codes in snapshots.items() for c in sorted(codes)]
        pd.DataFrame(snap).to_parquet(DATA/'membership.parquet',index=False)
        print(f'Fetched {len(prices):,} bars, {len(union)} unique constituents, failed={len(failed)}')
        return prices,idx,snapshots
    finally: bs.logout()

def clean(prices,idx):
    prices=prices.copy(); prices['date']=pd.to_datetime(prices.date)
    for c in ['open','high','low','close','preclose','volume','amount','turn','pctChg']:
        prices[c]=pd.to_numeric(prices[c],errors='coerce')
    prices['tradestatus']=pd.to_numeric(prices.tradestatus,errors='coerce').fillna(0).astype(int)
    prices['isST']=pd.to_numeric(prices.isST,errors='coerce').fillna(0).astype(int)
    prices=prices.dropna(subset=['date','code','open','close']).sort_values(['code','date'])
    idx=idx.copy(); idx['date']=pd.to_datetime(idx.date)
    for c in ['open','high','low','close','preclose','volume','amount','pctChg']: idx[c]=pd.to_numeric(idx[c],errors='coerce')
    return prices,idx.dropna(subset=['date','close']).sort_values('date')

def add_features(d):
    d=d.sort_values(['code','date']).copy(); g=d.groupby('code',group_keys=False)
    d['ret1']=g.close.pct_change(); d['mom5']=g.close.pct_change(5); d['mom20']=g.close.pct_change(20); d['mom60']=g.close.pct_change(60)
    d['ma20']=g.close.transform(lambda s:s.rolling(20).mean()); d['ma60']=g.close.transform(lambda s:s.rolling(60).mean())
    d['vol20']=g.ret1.transform(lambda s:s.rolling(20).std()); d['vol60']=g.ret1.transform(lambda s:s.rolling(60).std())
    d['vol_ma20']=g.volume.transform(lambda s:s.rolling(20).mean()); d['amount_ma20']=g.amount.transform(lambda s:s.rolling(20).mean())
    d['prev20_high']=g.high.transform(lambda s:s.shift(1).rolling(20).max()); d['vol_ratio']=d.volume/d.vol_ma20; d['amount_ratio']=d.amount/d.amount_ma20
    return d

def membership(snapshots,date):
    ds=[d for d in snapshots if d<=date]
    return snapshots[max(ds)] if ds else set()

def rp(s,asc=True): return s.rank(pct=True,ascending=asc,method='average')

def select(x,strategy,risk_on=True):
    x=x[(x.tradestatus==1)&(x.amount>5e7)].copy().dropna(subset=['mom20','mom60','vol20','ma20','ma60','prev20_high'])
    if x.empty: return []
    if strategy=='trend_breakout':
        x=x[(x.close>x.ma20)&(x.ma20>x.ma60)&(x.mom20>0)]; x['score']=rp(x.mom20)+rp(x.mom60)+rp(x.vol_ratio)
    elif strategy=='relative_momentum':
        x=x[(x.mom60>0)&(x.close>x.ma20)]; x['score']=.45*rp(x.mom60)+.35*rp(x.mom20)+.20*rp(x.amount)
    elif strategy=='mean_reversion':
        x=x[(x.mom60>-.05)&(x.mom5<-.05)]; x['score']=.70*rp(-x.mom5)+.30*rp(x.amount)
    elif strategy=='lowvol_trend':
        if not risk_on:return []
        x=x[(x.mom60>0)&(x.close>x.ma60)&(x.vol20>0)]; x['score']=.55*rp(x.mom60)+.45*rp(-x.vol20)
    elif strategy=='volume_breakout':
        x=x[(x.close>x.prev20_high)&(x.vol_ratio>1.5)&(x.mom20>0)]; x['score']=.45*rp(x.vol_ratio)+.35*rp(x.mom20)+.20*rp(x.amount)
    elif strategy=='multifactor':
        x=x[(x.mom60>-.10)&(x.vol20>0)]; x['score']=.30*rp(x.mom20)+.25*rp(x.mom60)+.20*rp(-x.vol20)+.15*rp(x.amount)+.10*rp(x.vol_ratio)
    if x.empty:return []
    return x.sort_values('score',ascending=False).code.head(MAX_NAMES).tolist()

@dataclass
class Position:
    shares:int; entry_price:float; entry_date:pd.Timestamp; entry_cost:float

def locked(row,side):
    if row is None or row.tradestatus!=1:return True
    if not all(np.isfinite(row[c]) for c in ['open','high','low']):return True
    same=abs(row.high-row.low)<1e-8 and abs(row.open-row.high)<1e-8
    if not same:return False
    pct=row.pctChg
    return np.isfinite(pct) and ((side=='buy' and pct>4.5) or (side=='sell' and pct<-4.5))

def fee(gross,side):
    return max(MIN_COMMISSION,gross*COMMISSION)+gross*TRANSFER_FEE+(gross*STAMP_DUTY_SELL if side=='sell' else 0)

def backtest(f,idx,snapshots,strategy):
    dates=sorted(d for d in f.date.unique() if pd.Timestamp(START)<=d<=pd.Timestamp(END)); by={d:z.set_index('code',drop=False) for d,z in f.groupby('date')}
    ix=idx.set_index('date').sort_index(); ix['ma60']=ix.close.rolling(60).mean()
    cash=INITIAL_CASH; pos={}; target=None; sigdate=None; trades=[]; equity=[]
    for d in dates:
        day=by.get(d)
        if day is None:continue
        if target is not None:
            tgt=set(target)
            for code in list(pos):
                if code in tgt:continue
                row=day.loc[code] if code in day.index else None
                if row is None or locked(row,'sell'):continue
                px=float(row.open)*(1-SLIPPAGE); gross=pos[code].shares*px; cost=fee(gross,'sell'); cash+=gross-cost; p=pos.pop(code); pnl=gross-cost-p.entry_cost
                trades.append({'strategy':strategy,'code':code,'signal_date':sigdate,'entry_date':p.entry_date,'entry_price':p.entry_price,'exit_date':d,'exit_price':px,'shares':p.shares,'net_pnl':pnl,'net_return':pnl/p.entry_cost,'exit_reason':'rank_rebalance'})
            navopen=cash
            for code,p in pos.items():
                row=day.loc[code] if code in day.index else None; px=float(row.open) if row is not None and np.isfinite(row.open) else p.entry_price; navopen+=p.shares*px
            budget=min(navopen*MAX_WEIGHT,navopen/max(1,len(tgt)))
            for code in target:
                if code in pos:continue
                row=day.loc[code] if code in day.index else None
                if row is None or locked(row,'buy'):continue
                px=float(row.open)*(1+SLIPPAGE); shares=int(min(budget,cash*.98)//(px*100))*100
                if shares<=0:continue
                gross=shares*px; cost=fee(gross,'buy'); total=gross+cost
                if total>cash:continue
                cash-=total; pos[code]=Position(shares,px,d,total)
        nav=cash
        for code,p in pos.items():
            row=day.loc[code] if code in day.index else None; px=float(row.close) if row is not None and np.isfinite(row.close) else p.entry_price; nav+=p.shares*px
        equity.append({'date':d,'strategy':strategy,'equity':nav,'cash':cash,'n_positions':len(pos)})
        mem=membership(snapshots,pd.Timestamp(d)); u=day[day.code.isin(mem)].copy() if mem else day.copy(); risk=True
        if d in ix.index and np.isfinite(ix.loc[d,'ma60']):risk=ix.loc[d,'close']>=ix.loc[d,'ma60']
        target=select(u,strategy,risk); sigdate=d
    return pd.DataFrame(equity),pd.DataFrame(trades)

def stats(eq,tr):
    e=eq.set_index('date').equity.astype(float); r=e.pct_change().dropna(); total=e.iloc[-1]/e.iloc[0]-1; years=max((e.index[-1]-e.index[0]).days/365.25,1/252); cagr=(e.iloc[-1]/e.iloc[0])**(1/years)-1
    mdd=(e/e.cummax()-1).min(); sd=r.std(ddof=0); sharpe=np.sqrt(252)*r.mean()/sd if sd>0 else np.nan; dn=r[r<0].std(ddof=0); sortino=np.sqrt(252)*r.mean()/dn if dn and dn>0 else np.nan
    if tr.empty:return dict(total_return=total,cagr=cagr,max_drawdown=mdd,sharpe=sharpe,sortino=sortino,trades=0,win_rate=np.nan,avg_win=np.nan,avg_loss=np.nan,profit_factor=np.nan,max_losing_streak=0)
    rr=tr.net_return.dropna(); wins=rr[rr>0]; losses=rr[rr<=0]; loss_pnl=abs(tr.loc[tr.net_pnl<0,'net_pnl'].sum()); pf=tr.loc[tr.net_pnl>0,'net_pnl'].sum()/loss_pnl if loss_pnl>0 else np.inf
    streak=best=0
    for v in rr:
        if v<=0:streak+=1; best=max(best,streak)
        else:streak=0
    return dict(total_return=total,cagr=cagr,max_drawdown=mdd,sharpe=sharpe,sortino=sortino,trades=len(rr),win_rate=(rr>0).mean(),avg_win=wins.mean() if len(wins) else np.nan,avg_loss=losses.mean() if len(losses) else np.nan,profit_factor=pf,max_losing_streak=best)

def benchmark(idx):
    x=idx[(idx.date>=pd.Timestamp(START))&(idx.date<=pd.Timestamp(END))].sort_values('date'); e=x.close/x.close.iloc[0]*INITIAL_CASH; eq=pd.DataFrame({'date':x.date.values,'equity':e.values}); return stats(eq,pd.DataFrame()),eq

def main():
    print('Backtest',START,'to',END); prices,idx,snapshots=fetch_baostock(); prices,idx=clean(prices,idx); f=add_features(prices)
    names=['trend_breakout','relative_momentum','mean_reversion','lowvol_trend','volume_breakout','multifactor']; rows=[]; eqs=[]; trs=[]; bm,bmeq=benchmark(idx); bmret=bm['total_return']
    for s in names:
        print('Running',s); eq,tr=backtest(f,idx,snapshots,s); m=stats(eq,tr); m.update(strategy=s,excess_vs_csi500=m['total_return']-bmret); rows.append(m); eqs.append(eq); trs.append(tr) if not tr.empty else None
    b=bm.copy(); b.update(strategy='CSI500_buy_hold',excess_vs_csi500=0.0); rows.append(b)
    summary=pd.DataFrame(rows)[['strategy','total_return','excess_vs_csi500','cagr','max_drawdown','sharpe','sortino','trades','win_rate','avg_win','avg_loss','profit_factor','max_losing_streak']]
    summary.to_csv(OUT/'summary.csv',index=False); pd.concat(eqs).to_csv(OUT/'equity.csv',index=False); (pd.concat(trs) if trs else pd.DataFrame()).to_csv(OUT/'trades.csv',index=False); bmeq.to_csv(OUT/'benchmark_equity.csv',index=False)
    pd.DataFrame([{'start':START,'end':END,'initial_cash':INITIAL_CASH,'commission':COMMISSION,'min_commission':MIN_COMMISSION,'stamp_duty_sell':STAMP_DUTY_SELL,'transfer_fee':TRANSFER_FEE,'slippage_each_side':SLIPPAGE,'membership_snapshots':len(snapshots),'unique_codes':prices.code.nunique(),'price_rows':len(prices),'note':'CSI500 point-in-time weekly membership from BaoStock; qfq bars for continuity; signals at T close and execution at T+1 open.'}]).to_csv(OUT/'metadata.csv',index=False)
    print('\n=== RESULTS ===\n',summary.sort_values('total_return',ascending=False).to_string(index=False))
if __name__=='__main__':main()
