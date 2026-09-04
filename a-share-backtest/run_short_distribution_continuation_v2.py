from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import glob, json
import numpy as np
import pandas as pd

import run_short_distribution_continuation_v1 as v1
import run_10y_era_backtest as base
import run_short_t1_inventory_v1 as t1

START=v1.START; SPLIT=v1.SPLIT; END=v1.END; INITIAL=v1.INITIAL; N=v1.N; MEM=v1.MEM; SLIP=v1.SLIP; VP=v1.VP
OUT=Path('results_short_distribution_continuation_v2'); OUT.mkdir(exist_ok=True)

@dataclass
class Pos:
    units: float; entry_px: float; entry_mkt_px: float; entry_fee: float; entry_date: pd.Timestamp; expiry_idx: int; last_px: float; borrow_paid: float=0.0

class FullFallback:
    def __init__(self,cal,members):
        self.cal=pd.DatetimeIndex(cal); self.members=members; self.cache={}; self.calls=0; self.codes=set()
    def _load(self,code):
        if code in self.cache:return self.cache[code]
        cols={}
        for f in ['open','high','low','close','volume','factor']:
            s=base.qb.read_bin(code,f,self.cal)
            if not s.empty:cols[f]=s
        z=pd.concat(cols,axis=1) if cols else pd.DataFrame()
        if len(z):
            if 'factor' not in z:z['factor']=1.0
            z['factor']=z.factor.replace(0,np.nan).ffill().fillna(1.0)
            z['raw_close']=z.close/z.factor
            z['prev_raw_close']=z.raw_close.ffill().shift(1)
        self.cache[code]=z; return z
    def row(self,code,d):
        self.calls+=1; self.codes.add(code); d=pd.Timestamp(d)
        k=self.cal.searchsorted(d,side='right')
        if k>=len(self.cal):return None
        td=pd.Timestamp(self.cal[k]); z=self._load(code)
        if z.empty or td not in z.index:return None
        r=z.loc[td]
        need=[r.get('open',np.nan),r.get('high',np.nan),r.get('low',np.nan),r.get('volume',np.nan),r.get('factor',np.nan)]
        if not np.isfinite(need).all():return None
        f=float(r.factor) if np.isfinite(r.factor) and r.factor>0 else 1.0
        rawopen=float(r.open/f); pc=float(r.prev_raw_close) if np.isfinite(r.prev_raw_close) else np.nan
        lim=t1.board_limit(code,td); gap=rawopen/pc-1 if np.isfinite(pc) and pc>0 else np.nan
        return {'signal_date':d,'trade_date':td,'code':code,'exec_open':float(r.open),'exec_high':float(r.high),'exec_low':float(r.low),'exec_volume':float(r.volume),'exec_factor':f,'exec_open_gap':gap,'exec_limit_proxy':lim,'exec_buy_allowed':bool(np.isfinite(gap) and gap<lim-.002),'exec_sell_allowed':bool(np.isfinite(gap) and gap>-lim+.002),'fallback':1}


def simulate(signals,exec_rows,tc,fb,borrow_rate=.08,cost_mult=1.0,label='exact_mirror'):
    tc=pd.DatetimeIndex(tc[(tc>=START)&(tc<END)])
    tdmap={d:pd.Timestamp(tc[tc.get_loc(d)+1]) if tc.get_loc(d)+1<len(tc) else pd.NaT for d in tc}
    byx={pd.Timestamp(d):g.set_index('code',drop=False) for d,g in exec_rows.groupby('signal_date')}
    fresh={pd.Timestamp(d):g for d,g in signals.groupby('signal_date')}
    capital=float(INITIAL); pos={}; eq=[]; trades=[]; timing=[]
    blocked_open=blocked_cover=missing_after_fallback=0; fallback_rows=0; max_gross_ratio=0.; max_entry_gross_ratio=0.
    def getrow(d,c):
        nonlocal fallback_rows,missing_after_fallback
        g=byx.get(pd.Timestamp(d))
        if g is not None and c in g.index and np.isfinite(g.loc[c].exec_open):return g.loc[c]
        q=fb.row(c,d)
        if q is None:missing_after_fallback+=1; return None
        fallback_rows+=1; return pd.Series(q)
    def enav():return capital+sum((p.entry_px-p.last_px)*p.units for p in pos.values())

    idxmap={pd.Timestamp(d):i for i,d in enumerate(tc)}
    for i,d in enumerate(tc):
        td=tdmap[d]
        if pd.isna(td) or td>END:continue
        fs=fresh.get(pd.Timestamp(d))
        if fs is not None:
            for c in set(fs.code).intersection(pos):pos[c].expiry_idx=max(pos[c].expiry_idx,i+MEM)
        # mark all open names from full data when cache is missing
        for c,p in pos.items():
            r=getrow(d,c)
            if r is not None:p.last_px=float(r.exec_open)
        # persistent cover
        for c in list(pos):
            p=pos[c]
            if i<p.expiry_idx:continue
            r=getrow(d,c)
            if r is None:continue
            if not bool(r.get('exec_buy_allowed',True)):
                blocked_cover+=1;continue
            cover=float(r.exec_open)*(1+SLIP*cost_mult); gross=p.units*cover; cf=v1.fee(gross,'buy',td,cost_mult); pnl=(p.entry_px-cover)*p.units
            capital+=pnl-cf; net=pnl-p.entry_fee-cf-p.borrow_paid; denom=max(1e-12,p.units*p.entry_mkt_px)
            trades.append({'variant':label,'code':c,'entry_date':p.entry_date,'exit_date':td,'net_pnl':net,'net_return':net/denom,'holding_days':(td-p.entry_date).days,'borrow_paid':p.borrow_paid})
            timing.append({'signal_date':d,'trade_date':td,'side':'buy_to_cover','code':c});del pos[c]
        nav=max(1.,enav());gross_now=sum(p.units*p.last_px for p in pos.values());per=nav*.99/N
        if fs is not None and len(pos)<N:
            for rr in fs.itertuples(index=False):
                c=rr.code
                if c in pos or len(pos)>=N:continue
                r=getrow(d,c)
                if r is None:continue
                if not bool(r.get('exec_sell_allowed',True)):
                    blocked_open+=1;continue
                factor=float(r.exec_factor) if np.isfinite(r.exec_factor) and r.exec_factor>0 else 1.;mkt=float(r.exec_open);sell=mkt*(1-SLIP*cost_mult);rawpx=sell/factor
                cap=max(0.,nav*.99-gross_now);target=min(per,cap);shares=int(target//(rawpx*100))*100
                maxraw=max(0,int(abs(float(r.exec_volume))*factor*VP//100)*100)
                if maxraw>0:shares=min(shares,maxraw)
                if shares<=0:continue
                units=shares/factor;notional=units*sell;ef=v1.fee(notional,'sell',td,cost_mult);capital-=ef
                pos[c]=Pos(units,sell,mkt,ef,td,i+MEM,mkt,0.);gross_now+=units*mkt;timing.append({'signal_date':d,'trade_date':td,'side':'sell_short','code':c})
                max_entry_gross_ratio=max(max_entry_gross_ratio,gross_now/max(1.,enav()))
        daily_b=0.
        for p in pos.values():
            b=p.units*p.last_px*borrow_rate/252.;p.borrow_paid+=b;daily_b+=b
        capital-=daily_b;nav=enav();gross=sum(p.units*p.last_px for p in pos.values());ratio=gross/nav if nav>0 else np.inf;max_gross_ratio=max(max_gross_ratio,ratio)
        eq.append({'variant':label,'signal_date':d,'trade_date':td,'equity':nav,'positions':len(pos),'gross_short':gross,'gross_ratio':ratio,'daily_borrow':daily_b})
    E=pd.DataFrame(eq).drop_duplicates('trade_date',keep='last').sort_values('trade_date');T=pd.DataFrame(trades);TM=pd.DataFrame(timing)
    timing_bad=int((pd.to_datetime(TM.signal_date)>=pd.to_datetime(TM.trade_date)).sum()) if len(TM) else 0
    st={**v1.perf(E),**{f'split1_{k}':v for k,v in v1.perf(E,START,'2021-12-31').items()},**{f'split2_{k}':v for k,v in v1.perf(E,SPLIT,END).items()}}
    st.update({'variant':label,'borrow_rate':borrow_rate,'cost_mult':cost_mult,'signals':len(signals),'trades':len(T),'positions_open_end':len(pos),'blocked_open':blocked_open,'blocked_cover':blocked_cover,'fallback_rows':fallback_rows,'fallback_codes':len(fb.codes),'missing_after_fallback':missing_after_fallback,'max_gross_ratio':max_gross_ratio,'max_entry_gross_ratio':max_entry_gross_ratio,'timing_violations':timing_bad})
    if len(T):st.update({'win_rate':float((T.net_return>0).mean()),'mean_trade_return':float(T.net_return.mean()),'median_trade_return':float(T.net_return.median()),'median_holding_days':float(T.holding_days.median()),'p90_holding_days':float(T.holding_days.quantile(.90)),'max_holding_days':float(T.holding_days.max())})
    return st,E,T,TM


def main():
    # Full Qlib calendar is used only to fill execution/marking rows absent from the frozen 18-session cache.
    base.START=START;base.END=END;base.WARM=pd.Timestamp('2014-01-01')
    cal,members,ua=base.load_base();X=v1.load_exec();fb=FullFallback(cal,members)
    S=v1.load_signals('eventonly',False);rows=[]
    st,eq,tr,tm=simulate(S,X,cal,fb,.08,1.,'exact_eventonly_short');rows.append(st);eq.to_csv(OUT/'equity_exact.csv',index=False);tr.to_csv(OUT/'trades_exact.csv',index=False);tm.to_csv(OUT/'timing_exact.csv',index=False);v1.annual(eq).to_csv(OUT/'annual_exact.csv',index=False)
    for br in (0.,.15,.30):rows.append(simulate(S,X,cal,fb,br,1.,f'borrow_{br:.2f}')[0])
    for cm in (2.,4.):rows.append(simulate(S,X,cal,fb,.08,cm,f'cost_{cm:.0f}x')[0])
    SL=v1.load_signals('eventonly',True);rows.append(simulate(SL,X,cal,fb,.08,1.,'liquid_proxy_2x_cutoff')[0])
    SC=v1.load_signals('confirmed',False);rows.append(simulate(SC,X,cal,fb,.08,1.,'t1_confirmed_short_control')[0])
    R=pd.DataFrame(rows);R.to_csv(OUT/'summary.csv',index=False)
    p=R[R.variant=='exact_eventonly_short'].iloc[0];c2=R[R.variant=='cost_2x'].iloc[0]
    gates={'full_cagr_positive':int(p.cagr>0),'split1_cagr_positive':int(p.split1_cagr>0),'split2_cagr_positive':int(p.split2_cagr>0),'full_sharpe_positive':int(p.sharpe>0),'mdd_better_than_minus45':int(p.max_drawdown>-.45),'cost2_cagr_positive':int(c2.cagr>0),'timing_zero':int(p.timing_violations==0),'entry_gross_le_1x':int(p.max_entry_gross_ratio<=1.000001),'fallback_complete':int(p.missing_after_fallback==0)}
    pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_csv(OUT/'gates.csv',index=False)
    (OUT/'metadata.json').write_text(json.dumps({'label':'POST_DIAGNOSTIC_TRUE_SHORT_WITH_FULL_PRICE_FALLBACK','gates_passed':sum(gates.values()),'gates_total':len(gates),'universe_audit':ua},default=str,ensure_ascii=False,indent=2))
    print('=== V2 TRUE SHORT ===');print(R.to_string(index=False),flush=True);print('=== GATES ===');print(pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_string(index=False),flush=True)

if __name__=='__main__':main()
