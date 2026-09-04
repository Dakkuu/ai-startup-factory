from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import glob, json, math
import numpy as np
import pandas as pd

START=pd.Timestamp('2016-08-02')
SPLIT=pd.Timestamp('2022-01-01')
END=pd.Timestamp('2026-07-29')
INITIAL=1_000_000.0
N=15
MEM=8
SLIP=.001
VP=.05
OUT=Path('results_short_distribution_continuation_v1'); OUT.mkdir(exist_ok=True)

COMMISSION=.00025; MIN_COMMISSION=5.0

def fee(gross,side,d,mult=1.0):
    d=pd.Timestamp(d)
    stamp=.001 if d<pd.Timestamp('2023-08-28') else .0005
    transfer=.00002 if d<pd.Timestamp('2022-04-29') else .00001
    return mult*(max(MIN_COMMISSION,gross*COMMISSION)+gross*transfer+(gross*stamp if side=='sell' else 0.0))

@dataclass
class Pos:
    units: float
    entry_px: float
    entry_mkt_px: float
    entry_fee: float
    entry_date: pd.Timestamp
    expiry_idx: int
    last_px: float
    borrow_paid: float=0.0


def perf(eq,a=None,b=None):
    x=eq.copy(); x['trade_date']=pd.to_datetime(x.trade_date)
    if a is not None:x=x[x.trade_date>=pd.Timestamp(a)]
    if b is not None:x=x[x.trade_date<=pd.Timestamp(b)]
    if len(x)<20:return dict(cagr=np.nan,max_drawdown=np.nan,sharpe=np.nan,total_return=np.nan)
    s=x.set_index('trade_date').equity.astype(float).sort_index(); s=s/s.iloc[0]
    days=max(1,(s.index[-1]-s.index[0]).days); c=float(s.iloc[-1]**(365.25/days)-1)
    dd=float((s/s.cummax()-1).min()); r=s.pct_change().dropna(); sd=float(r.std(ddof=1))
    sh=float(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan
    return dict(cagr=c,max_drawdown=dd,sharpe=sh,total_return=float(s.iloc[-1]-1))


def load_exec():
    files=sorted(glob.glob('artifact_cache/exec/**/execution_rows_*.pkl.gz',recursive=True))
    if len(files)!=12: raise RuntimeError(f'expected 12 exec shards, got {len(files)}')
    x=pd.concat([pd.read_pickle(f,compression='gzip') for f in files],ignore_index=True)
    x['signal_date']=pd.to_datetime(x.signal_date); x['trade_date']=pd.to_datetime(x.trade_date)
    x=x.sort_values(['signal_date','code']).drop_duplicates(['signal_date','code'],keep='last')
    return x


def load_signals(kind='eventonly', liquid_proxy=False):
    fn='eventonly_strict_absorption.csv.gz' if kind=='eventonly' else 'signals_strict_absorption.csv.gz'
    p=glob.glob(f'artifact_cache/prepared/**/{fn}',recursive=True)[0]
    s=pd.read_csv(p,compression='gzip'); s['signal_date']=pd.to_datetime(s.signal_date)
    s=s[(s.signal_date>=START)&(s.signal_date<=END)].copy()
    if liquid_proxy:s=s[s.liq20>=2.0*s.liq_threshold].copy()
    return s.sort_values(['signal_date','score_rank','liq20','code'],ascending=[True,True,False,True])


def simulate(signals,exec_rows,borrow_rate=.08,cost_mult=1.0,label='exact_mirror'):
    tc=pd.DatetimeIndex(sorted(exec_rows.signal_date.unique()))
    posidx={pd.Timestamp(d):i for i,d in enumerate(tc)}
    tdmap=exec_rows.groupby('signal_date').trade_date.first().to_dict()
    byx={pd.Timestamp(d):g.set_index('code',drop=False) for d,g in exec_rows.groupby('signal_date')}
    fresh={pd.Timestamp(d):g for d,g in signals.groupby('signal_date')}
    capital=float(INITIAL); pos={}; eq=[]; trades=[]; timing=[]
    blocked_open=blocked_cover=missing_rows=0; max_gross_ratio=0.0; timing_bad=0

    def equity_now():
        u=sum((p.entry_px-p.last_px)*p.units for p in pos.values())
        return capital+u

    for i,d in enumerate(tc):
        if d<START or d>END: continue
        g=byx.get(pd.Timestamp(d)); td=pd.Timestamp(tdmap[d])
        fs=fresh.get(pd.Timestamp(d))

        # refresh memory for held names receiving a new event
        if fs is not None:
            for c in set(fs.code).intersection(pos): pos[c].expiry_idx=max(pos[c].expiry_idx,i+MEM)

        # mark at execution open when a row exists
        if g is not None:
            for c,p in pos.items():
                if c in g.index and np.isfinite(g.loc[c].exec_open): p.last_px=float(g.loc[c].exec_open)
                else: missing_rows+=1

        # cover expired positions; blocked covers persist
        for c in list(pos):
            p=pos[c]
            if i<p.expiry_idx: continue
            if g is None or c not in g.index or not np.isfinite(g.loc[c].exec_open):
                missing_rows+=1; continue
            r=g.loc[c]
            allowed=bool(r.get('exec_buy_allowed',True))
            if not allowed:
                blocked_cover+=1; continue
            mkt=float(r.exec_open); cover=mkt*(1+SLIP*cost_mult)
            gross=p.units*cover; cf=fee(gross,'buy',td,cost_mult)
            pnl=(p.entry_px-cover)*p.units
            capital+=pnl-cf
            net=pnl-p.entry_fee-cf-p.borrow_paid
            denom=max(1e-12,p.units*p.entry_mkt_px)
            trades.append({'variant':label,'code':c,'entry_date':p.entry_date,'exit_date':td,'net_pnl':net,'net_return':net/denom,'holding_days':(td-p.entry_date).days,'borrow_paid':p.borrow_paid,'exit_reason':'memory_expiry_cover'})
            timing.append({'signal_date':d,'trade_date':td,'side':'buy_to_cover','code':c})
            del pos[c]

        # open new shorts only from today's fresh events
        nav=max(1.0,equity_now())
        gross_now=sum(p.units*p.last_px for p in pos.values())
        per=nav*.99/N
        if fs is not None and len(pos)<N:
            for rr in fs.itertuples(index=False):
                c=rr.code
                if c in pos or len(pos)>=N: continue
                if g is None or c not in g.index or not np.isfinite(g.loc[c].exec_open):
                    missing_rows+=1; continue
                r=g.loc[c]; allowed=bool(r.get('exec_sell_allowed',True))
                if not allowed:
                    blocked_open+=1; continue
                factor=float(r.exec_factor) if np.isfinite(r.exec_factor) and r.exec_factor>0 else 1.0
                mkt=float(r.exec_open); sell=mkt*(1-SLIP*cost_mult); rawpx=sell/factor
                if rawpx<=0: continue
                # never exceed 0.99x gross short exposure
                cap=max(0.0,nav*.99-gross_now)
                target=min(per,cap)
                shares=int(target//(rawpx*100))*100
                maxraw=max(0,int(abs(float(r.exec_volume))*factor*VP//100)*100)
                if maxraw>0: shares=min(shares,maxraw)
                if shares<=0: continue
                units=shares/factor; notional=units*sell; ef=fee(notional,'sell',td,cost_mult)
                capital-=ef
                pos[c]=Pos(units,sell,mkt,ef,td,i+MEM,mkt,0.0)
                gross_now+=units*mkt
                timing.append({'signal_date':d,'trade_date':td,'side':'sell_short','code':c})

        # daily borrow charge on current market value
        daily_borrow=0.0
        if borrow_rate>0 and pos:
            for p in pos.values():
                b=p.units*p.last_px*borrow_rate/252.0
                p.borrow_paid+=b; daily_borrow+=b
            capital-=daily_borrow

        nav=equity_now(); gross=sum(p.units*p.last_px for p in pos.values()); ratio=gross/nav if nav>0 else np.inf
        max_gross_ratio=max(max_gross_ratio,ratio)
        eq.append({'variant':label,'signal_date':d,'trade_date':td,'equity':nav,'positions':len(pos),'gross_short':gross,'gross_ratio':ratio,'daily_borrow':daily_borrow})

    E=pd.DataFrame(eq).drop_duplicates('trade_date',keep='last').sort_values('trade_date')
    T=pd.DataFrame(trades); TM=pd.DataFrame(timing)
    if len(TM): timing_bad=int((pd.to_datetime(TM.signal_date)>=pd.to_datetime(TM.trade_date)).sum())
    st={**perf(E),**{f'split1_{k}':v for k,v in perf(E,START,'2021-12-31').items()},**{f'split2_{k}':v for k,v in perf(E,SPLIT,END).items()}}
    st.update({'variant':label,'borrow_rate':borrow_rate,'cost_mult':cost_mult,'signals':len(signals),'trades':len(T),'positions_open_end':len(pos),'blocked_open':blocked_open,'blocked_cover':blocked_cover,'missing_exec_rows':missing_rows,'max_gross_ratio':max_gross_ratio,'timing_violations':timing_bad})
    if len(T):
        st.update({'win_rate':float((T.net_return>0).mean()),'mean_trade_return':float(T.net_return.mean()),'median_trade_return':float(T.net_return.median()),'median_holding_days':float(T.holding_days.median()),'p90_holding_days':float(T.holding_days.quantile(.90)),'max_holding_days':float(T.holding_days.max())})
    return st,E,T,TM


def annual(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); rows=[]
    for y,g in s.groupby(s.index.year):
        before=s[s.index<pd.Timestamp(f'{y}-01-01')]; st=float(before.iloc[-1]) if len(before) else float(g.iloc[0])
        rows.append({'year':int(y),'return':float(g.iloc[-1]/st-1)})
    return pd.DataFrame(rows)


def main():
    X=load_exec(); S=load_signals('eventonly',False)
    rows=[]
    # exact mirror primary
    st,eq,tr,tm=simulate(S,X,.08,1.0,'exact_eventonly_short'); rows.append(st)
    eq.to_csv(OUT/'equity_exact.csv',index=False); tr.to_csv(OUT/'trades_exact.csv',index=False); tm.to_csv(OUT/'timing_exact.csv',index=False); annual(eq).to_csv(OUT/'annual_exact.csv',index=False)
    # borrow stress, no selection
    for br in (0.0,.15,.30):
        x,_,_,_=simulate(S,X,br,1.0,f'borrow_{br:.2f}'); rows.append(x)
    # cost stress, borrow fixed 8%
    for cm in (2.0,4.0):
        x,_,_,_=simulate(S,X,.08,cm,f'cost_{cm:.0f}x'); rows.append(x)
    # liquid feasibility proxy
    SL=load_signals('eventonly',True); x,_,_,_=simulate(SL,X,.08,1.0,'liquid_proxy_2x_cutoff'); rows.append(x)
    # secondary T+1-confirmed short control
    SC=load_signals('confirmed',False); x,_,_,_=simulate(SC,X,.08,1.0,'t1_confirmed_short_control'); rows.append(x)
    R=pd.DataFrame(rows); R.to_csv(OUT/'summary.csv',index=False)
    p=R[R.variant=='exact_eventonly_short'].iloc[0]; c2=R[R.variant=='cost_2x'].iloc[0]
    gates={
      'full_cagr_positive':int(p.cagr>0),
      'split1_cagr_positive':int(p.split1_cagr>0),
      'split2_cagr_positive':int(p.split2_cagr>0),
      'full_sharpe_positive':int(p.sharpe>0),
      'mdd_better_than_minus45':int(p.max_drawdown>-.45),
      'cost2_cagr_positive':int(c2.cagr>0),
      'timing_zero':int(p.timing_violations==0),
      'gross_le_1x':int(p.max_gross_ratio<=1.000001),
    }
    pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_csv(OUT/'gates.csv',index=False)
    spec={'label':'POST_DIAGNOSTIC_TRUE_SHORT_RESEARCH_NOT_CLEAN_OOS','primary':'exact mirror of prior event-only strict_absorption / N15 / memory8','borrow_availability':'assumed; historical per-stock lend availability unavailable','baseline_borrow_rate':.08,'gates_passed':sum(gates.values()),'gates_total':len(gates)}
    (OUT/'metadata.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False))
    print('=== TRUE SHORT SUMMARY ==='); print(R.to_string(index=False),flush=True)
    print('=== GATES ==='); print(pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_string(index=False),flush=True)

if __name__=='__main__': main()
