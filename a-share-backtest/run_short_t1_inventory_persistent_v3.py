from __future__ import annotations

import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_skewfilter_hard as hard
import run_10y_hard_executor_v2 as hv2
import run_short_t1_inventory_v1 as t1

OUT=Path('results_short_t1_inventory_persistent_v3')
GRID=OUT/'grid'; FINAL=OUT/'final'
GRID.mkdir(parents=True,exist_ok=True); FINAL.mkdir(parents=True,exist_ok=True)


def one(pattern):
    h=glob.glob(pattern,recursive=True)
    if not h: raise FileNotFoundError(pattern)
    return h[0]


def setup():
    base.START=t1.START; base.WARM=t1.WARM; base.END=t1.END
    sim.START=t1.START; sim.WARM=t1.WARM; sim.END=t1.END
    base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    bm=market_close.loc[t1.START:t1.END].dropna(); tc,_=t1.next_map(cal)
    return cal,members,ua,market_code,bm,tc


def load_exec():
    fs=glob.glob('exec_input/**/execution_rows_*.pkl.gz',recursive=True)
    if not fs: raise FileNotFoundError('execution shards')
    x=pd.concat([pd.read_pickle(f,compression='gzip') for f in sorted(fs)],ignore_index=True)
    x['signal_date']=pd.to_datetime(x.signal_date); x['trade_date']=pd.to_datetime(x.trade_date)
    return x.drop_duplicates(['signal_date','code'],keep='last')


def load_signal(name,event=False):
    stem='eventonly' if event else 'signals'
    x=pd.read_csv(one(f'prepare_input/**/{stem}_{name}.csv.gz'),compression='gzip')
    x['signal_date']=pd.to_datetime(x.signal_date); x['trade_date']=pd.to_datetime(x.trade_date)
    return x


def build_quote_cache(cal):
    cache={}
    def get(code,td):
        td=pd.Timestamp(td)
        if code not in cache:
            cols={}
            for f in ('open','high','low','close','volume','factor'):
                s=base.qb.read_bin(code,f,cal)
                if not s.empty: cols[f]=s.loc[t1.WARM:t1.END]
            if not all(f in cols for f in ('open','high','low','close','volume')):
                cache[code]=None
            else:
                z=pd.concat(cols,axis=1)
                if 'factor' not in z:z['factor']=1.0
                z['factor']=z.factor.replace(0,np.nan).ffill().fillna(1.0)
                z['raw_close']=z.close/z.factor
                z['prev_raw_close']=z.raw_close.ffill().shift(1)
                cache[code]=z
        z=cache.get(code)
        if z is None or td not in z.index:return None
        r=z.loc[td]
        vals=[r.open,r.high,r.low,r.close,r.volume,r.factor]
        if not np.isfinite(vals).all() or float(r.open)<=0 or float(r.volume)<=0 or float(r.factor)<=0:return None
        rawopen=float(r.open/r.factor); pc=float(r.prev_raw_close) if np.isfinite(r.prev_raw_close) else np.nan
        lim=t1.board_limit(code,td); gap=rawopen/pc-1 if np.isfinite(pc) and pc>0 else np.nan
        return {'open':float(r.open),'high':float(r.high),'low':float(r.low),'close':float(r.close),'volume':float(r.volume),'factor':float(r.factor),'gap':gap,'lim':lim,
                'buy_allowed':bool(np.isfinite(gap) and gap<lim-.002),'sell_allowed':bool(np.isfinite(gap) and gap>-lim+.002)}
    def close(code,td):
        q=get(code,td); return np.nan if q is None else q['close']
    return get,close


def simulate_persistent(signals,cfg,exec_rows,tc,cal,members,bm,cost_mult=1.0,reverse=False):
    # Signal state controls desired holdings. After memory expiry, a held name is absent from target;
    # if it cannot sell, it remains and the sell is retried every following session until executable.
    sch=t1.schedule(signals,cfg,tc,reverse)
    if sch.empty:return None,pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    sch['signal_date']=pd.to_datetime(sch.signal_date)
    by={d:g.copy() for d,g in sch.groupby('signal_date')}
    x=exec_rows.set_index(['signal_date','code'],drop=False)
    getq,getclose=build_quote_cache(cal)
    member_end=members.groupby('code').end.max().to_dict()
    cash=float(cfg['initial_cash']); pos={}; equity=[]; trades=[]; timing=[]; turnover=0.0
    slip=sim.SLIPPAGE*float(cost_mult); n=int(cfg['n_hold']); entry=float(cfg['entry_pct']); keep=float(cfg['keep_pct']); vp=float(cfg['volume_participation'])
    old_n,old_e,old_k=hard.N_HOLD,sim.ENTRY_PCT,sim.KEEP_PCT
    hard.N_HOLD=n; sim.ENTRY_PCT=entry; sim.KEEP_PCT=keep
    try:
      trade_days=pd.DatetimeIndex(tc)
      for i,d in enumerate(trade_days[:-1]):
        d=pd.Timestamp(d); td=pd.Timestamp(trade_days[i+1])
        sg=by.get(d,pd.DataFrame(columns=['code','rank_test']))
        if len(sg):
            gg=sg[['code','rank_test']].copy(); gg['ivol60_pct']=gg.rank_test; gg['liq20']=1.0
            target=hard.choose_det(gg,set(pos))
        else: target=[]
        tgt=set(target)
        # mark at open / membership writeoff
        for c,pp in list(pos.items()):
            q=getq(c,td)
            if q is not None: pp.last_price=q['open']
            elif pd.Timestamp(member_end.get(c,t1.END))<td:
                old=pos.pop(c); trades.append({'variant':'t1ie_persistent','code':c,'entry_date':old.entry_date,'exit_date':td,'net_pnl':-old.entry_cost,'net_return':-1.0,'exit_reason':'membership_end_writeoff'})
        nav_open=cash+sum(pp.units*pp.last_price for pp in pos.values())
        # persistent exits: every day once no longer in target
        for c in sorted(list(pos)):
            if c in tgt: continue
            q=getq(c,td)
            if q is None or not q['sell_allowed']: continue
            locked=abs(q['high']-q['low'])<1e-12 and abs(q['open']-q['high'])<1e-12
            if locked: continue
            px=q['open']*(1-slip); gross=pos[c].units*px; fee=sim.fee(gross,'sell',td,cost_mult); old=pos.pop(c)
            cash+=gross-fee; turnover+=gross
            trades.append({'variant':'t1ie_persistent','code':c,'entry_date':old.entry_date,'exit_date':td,'net_pnl':gross-fee-old.entry_cost,'net_return':(gross-fee)/old.entry_cost-1,'exit_reason':'rank_exit'})
            timing.append({'variant':'t1ie_persistent','signal_date':d,'trade_date':td,'side':'sell','code':c,'gross':gross})
        # buy target entrants; use cached execution shard first, direct quote only as fallback
        per=nav_open*.99/n
        for c in target:
            if len(pos)>=n: break
            if c in pos: continue
            q=None
            try:
                r=x.loc[(d,c)]
                if isinstance(r,pd.DataFrame):r=r.iloc[-1]
                if np.isfinite([r.exec_open,r.exec_high,r.exec_low,r.exec_volume,r.exec_factor]).all():
                    q={'open':float(r.exec_open),'high':float(r.exec_high),'low':float(r.exec_low),'volume':float(r.exec_volume),'factor':float(r.exec_factor),
                       'buy_allowed':bool(r.exec_buy_allowed) if 'exec_buy_allowed' in r else True}
            except KeyError: pass
            if q is None:
                q0=getq(c,td)
                if q0 is not None:q=q0
            if q is None or not q.get('buy_allowed',True):continue
            locked=abs(q['high']-q['low'])<1e-12 and abs(q['open']-q['high'])<1e-12
            if locked or q['factor']<=0:continue
            factor=q['factor']; adjpx=q['open']*(1+slip); rawpx=adjpx/factor
            if not np.isfinite(rawpx) or rawpx<=0:continue
            rawvol=hv2.raw_share_volume(q['volume'],factor); maxraw=hv2.max_participation_shares(q['volume'],factor,vp)
            shares=int(min(per,cash*.98)//(rawpx*100))*100; shares=min(shares,maxraw)
            if shares<=0:continue
            units=shares/factor; gross=units*adjpx; fee=sim.fee(gross,'buy',td,cost_mult); total=gross+fee
            if total>cash:continue
            cash-=total; pos[c]=sim.Pos(units,total,td,q['open']); turnover+=gross
            timing.append({'variant':'t1ie_persistent','signal_date':d,'trade_date':td,'side':'buy','code':c,'gross':gross})
        if len(pos)>n:raise RuntimeError('position cap violation')
        # close MTM for this execution day
        for c,pp in pos.items():
            px=getclose(c,td)
            if np.isfinite(px) and px>0:pp.last_price=float(px)
        nav=cash+sum(pp.units*pp.last_price for pp in pos.values())
        equity.append({'variant':'t1ie_persistent','signal_date':d,'trade_date':td,'equity':nav,'cash':cash,'positions':len(pos)})
      eq=pd.DataFrame(equity).drop_duplicates('trade_date',keep='last').sort_values('trade_date'); tr=pd.DataFrame(trades); tm=pd.DataFrame(timing)
      if len(tm) and (pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).any():raise RuntimeError('timing violation')
      f=t1.perf_series(eq); train=t1.perf_series(eq,t1.START,t1.TRAIN_END); h1=t1.perf_series(eq,t1.START,t1.HALF1_END); h2=t1.perf_series(eq,t1.HALF2_START,t1.TRAIN_END); ps=t1.perf_series(eq,t1.PSEUDO,t1.END)
      out={**f,'train_cagr':train['cagr'],'train_mdd':train['max_drawdown'],'train_sharpe':train['sharpe'],'train_calmar':train['calmar'],'half1_cagr':h1['cagr'],'half2_cagr':h2['cagr'],'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['max_drawdown'],'pseudo_sharpe':ps['sharpe'],'turnover':turnover,'positions_max':int(eq.positions.max()) if len(eq) else 0,'positions_median':float(eq.positions.median()) if len(eq) else 0}
      return out,eq,tr,tm
    finally:
      hard.N_HOLD,sim.ENTRY_PCT,sim.KEEP_PCT=old_n,old_e,old_k


def grid_variant(name):
    cal,members,ua,market_code,bm,tc=setup(); X=load_exec(); sig=load_signal(name,False); rows=[]
    for mem in t1.MEMORIES:
      for n in t1.NS:
        cfg=t1.cfg_from(name,t1.VARIANTS[name],mem,n)
        print('GRID PERSISTENT',name,mem,n,flush=True)
        st,eq,tr,tm=simulate_persistent(sig,cfg,X,tc,cal,members,bm,1.0,False)
        if st is None:continue
        rows.append({**st,'key':f'{name}|m{mem}|n{n}','variant':name,'memory_sessions':mem,'n_hold':n,'signal_count':len(sig),'signal_days':sig.signal_date.nunique()})
    d=pd.DataFrame(rows); d.to_csv(GRID/f'grid_{name}.csv',index=False); print(d.to_string(index=False),flush=True)


def select_grid():
    fs=glob.glob('grid_input/**/grid_*.csv',recursive=True); G=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
    good=G[(G.train_cagr>0)&(G.train_mdd>-.45)&(G.half1_cagr>0)&(G.half2_cagr>0)].copy()
    if good.empty:good=G[(G.train_cagr>0)&(G.train_mdd>-.45)].copy()
    if good.empty:good=G.copy()
    win=good.sort_values(['train_calmar','train_sharpe','turnover'],ascending=[False,False,True]).iloc[0]
    return G,win


def final():
    cal,members,ua,market_code,bm,tc=setup(); X=load_exec(); G,win=select_grid(); G.to_csv(FINAL/'grid.csv',index=False)
    name=str(win.variant); mem=int(win.memory_sessions); n=int(win.n_hold); cfg=t1.cfg_from(name,t1.VARIANTS[name],mem,n); sig=load_signal(name,False)
    st,eq,tr,tm=simulate_persistent(sig,cfg,X,tc,cal,members,bm,1.0,False)
    pd.DataFrame([{**st,'key':f'{name}|m{mem}|n{n}','variant':name,'memory_sessions':mem,'n_hold':n,'signal_count':len(sig),'signal_days':sig.signal_date.nunique()}]).to_csv(FINAL/'selected_metrics.csv',index=False)
    sig.to_csv(FINAL/'selected_signals.csv.gz',index=False,compression='gzip'); eq.to_csv(FINAL/'selected_equity.csv',index=False); tr.to_csv(FINAL/'selected_trades.csv',index=False); tm.to_csv(FINAL/'selected_timing.csv',index=False); t1.annual(eq).to_csv(FINAL/'selected_annual.csv',index=False)
    stress=[]
    for cm in (2.0,4.0):
        s,e,tt,mt=simulate_persistent(sig,cfg,X,tc,cal,members,bm,cm,False); stress.append({'cost_mult':cm,**s}); e.to_csv(FINAL/f'selected_equity_cost{int(cm)}.csv',index=False)
    pd.DataFrame(stress).to_csv(FINAL/'cost_stress.csv',index=False)
    sr,er,trr,tmr=simulate_persistent(sig,cfg,X,tc,cal,members,bm,1.0,True); pd.DataFrame([{'control':'reverse_score',**sr}]).to_csv(FINAL/'reverse_control.csv',index=False)
    esig=load_signal(name,True); se,ee,te,tme=simulate_persistent(esig,cfg,X,tc,cal,members,bm,1.0,False); pd.DataFrame([{'control':'event_only_no_flush_confirmation',**se,'signal_count':len(esig)}]).to_csv(FINAL/'event_only_ablation.csv',index=False)
    timing_bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    durs=(pd.to_datetime(tr.exit_date)-pd.to_datetime(tr.entry_date)).dt.days if len(tr) and 'exit_date' in tr else pd.Series(dtype=float)
    stx=pd.DataFrame(stress).set_index('cost_mult')
    gates={'train_cagr_positive':int(st['train_cagr']>0),'pseudo_cagr_positive':int(st['pseudo_cagr']>0),'train_sharpe_positive':int(st['train_sharpe']>0),'pseudo_sharpe_positive':int(st['pseudo_sharpe']>0),'full_mdd_better_than_minus45':int(st['max_drawdown']>-.45),'cost2_cagr_positive':int(float(stx.loc[2.0,'cagr'])>0),'absorption_train_calmar_gt_event_only':int(st['train_calmar']>se['train_calmar'] if se is not None and np.isfinite(se['train_calmar']) else 0),'timing_zero':int(timing_bad==0)}
    pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_csv(FINAL/'gates.csv',index=False)
    pd.DataFrame([{'market_factor':market_code,'timing_violations':timing_bad,'trades':len(tr),'max_holding_calendar_days':float(durs.max()) if len(durs) else np.nan,'holding_days_p99':float(durs.quantile(.99)) if len(durs) else np.nan,'persistent_exit_fix':1}]).to_csv(FINAL/'audit.csv',index=False)
    spec={'label':'NEW_STOCK_LEVEL_CAUSAL_SHORT_ALPHA_RESEARCH_NOT_ORIGINAL_EXACT','alpha':'T1 Inventory Exhaustion (T1-IE)','selected_key':f'{name}|m{mem}|n{n}','selected_config':cfg,'market_factor':market_code,'selection_uses':'2016-08-02..2021-12-31 only','pseudo':'2022-01-01..2026-07-29 research diagnostic, not clean OOS','prereg':'T1_INVENTORY_EXHAUSTION_PREREG_2026-09-04.md','execution_fix':'blocked exits persist and retry every session until executable; signal rules and search grid unchanged','gates_passed':sum(gates.values()),'gates_total':len(gates),'universe_audit':ua}
    (FINAL/'strategy_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2,default=str))
    print('=== SELECTED ===');print(pd.DataFrame([{**st,'variant':name,'memory_sessions':mem,'n_hold':n}]).to_string(index=False),flush=True)
    print('=== COST ===');print(pd.DataFrame(stress).to_string(index=False),flush=True)
    print('=== EVENT ONLY ===');print(pd.DataFrame([{'control':'event_only_no_flush_confirmation',**se,'signal_count':len(esig)}]).to_string(index=False),flush=True)
    print('=== REVERSE ===');print(pd.DataFrame([{'control':'reverse_score',**sr}]).to_string(index=False),flush=True)
    print('=== GATES ===');print(pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_string(index=False),flush=True)
    print('=== AUDIT ===');print(pd.read_csv(FINAL/'audit.csv').to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['grid','final']); ap.add_argument('--variant',choices=list(t1.VARIANTS)); a=ap.parse_args()
    if a.mode=='grid':grid_variant(a.variant)
    else:final()
if __name__=='__main__':main()
