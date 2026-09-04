from __future__ import annotations

from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_max_audit as ma
import run_10y_skewfilter_hard as hard
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_short_t1_inventory_v1'); OUT.mkdir(exist_ok=True)
RUNTIME=Path('short_t1_inventory_runtime.json')
START=pd.Timestamp('2016-08-02'); TRAIN_END=pd.Timestamp('2021-12-31')
HALF1_END=pd.Timestamp('2019-12-31'); HALF2_START=pd.Timestamp('2020-01-01')
PSEUDO=pd.Timestamp('2022-01-01'); END=pd.Timestamp('2026-07-29')
WARM=pd.Timestamp('2015-01-01')
ENTRY=.10; KEEP=.30

VARIANTS={
 'balanced': dict(event_pressure_min=.75,event_fade_min=.12,event_close_seal_max=.90,event_vol_ratio_min=1.30,flush_ret_n_min=-.50,flush_ret_n_max=.10,flush_gap_n_max=.10,flush_vol_ratio_max=.80,flush_recovery_min=.55,flush_low_n_max=-.08,liquidity_keep_pct=.70),
 'strict_pressure': dict(event_pressure_min=.85,event_fade_min=.12,event_close_seal_max=.90,event_vol_ratio_min=1.30,flush_ret_n_min=-.50,flush_ret_n_max=.10,flush_gap_n_max=.10,flush_vol_ratio_max=.80,flush_recovery_min=.55,flush_low_n_max=-.08,liquidity_keep_pct=.70),
 'strict_absorption': dict(event_pressure_min=.75,event_fade_min=.12,event_close_seal_max=.90,event_vol_ratio_min=1.30,flush_ret_n_min=-.50,flush_ret_n_max=.10,flush_gap_n_max=.10,flush_vol_ratio_max=.65,flush_recovery_min=.65,flush_low_n_max=-.08,liquidity_keep_pct=.70),
 'deep_trap': dict(event_pressure_min=.75,event_fade_min=.20,event_close_seal_max=.90,event_vol_ratio_min=1.50,flush_ret_n_min=-.50,flush_ret_n_max=.10,flush_gap_n_max=.10,flush_vol_ratio_max=.80,flush_recovery_min=.55,flush_low_n_max=-.08,liquidity_keep_pct=.70),
 'quiet_release': dict(event_pressure_min=.75,event_fade_min=.12,event_close_seal_max=.90,event_vol_ratio_min=1.50,flush_ret_n_min=-.50,flush_ret_n_max=.10,flush_gap_n_max=.10,flush_vol_ratio_max=.65,flush_recovery_min=.55,flush_low_n_max=-.08,liquidity_keep_pct=.70),
 'high_recovery': dict(event_pressure_min=.75,event_fade_min=.12,event_close_seal_max=.90,event_vol_ratio_min=1.30,flush_ret_n_min=-.50,flush_ret_n_max=.10,flush_gap_n_max=.10,flush_vol_ratio_max=.80,flush_recovery_min=.70,flush_low_n_max=-.08,liquidity_keep_pct=.70),
}
MEMORIES=(3,5,8); NS=(5,10,15); EXIT_RETRY=10


def board_limit(code,d):
    s=str(code).upper(); d=pd.Timestamp(d)
    if s.startswith('BJ'): return .30
    if s.startswith('SH688'): return .20
    if (s.startswith('SZ300') or s.startswith('SZ301')) and d>=pd.Timestamp('2020-08-24'): return .20
    return .10


def active_mask(mm,dates):
    out=np.zeros(len(dates),dtype=bool)
    for r in mm.itertuples(index=False):
        out |= (dates>=pd.Timestamp(r.start))&(dates<=pd.Timestamp(r.end))
    return out


def perf_series(eq,a=None,b=None):
    if eq is None or len(eq)<2:return dict(cagr=np.nan,max_drawdown=np.nan,sharpe=np.nan,total_return=np.nan,calmar=np.nan)
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index()
    if a is not None:s=s[s.index>=pd.Timestamp(a)]
    if b is not None:s=s[s.index<=pd.Timestamp(b)]
    if len(s)<20:return dict(cagr=np.nan,max_drawdown=np.nan,sharpe=np.nan,total_return=np.nan,calmar=np.nan)
    s=s/s.iloc[0]; r=s.pct_change().dropna(); days=max(1,(s.index[-1]-s.index[0]).days)
    c=float(s.iloc[-1]**(365.25/days)-1); dd=float((s/s.cummax()-1).min())
    sh=float(r.mean()/r.std(ddof=1)*np.sqrt(252)) if len(r)>2 and r.std(ddof=1)>0 else np.nan
    return dict(cagr=c,max_drawdown=dd,sharpe=sh,total_return=float(s.iloc[-1]-1),calmar=float(c/abs(dd)) if dd<0 else np.nan)


def next_map(cal):
    tc=cal[(cal>=START)&(cal<=END)]
    return tc,{pd.Timestamp(tc[i]):pd.Timestamp(tc[i+1]) for i in range(len(tc)-1)}


def build_features(cal,members):
    tc,nxt=next_map(cal); sigdates=tc[:-1]
    codes=sorted(members.code.unique())
    liq=np.full((len(codes),len(tc)),np.nan,dtype=np.float32)
    cand=[]; events=[]
    for i,code in enumerate(codes):
        mm=members[members.code==code]
        cols={}
        for f in ('open','high','low','close','volume','factor'):
            s=base.qb.read_bin(code,f,cal)
            if not s.empty: cols[f]=s.loc[WARM:END]
        if not all(f in cols for f in ('open','high','low','close','volume')):continue
        z=pd.concat(cols,axis=1)
        if 'factor' not in z:z['factor']=1.0
        z['factor']=z.factor.replace(0,np.nan).ffill().fillna(1.0)
        for c in ('open','high','low','close'):z['raw_'+c]=z[c]/z.factor
        z['rawvol']=z.volume.abs()*z.factor.abs()*100.0
        z['turnover']=z.raw_close.abs()*z.rawvol.abs()
        z['liq20']=z.turnover.rolling(20,min_periods=15).mean()
        liq[i,:]=z.liq20.reindex(tc).to_numpy(dtype=np.float32)
        z['prev']=z.raw_close.shift(1)
        lim=pd.Series([board_limit(code,d) for d in z.index],index=z.index,dtype=float)
        z['pressure']=(z.raw_high/z.prev-1)/lim
        z['close_pressure']=(z.raw_close/z.prev-1)/lim
        z['fade']=(z.raw_high-z.raw_close)/z.prev/lim
        z['volbase']=z.rawvol.shift(1).rolling(20,min_periods=12).median()
        z['event_vr']=z.rawvol/z.volbase
        # Flush-day features reference the immediately preceding event day.
        z['event_pressure']=z.pressure.shift(1); z['event_fade']=z.fade.shift(1)
        z['event_close_pressure']=z.close_pressure.shift(1); z['event_vol_ratio']=z.event_vr.shift(1)
        z['event_rawvol']=z.rawvol.shift(1)
        z['flush_gap_n']=(z.raw_open/z.raw_close.shift(1)-1)/lim
        z['flush_ret_n']=(z.raw_close/z.raw_close.shift(1)-1)/lim
        z['flush_low_n']=(z.raw_low/z.raw_close.shift(1)-1)/lim
        z['flush_vol_ratio']=z.rawvol/z.event_rawvol
        rng=(z.raw_high-z.raw_low).replace(0,np.nan)
        z['flush_recovery']=(z.raw_close-z.raw_low)/rng
        z['age120']=z.raw_close.notna().rolling(120,min_periods=120).sum()
        av=active_mask(mm,sigdates); av2=active_mask(mm,pd.DatetimeIndex([nxt[pd.Timestamp(d)] for d in sigdates]))
        valid=av&av2&(z.age120.reindex(sigdates).to_numpy()>=120)
        # Broad envelope contains every preregistered threshold and leaves room for future config changes.
        f=z.reindex(sigdates)
        broad=valid & (f.event_pressure.to_numpy()>=.50)&(f.event_fade.to_numpy()>=.03)&(f.event_close_pressure.to_numpy()<1.02)&(f.event_vol_ratio.to_numpy()>=.80)&(f.flush_vol_ratio.to_numpy()<=1.25)&(f.flush_recovery.to_numpy()>=.15)&(f.flush_ret_n.to_numpy()>=-.85)&(f.flush_ret_n.to_numpy()<=.35)
        if broad.any():
            d=f.loc[sigdates[broad],['event_pressure','event_fade','event_close_pressure','event_vol_ratio','flush_gap_n','flush_ret_n','flush_low_n','flush_vol_ratio','flush_recovery','liq20']].copy()
            d['signal_date']=d.index; d['trade_date']=[nxt[pd.Timestamp(x)] for x in d.index]; d['code']=code
            cand.append(d.reset_index(drop=True))
        eb=valid & (f.pressure.to_numpy()>=.50)&(f.fade.to_numpy()>=.03)&(f.close_pressure.to_numpy()<1.02)&(f.event_vr.to_numpy()>=.80)
        if eb.any():
            d=f.loc[sigdates[eb],['pressure','fade','close_pressure','event_vr','liq20']].copy()
            d=d.rename(columns={'pressure':'event_pressure','fade':'event_fade','close_pressure':'event_close_pressure','event_vr':'event_vol_ratio'})
            d['signal_date']=d.index; d['trade_date']=[nxt[pd.Timestamp(x)] for x in d.index]; d['code']=code
            events.append(d.reset_index(drop=True))
        if (i+1)%1000==0:print('FEATURE',i+1,'/',len(codes),flush=True)
    C=pd.concat(cand,ignore_index=True) if cand else pd.DataFrame(); E=pd.concat(events,ignore_index=True) if events else pd.DataFrame()
    return C,E,liq,tc,codes,nxt


def liq_thresholds(liq,tc,keep_pct):
    # top keep_pct by liquidity means threshold at lower (1-keep_pct) quantile.
    q=max(0.,min(1.,1-float(keep_pct)))
    vals=np.nanquantile(liq,q,axis=0)
    return pd.Series(vals,index=tc)


def filter_signals(C,cfg,liq,tc,event_only=False):
    th=liq_thresholds(liq,tc,cfg['liquidity_keep_pct'])
    x=C.copy(); x['liq_threshold']=pd.to_datetime(x.signal_date).map(th)
    m=x.liq20>=x.liq_threshold
    m &= x.event_pressure>=cfg['event_pressure_min']
    m &= x.event_fade>=cfg['event_fade_min']
    m &= x.event_close_pressure<cfg['event_close_seal_max']
    m &= x.event_vol_ratio>=cfg['event_vol_ratio_min']
    if not event_only:
        m &= x.flush_ret_n.between(cfg['flush_ret_n_min'],cfg['flush_ret_n_max'])
        m &= x.flush_gap_n<=cfg['flush_gap_n_max']
        m &= x.flush_vol_ratio<=cfg['flush_vol_ratio_max']
        m &= x.flush_recovery>=cfg['flush_recovery_min']
        m &= x.flush_low_n<=cfg['flush_low_n_max']
    x=x[m].copy()
    if x.empty:return x
    if event_only:
        comps=[('event_pressure',.40,False),('event_fade',.30,False),('event_vol_ratio',.30,False)]
    else:
        x['vol_compression']=1-x.flush_vol_ratio
        x['flush_depth']=(-x.flush_ret_n).clip(0,.50)
        comps=[('event_pressure',.25,False),('event_fade',.20,False),('vol_compression',.25,False),('flush_recovery',.25,False),('flush_depth',.05,False)]
    raw=pd.Series(0.,index=x.index)
    for c,w,_ in comps:
        # high component is good; ascending=False makes best percentile smallest.
        r=x.groupby('signal_date')[c].rank(pct=True,method='average',ascending=False)
        raw += w*r
    x['score_rank']=raw.groupby(x.signal_date).rank(pct=True,method='average',ascending=True)
    return x


def build_exec_rows(cal,members,all_origins,tc,max_window):
    nxt={pd.Timestamp(tc[i]):pd.Timestamp(tc[i+1]) for i in range(len(tc)-1)}; pos={pd.Timestamp(d):i for i,d in enumerate(tc)}
    rows=[]
    for j,(code,g) in enumerate(all_origins.groupby('code'),1):
        mm=members[members.code==code]; need=set()
        for d in pd.to_datetime(g.signal_date).unique():
            k=pos.get(pd.Timestamp(d));
            if k is None:continue
            for h in range(max_window+1):
                if k+h < len(tc)-1:need.add(pd.Timestamp(tc[k+h]))
        if not need:continue
        cols={}
        for f in ('open','high','low','close','volume','factor'):
            s=base.qb.read_bin(code,f,cal)
            if not s.empty:cols[f]=s.loc[WARM:END]
        if not all(f in cols for f in ('open','high','low','close','volume')):continue
        z=pd.concat(cols,axis=1)
        if 'factor' not in z:z['factor']=1.0
        z['factor']=z.factor.replace(0,np.nan).ffill().fillna(1.0)
        z['raw_close']=z.close/z.factor; prevraw=z.raw_close.ffill().shift(1)
        for d in sorted(need):
            td=nxt.get(d)
            if td is None or not active_mask(mm,pd.DatetimeIndex([d]))[0] or not active_mask(mm,pd.DatetimeIndex([td]))[0]:continue
            if td not in z.index:continue
            r=z.loc[td]
            if not np.isfinite([r.open,r.high,r.low,r.volume,r.factor]).all():continue
            rawopen=float(r.open/r.factor); pc=float(prevraw.get(td,np.nan)); lim=board_limit(code,td); gap=rawopen/pc-1 if np.isfinite(pc) and pc>0 else np.nan
            rows.append({'signal_date':d,'trade_date':td,'code':code,'liq20':1.0,'exec_open':float(r.open),'exec_high':float(r.high),'exec_low':float(r.low),'exec_volume':float(r.volume),'exec_factor':float(r.factor),'exec_open_gap':gap,'exec_limit_proxy':lim,'exec_buy_allowed':bool(np.isfinite(gap) and gap<lim-.002),'exec_sell_allowed':bool(np.isfinite(gap) and gap>-lim+.002)})
        if j%1000==0:print('EXEC',j,'/',all_origins.code.nunique(),flush=True)
    return pd.DataFrame(rows)


def schedule(signals,cfg,tc,reverse=False):
    if signals.empty:return pd.DataFrame(columns=['signal_date','code','rank_test'])
    x=signals.copy(); er=x.score_rank.copy()
    if reverse:er=1-er+er.groupby(x.signal_date).transform('min')*0
    # every fresh event is eligible to enter; cross-sectional score only determines priority when slots are scarce.
    x['entry_rank']=np.minimum(float(cfg['entry_pct'])*.95, er*float(cfg['entry_pct'])*.95)
    pos={pd.Timestamp(d):i for i,d in enumerate(tc)}; rec=[]; keep_rank=(float(cfg['entry_pct'])+float(cfg['keep_pct']))/2
    mem=int(cfg['memory_sessions']); retry=int(cfg['exit_retry_sessions'])
    for r in x.itertuples(index=False):
        d=pd.Timestamp(r.signal_date); k=pos.get(d)
        if k is None:continue
        rec.append({'signal_date':d,'code':r.code,'rank_test':float(r.entry_rank),'state':'entry'})
        for h in range(1,mem):
            if k+h<len(tc)-1:rec.append({'signal_date':pd.Timestamp(tc[k+h]),'code':r.code,'rank_test':keep_rank,'state':'keep'})
        for h in range(mem,mem+retry+1):
            if k+h<len(tc)-1:rec.append({'signal_date':pd.Timestamp(tc[k+h]),'code':r.code,'rank_test':np.nan,'state':'exit'})
    s=pd.DataFrame(rec)
    if s.empty:return s
    # Fresh entry overrides keep/exit, then keep overrides exit.
    pri={'entry':0,'keep':1,'exit':2}; s['_p']=s.state.map(pri)
    s=s.sort_values(['signal_date','code','_p','rank_test'],na_position='last').drop_duplicates(['signal_date','code'],keep='first').drop(columns='_p')
    return s


def panel_from(signals,cfg,exec_rows,tc,reverse=False):
    s=schedule(signals,cfg,tc,reverse)
    if s.empty:return pd.DataFrame()
    p=s.merge(exec_rows,on=['signal_date','code'],how='left',validate='many_to_one')
    p=p[np.isfinite(p[['exec_open','exec_high','exec_low','exec_volume','exec_factor']]).all(axis=1)].copy()
    p['ivol60_pct']=p.rank_test
    return p.drop(columns=['rank_test','state'])


def run_panel(p,cfg,cal,members,bm,cost=1.0):
    if p.empty:return None,pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    st,eq,tr,tm=ma.run_panel(p,cal,members,bm,n=int(cfg['n_hold']),entry=float(cfg['entry_pct']),keep=float(cfg['keep_pct']),cost=float(cost),initial_cash=float(cfg['initial_cash']),vol_part=float(cfg['volume_participation']),start=START,end=END)
    f=perf_series(eq); train=perf_series(eq,START,TRAIN_END); h1=perf_series(eq,START,HALF1_END); h2=perf_series(eq,HALF2_START,TRAIN_END); ps=perf_series(eq,PSEUDO,END)
    out={**f,'train_cagr':train['cagr'],'train_mdd':train['max_drawdown'],'train_sharpe':train['sharpe'],'train_calmar':train['calmar'],'half1_cagr':h1['cagr'],'half2_cagr':h2['cagr'],'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['max_drawdown'],'pseudo_sharpe':ps['sharpe'],'turnover':float(st.get('turnover',np.nan)),'positions_max':int(eq.positions.max()) if len(eq) else 0,'positions_median':float(eq.positions.median()) if len(eq) else 0}
    return out,eq,tr,tm


def annual(eq):
    if eq.empty:return pd.DataFrame(columns=['year','return'])
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); rows=[]
    for y,g in s.groupby(s.index.year):
        before=s[s.index<pd.Timestamp(f'{y}-01-01')]; start=float(before.iloc[-1]) if len(before) else float(g.iloc[0]); rows.append({'year':int(y),'return':float(g.iloc[-1]/start-1)})
    return pd.DataFrame(rows)


def cfg_from(name,basecfg,mem,n):
    d={**basecfg}; d.update({'name':name,'memory_sessions':int(mem),'n_hold':int(n),'entry_pct':ENTRY,'keep_pct':KEEP,'exit_retry_sessions':EXIT_RETRY,'volume_participation':.05,'initial_cash':1_000_000.0})
    return d


def main():
    runtime=json.loads(RUNTIME.read_text()) if RUNTIME.exists() else {'mode':'sweep'}
    base.START=START; base.WARM=WARM; base.END=END; sim.START=START; sim.WARM=WARM; sim.END=END
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal); bm=market_close.loc[START:END].dropna()
    C,E,liq,tc,codes,nxt=build_features(cal,members)
    C.to_csv(OUT/'broad_flush_candidates.csv.gz',index=False,compression='gzip'); E.to_csv(OUT/'broad_event_candidates.csv.gz',index=False,compression='gzip')
    max_window=max(max(MEMORIES)+EXIT_RETRY,int(runtime.get('custom',{}).get('memory_sessions',5))+int(runtime.get('custom',{}).get('exit_retry_sessions',EXIT_RETRY)))
    origins=pd.concat([C[['signal_date','code']],E[['signal_date','code']]],ignore_index=True).drop_duplicates()
    X=build_exec_rows(cal,members,origins,tc,max_window); X.to_pickle(OUT/'execution_rows.pkl')

    grid=[]; cache={}
    configs=[]
    if runtime.get('mode','sweep')=='custom':
        c=runtime['custom'].copy(); configs=[c]
    else:
        for name,v in VARIANTS.items():
            for mem in MEMORIES:
                for n in NS:configs.append(cfg_from(name,v,mem,n))
    for ix,cfg in enumerate(configs,1):
        print('RUN',ix,'/',len(configs),cfg['name'],cfg['memory_sessions'],cfg['n_hold'],flush=True)
        sig=filter_signals(C,cfg,liq,tc,False); p=panel_from(sig,cfg,X,tc,False); st,eq,tr,tm=run_panel(p,cfg,cal,members,bm,1.0)
        if st is None:continue
        key=f"{cfg['name']}|m{cfg['memory_sessions']}|n{cfg['n_hold']}"
        row={**st,'key':key,'variant':cfg['name'],'memory_sessions':cfg['memory_sessions'],'n_hold':cfg['n_hold'],'signal_count':len(sig),'signal_days':sig.signal_date.nunique()}; grid.append(row); cache[key]=(cfg,sig,p,eq,tr,tm)
    G=pd.DataFrame(grid); G.to_csv(OUT/'grid.csv',index=False)
    if G.empty:raise RuntimeError('no runnable configurations')
    good=G[(G.train_cagr>0)&(G.train_mdd>-.45)&(G.half1_cagr>0)&(G.half2_cagr>0)].copy()
    if good.empty:good=G[(G.train_cagr>0)&(G.train_mdd>-.45)].copy()
    if good.empty:good=G.copy()
    win=good.sort_values(['train_calmar','train_sharpe','turnover'],ascending=[False,False,True]).iloc[0]; key=str(win.key)
    cfg,sig,p,eq,tr,tm=cache[key]
    pd.DataFrame([win]).to_csv(OUT/'selected_metrics.csv',index=False); sig.to_csv(OUT/'selected_signals.csv.gz',index=False,compression='gzip'); eq.to_csv(OUT/'selected_equity.csv',index=False); tr.to_csv(OUT/'selected_trades.csv',index=False); tm.to_csv(OUT/'selected_timing.csv',index=False); annual(eq).to_csv(OUT/'selected_annual.csv',index=False)
    sig.assign(year=pd.to_datetime(sig.signal_date).dt.year).groupby('year').size().rename('signals').reset_index().to_csv(OUT/'selected_signal_counts_year.csv',index=False)

    stress=[]
    for cm in (2.0,4.0):
        st2,e2,t2,tm2=run_panel(p,cfg,cal,members,bm,cm); stress.append({'cost_mult':cm,**st2}); e2.to_csv(OUT/f'selected_equity_cost{int(cm)}.csv',index=False)
    pd.DataFrame(stress).to_csv(OUT/'cost_stress.csv',index=False)

    # Reverse ranking control on identical events.
    pr=panel_from(sig,cfg,X,tc,True); sr,er,trr,tmr=run_panel(pr,cfg,cal,members,bm,1.0); pd.DataFrame([{'control':'reverse_score',**sr}]).to_csv(OUT/'reverse_control.csv',index=False)

    # Event-only ablation: same event thresholds but enter after event day without requiring the T+1 flush/absorption day.
    esig=filter_signals(E,cfg,liq,tc,True); pe=panel_from(esig,cfg,X,tc,False); se,ee,te,tme=run_panel(pe,cfg,cal,members,bm,1.0); pd.DataFrame([{'control':'event_only_no_flush_confirmation',**se,'signal_count':len(esig)}]).to_csv(OUT/'event_only_ablation.csv',index=False)

    timing_bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    st2=pd.DataFrame(stress).set_index('cost_mult')
    gates={
      'train_cagr_positive':int(float(win.train_cagr)>0),
      'pseudo_cagr_positive':int(float(win.pseudo_cagr)>0),
      'train_sharpe_positive':int(float(win.train_sharpe)>0),
      'pseudo_sharpe_positive':int(float(win.pseudo_sharpe)>0),
      'full_mdd_better_than_minus45':int(float(win.max_drawdown)>-.45),
      'cost2_cagr_positive':int(float(st2.loc[2.0,'cagr'])>0),
      'absorption_train_calmar_gt_event_only':int(float(win.train_calmar)>float(se['train_calmar']) if se is not None and np.isfinite(se['train_calmar']) else 0),
      'timing_zero':int(timing_bad==0),
    }
    pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_csv(OUT/'gates.csv',index=False)
    codehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec={'label':'NEW_STOCK_LEVEL_CAUSAL_SHORT_ALPHA_RESEARCH_NOT_ORIGINAL_EXACT','alpha':'T1 Inventory Exhaustion (T1-IE)','selected_key':key,'selected_config':cfg,'market_factor':market_code,'runtime':runtime,'code_sha256':codehash,'prereg':'T1_INVENTORY_EXHAUSTION_PREREG_2026-09-04.md','selection_uses':'2016-08-02..2021-12-31 only','pseudo':'2022-01-01..2026-07-29 research diagnostic, not clean OOS','gates_passed':sum(gates.values()),'gates_total':len(gates),'universe_audit':ua}
    (OUT/'strategy_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2,default=str))
    pd.DataFrame([{'broad_flush_rows':len(C),'broad_event_rows':len(E),'execution_rows':len(X),'codes':len(codes),'market_factor':market_code,'timing_violations':timing_bad,'code_sha256':codehash}]).to_csv(OUT/'audit.csv',index=False)
    print('=== SELECTED ==='); print(pd.DataFrame([win]).to_string(index=False),flush=True)
    print('=== COST ==='); print(pd.DataFrame(stress).to_string(index=False),flush=True)
    print('=== EVENT ONLY ==='); print(pd.DataFrame([{'control':'event_only_no_flush_confirmation',**se,'signal_count':len(esig)}]).to_string(index=False),flush=True)
    print('=== REVERSE ==='); print(pd.DataFrame([{'control':'reverse_score',**sr}]).to_string(index=False),flush=True)
    print('=== GATES ==='); print(pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_string(index=False),flush=True)

if __name__=='__main__':main()
