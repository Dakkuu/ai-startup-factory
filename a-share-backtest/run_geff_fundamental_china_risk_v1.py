from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import glob, json, math
import numpy as np
import pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_baseline_maxopt_v3 as mo
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
import run_10y_hard_executor_v2 as hv2
import run_geff_fundamental_integrated_v3 as iv3
hv3.patch()

OUT=Path('results_geff_fundamental_china_risk_v1'); OUT.mkdir(exist_ok=True)
START=pd.Timestamp('2016-08-02'); TRAIN_END=pd.Timestamp('2021-12-31'); PSEUDO=pd.Timestamp('2022-01-01'); END=pd.Timestamp('2026-07-29')
HALF1_END=pd.Timestamp('2019-12-31'); HALF2_START=pd.Timestamp('2020-01-01')
H_ALL=(45,60,75,90,105); H_STOP=(60,75,90)
NS=(5,7,10,12,15); ENTRY=.10; KEEP=.30
BASE_CONTROL_H90_PHASE0=0.1970860277444281

@dataclass
class SPos:
    units: float
    entry_cost: float
    entry_date: pd.Timestamp
    entry_px: float
    last_price: float
    peak_close: float
    atr_pct: float
    last_rank: float

class PriceCache:
    def __init__(self,cal): self.cal=cal; self.d={}
    def get(self,code):
        if code not in self.d:
            cols={}
            for f in ('open','high','low','close','volume','factor'):
                s=base.qb.read_bin(code,f,self.cal)
                if len(s): cols[f]=s
            z=pd.concat(cols,axis=1).loc[pd.Timestamp('2015-01-01'):END].copy() if cols else pd.DataFrame()
            if len(z):
                if 'factor' not in z:z['factor']=1.0
                z['factor']=z.factor.replace(0,np.nan).fillna(1.0)
                pc=z.close.shift(1)
                tr=pd.concat([(z.high-z.low).abs(),(z.high-pc).abs(),(z.low-pc).abs()],axis=1).max(axis=1)
                z['atr20_pct']=(tr/pc.abs().replace(0,np.nan)).rolling(20,min_periods=15).mean().shift(1)
                z['prev_close']=pc
            self.d[code]=z
        return self.d[code]


def locate(pattern):
    hits=glob.glob(pattern,recursive=True)
    if not hits: raise FileNotFoundError(pattern)
    return hits[0]


def build_candidate():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    va=pd.read_csv(locate('artifact_cache/value/**/pit_value_attached.csv.gz'),compression='gzip',low_memory=False)
    sa=pd.read_csv(locate('artifact_cache/stmt/**/pit_attached.csv.gz'),compression='gzip',low_memory=False)
    iv3.verify_attach(p,va,'value'); iv3.verify_attach(p,sa,'3stmt')
    p2,z=iv3.fund_ranks(p,va,sa); q=iv3.build_candidates(p2)['mom_cfo10_qv10']
    return q,p2,cal,members,ua,market_code,bm


def chosen_dates(q,h,ph):
    ds=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,round(h/5)); return set(ds[ph::step])


def subset(q,h,ph):
    ds=chosen_dates(q,h,ph)
    cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(ds)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')


def slice_stats(eq,a,b):
    z=eq.copy(); z['trade_date']=pd.to_datetime(z.trade_date); z=z[(z.trade_date>=pd.Timestamp(a))&(z.trade_date<=pd.Timestamp(b))].copy()
    if len(z)<20:return {'cagr':np.nan,'mdd':np.nan,'sharpe':np.nan}
    s=z.set_index('trade_date').equity.astype(float).sort_index(); r=s.pct_change().dropna(); days=max(1,(s.index[-1]-s.index[0]).days)
    cagr=float((s.iloc[-1]/s.iloc[0])**(365.25/days)-1) if s.iloc[0]>0 and s.iloc[-1]>0 else np.nan
    dd=s/s.cummax()-1; sd=float(r.std(ddof=1)); sh=float(r.mean()/sd*np.sqrt(252)) if len(r)>2 and sd>0 else np.nan
    return {'cagr':cagr,'mdd':float(dd.min()),'sharpe':sh}


def concentration_run(q,cal,members,bm):
    rows=[]
    for n in NS:
      for h in H_ALL:
        for ph in range(max(1,round(h/5))):
          print('CONC N',n,'H',h,'PH',ph,flush=True)
          st,eq,tr,tm=ma.run_panel(subset(q,h,ph),cal,members,bm,n=n,entry=ENTRY,keep=KEEP,cost=1.0)
          trn=slice_stats(eq,START,TRAIN_END); h1=slice_stats(eq,START,HALF1_END); h2=slice_stats(eq,HALF2_START,TRAIN_END); ps=slice_stats(eq,PSEUDO,END)
          st.update(N=n,H=h,phase=ph,train_cagr=trn['cagr'],train_mdd=trn['mdd'],train_sharpe=trn['sharpe'],half1_cagr=h1['cagr'],half2_cagr=h2['cagr'],pseudo_cagr=ps['cagr'],pseudo_mdd=ps['mdd'])
          rows.append(st)
    d=pd.DataFrame(rows); d.to_csv(OUT/'concentration_all_phase.csv',index=False)
    ag=[]
    for n,g in d.groupby('N'):
        ag.append({'N':int(n),'runs':len(g),'full_cagr_median':g.cagr.median(),'full_cagr_p25':g.cagr.quantile(.25),'full_mdd_median':g.max_drawdown.median(),'full_mdd_worst':g.max_drawdown.min(),'sharpe_median':g.sharpe.median(),
                   'train_cagr_median':g.train_cagr.median(),'train_cagr_p25':g.train_cagr.quantile(.25),'train_mdd_median':g.train_mdd.median(),'train_mdd_worst':g.train_mdd.min(),'train_maximin':min(g.half1_cagr.median(),g.half2_cagr.median()),'train_calmar_robust':g.train_cagr.quantile(.25)/abs(g.train_mdd.median()),
                   'pseudo_cagr_median':g.pseudo_cagr.median(),'pseudo_cagr_p25':g.pseudo_cagr.quantile(.25),'pseudo_mdd_median':g.pseudo_mdd.median(),
                   **{f'h{h}_median':g[g.H==h].cagr.median() for h in H_ALL}})
    a=pd.DataFrame(ag).sort_values(['train_calmar_robust','train_maximin'],ascending=False); a.to_csv(OUT/'concentration_summary.csv',index=False)
    return d,a


def row_tradeable_day(r):
    return bool(np.isfinite([r.get('open',np.nan),r.get('high',np.nan),r.get('low',np.nan),r.get('volume',np.nan)]).all() and float(r['open'])>0 and float(r['volume'])>0)

def locked_limit_down(r):
    if not row_tradeable_day(r): return True
    op=float(r['open']); hi=float(r['high']); lo=float(r['low']); pc=float(r.get('prev_close',np.nan))
    one=abs(hi-lo)<1e-12 and abs(op-hi)<1e-12
    down=np.isfinite(pc) and pc>0 and op/pc-1 <= -0.045
    return bool(one and down)

def event_sell_allowed(r):
    try:return hv3.row_sell_allowed(r)
    except Exception:return bool(np.isfinite([r.exec_open,r.exec_high,r.exec_low,r.exec_volume]).all() and float(r.exec_volume)>0)

def event_buy_allowed(r):
    try:return hv3.row_buy_allowed(r)
    except Exception:return bool(np.isfinite([r.exec_open,r.exec_high,r.exec_low,r.exec_volume]).all() and float(r.exec_volume)>0)


def stop_trigger(pp:SPos,close:float,rule:str):
    if rule=='none':return False
    if not np.isfinite(close) or close<=0:return False
    ret=close/pp.entry_px-1; trail=close/pp.peak_close-1 if pp.peak_close>0 else 0
    if rule=='fixed10':return ret<=-.10
    if rule=='fixed12':return ret<=-.12
    if rule=='fixed15':return ret<=-.15
    if rule=='fixed20':return ret<=-.20
    if rule=='trail12':return trail<=-.12
    if rule=='trail18':return trail<=-.18
    if rule.startswith('hybrid12_18'):return (ret<=-.12) or (trail<=-.18)
    if rule=='atr3':
        th=float(np.clip(3*pp.atr_pct,.10,.25)) if np.isfinite(pp.atr_pct) else .15
        return ret<=-th
    raise ValueError(rule)

def cooldown_days(rule): return 20 if rule=='hybrid12_18_cd20' else 0


def choose_target(g,current,n,cooldown,td):
    x=g[np.isfinite(g.rank_test)].sort_values(['rank_test','liq20','code'],ascending=[True,False,True]).copy()
    keep=[c for c in x.loc[x.rank_test<=KEEP,'code'].tolist() if c in current][:n]
    if len(keep)<n:
        ent=[]
        for c in x.loc[x.rank_test<=ENTRY,'code'].tolist():
            if c in keep or c in current: continue
            until=cooldown.get(c,pd.Timestamp('1900-01-01'))
            if pd.Timestamp(td)<=pd.Timestamp(until): continue
            ent.append(c)
        keep.extend(ent[:n-len(keep)])
    return keep[:n]


def stop_sim(q,h,ph,n,rule,cal,members,bm,pcache,cost_mult=1.0):
    ds=chosen_dates(q,h,ph); qq=q[q.signal_date.isin(ds)].copy(); qq['signal_date']=pd.to_datetime(qq.signal_date); qq['trade_date']=pd.to_datetime(qq.trade_date)
    events={pd.Timestamp(g.trade_date.iloc[0]):g.set_index('code',drop=False) for _,g in qq.groupby('signal_date')}
    sig_for_td={pd.Timestamp(g.trade_date.iloc[0]):pd.Timestamp(g.signal_date.iloc[0]) for _,g in qq.groupby('signal_date')}
    trade_cal=cal[(cal>=START)&(cal<=END)]
    cash=sim.INITIAL_CASH; pos={}; pending={}; cooldown={}; equity=[]; trades=[]; timing=[]; turnover=0.; slip=sim.SLIPPAGE*cost_mult
    member_end=members.groupby('code').end.max().to_dict(); cd=cooldown_days(rule)

    def sell(c,day,reason,event_row=None):
        nonlocal cash,turnover
        if c not in pos:return False
        if event_row is not None:
            if not event_sell_allowed(event_row):return False
            px0=float(event_row.exec_open)
            locked=abs(float(event_row.exec_high)-float(event_row.exec_low))<1e-12 and abs(float(event_row.exec_open)-float(event_row.exec_high))<1e-12
            if locked:return False
        else:
            z=pcache.get(c)
            if day not in z.index:return False
            r=z.loc[day]
            if not row_tradeable_day(r) or locked_limit_down(r):return False
            px0=float(r['open'])
        px=px0*(1-slip); gross=pos[c].units*px; fee=sim.fee(gross,'sell',day,cost_mult); pp=pos.pop(c); cash+=gross-fee; turnover+=gross
        trades.append({'code':c,'entry_date':pp.entry_date,'exit_date':day,'net_pnl':gross-fee-pp.entry_cost,'net_return':(gross-fee)/pp.entry_cost-1,'exit_reason':reason})
        timing.append({'signal_date':sig_for_td.get(day,day-pd.Timedelta(days=1)),'trade_date':day,'side':'sell','code':c,'reason':reason})
        if reason=='stock_stop' and cd>0: cooldown[c]=day+pd.Timedelta(days=cd*2)
        pending.pop(c,None); return True

    for day in trade_cal:
        day=pd.Timestamp(day)
        # Pending stop exits are attempted at the next available open; one-price limit-down/suspension remains trapped.
        for c in sorted(list(pending)):
            sell(c,day,'stock_stop',None)
        g=events.get(day)
        if g is not None:
            current=set(pos); target=choose_target(g.reset_index(drop=True),current,n,cooldown,day); tgt=set(target)
            # rank exits before buys, but after stop attempts
            for c in sorted(list(pos)):
                if c in tgt:continue
                if c in g.index: sell(c,day,'rank_exit',g.loc[c])
                elif pd.Timestamp(member_end.get(c,END))<day:
                    pp=pos.pop(c); trades.append({'code':c,'entry_date':pp.entry_date,'exit_date':day,'net_pnl':-pp.entry_cost,'net_return':-1.0,'exit_reason':'membership_end_writeoff'})
            nav_open=cash
            for c,pp in pos.items():
                z=pcache.get(c); px=float(z.loc[day,'open']) if day in z.index and np.isfinite(z.loc[day,'open']) else pp.last_price; pp.last_price=px; nav_open+=pp.units*px
            per=nav_open*.99/n
            for c in target:
                if len(pos)>=n:break
                if c in pos or c not in g.index:continue
                r=g.loc[c]
                if not event_buy_allowed(r):continue
                locked=abs(float(r.exec_high)-float(r.exec_low))<1e-12 and abs(float(r.exec_open)-float(r.exec_high))<1e-12
                if locked or not np.isfinite(r.exec_factor) or float(r.exec_factor)<=0:continue
                factor=float(r.exec_factor); adjpx=float(r.exec_open)*(1+slip); rawpx=adjpx/factor
                rawvol=hv2.raw_share_volume(float(r.exec_volume),factor); maxraw=hv2.max_participation_shares(float(r.exec_volume),factor,sim.VOLUME_PARTICIPATION)
                shares=int(min(per,cash*.98)//(rawpx*100))*100; shares=min(shares,maxraw)
                if shares<=0:continue
                units=shares/factor; gross=units*adjpx; fee=sim.fee(gross,'buy',day,cost_mult); total=gross+fee
                if total>cash:continue
                z=pcache.get(c); atr=float(z.loc[day,'atr20_pct']) if day in z.index and 'atr20_pct' in z and np.isfinite(z.loc[day,'atr20_pct']) else np.nan
                cash-=total; pos[c]=SPos(units,total,day,float(r.exec_open),float(r.exec_open),float(r.exec_open),atr,float(r.rank_test)); turnover+=gross
                timing.append({'signal_date':sig_for_td[day],'trade_date':day,'side':'buy','code':c,'reason':'rank_entry'})
        # daily mark-to-market and close-based stop signal; execution cannot occur until a later trading day (T+1 conservative rule)
        for c,pp in list(pos.items()):
            z=pcache.get(c)
            if day in z.index:
                cl=z.loc[day,'close']
                if np.isfinite(cl) and cl>0:
                    pp.last_price=float(cl); pp.peak_close=max(pp.peak_close,float(cl))
                    if day>pp.entry_date and stop_trigger(pp,float(cl),rule): pending[c]=day
            if pd.Timestamp(member_end.get(c,END))<day and c not in pending: pending[c]=day
        nav=cash+sum(pp.units*pp.last_price for pp in pos.values()); equity.append({'signal_date':sig_for_td.get(day,pd.NaT),'trade_date':day,'equity':nav,'cash':cash,'positions':len(pos),'pending_stops':len(pending)})
    eq=pd.DataFrame(equity); tr=pd.DataFrame(trades); tm=pd.DataFrame(timing)
    st=sim.perf(eq,tr,turnover,bm.loc[START:END].dropna())
    st.update(N=n,H=h,phase=ph,stop_rule=rule,cost_mult=cost_mult,stop_exits=int((tr.exit_reason=='stock_stop').sum()) if len(tr) else 0,rank_exits=int((tr.exit_reason=='rank_exit').sum()) if len(tr) else 0)
    trn=slice_stats(eq,START,TRAIN_END); h1=slice_stats(eq,START,HALF1_END); h2=slice_stats(eq,HALF2_START,TRAIN_END); ps=slice_stats(eq,PSEUDO,END)
    st.update(train_cagr=trn['cagr'],train_mdd=trn['mdd'],train_sharpe=trn['sharpe'],half1_cagr=h1['cagr'],half2_cagr=h2['cagr'],pseudo_cagr=ps['cagr'],pseudo_mdd=ps['mdd'])
    return st,eq,tr,tm


def summarize_stop(d):
    rows=[]
    for (n,r),g in d.groupby(['N','stop_rule']):
        rows.append({'N':int(n),'stop_rule':r,'runs':len(g),'full_cagr_median':g.cagr.median(),'full_cagr_p25':g.cagr.quantile(.25),'full_cagr_min':g.cagr.min(),'mdd_median':g.max_drawdown.median(),'mdd_worst':g.max_drawdown.min(),'sharpe_median':g.sharpe.median(),
                     'train_cagr_median':g.train_cagr.median(),'train_cagr_p25':g.train_cagr.quantile(.25),'train_mdd_median':g.train_mdd.median(),'train_mdd_worst':g.train_mdd.min(),'train_maximin':min(g.half1_cagr.median(),g.half2_cagr.median()),'train_calmar_robust':g.train_cagr.quantile(.25)/abs(g.train_mdd.median()),
                     'pseudo_cagr_median':g.pseudo_cagr.median(),'pseudo_cagr_p25':g.pseudo_cagr.quantile(.25),'pseudo_mdd_median':g.pseudo_mdd.median(),'stop_exits_median':g.stop_exits.median(),
                     **{f'h{h}_median':g[g.H==h].cagr.median() for h in H_STOP}})
    return pd.DataFrame(rows).sort_values(['train_calmar_robust','train_maximin'],ascending=False)


def main():
    q,p,cal,members,ua,market_code,bm=build_candidate(); print('CANDIDATE READY',len(q),flush=True)
    conc,cs=concentration_run(q,cal,members,bm)
    ctrl=conc[(conc.N==10)&(conc.H==90)&(conc.phase==0)].iloc[0]; control_pass=bool(abs(float(ctrl.cagr)-BASE_CONTROL_H90_PHASE0)<5e-10)
    # Select two N values using TRAIN ONLY. Pseudo period is never used for selection.
    nsel=list(cs.sort_values(['train_calmar_robust','train_maximin'],ascending=False).N.astype(int).head(2)); print('TRAIN SELECTED N',nsel,flush=True)
    rules=('none','fixed10','fixed12','fixed15','fixed20','trail12','trail18','hybrid12_18','atr3','hybrid12_18_cd20')
    pcache=PriceCache(cal); rows=[]
    for n in nsel:
      for rule in rules:
       for h in H_STOP:
        for ph in range(max(1,round(h/5))):
          print('STOP N',n,rule,'H',h,'PH',ph,flush=True); st,eq,tr,tm=stop_sim(q,h,ph,n,rule,cal,members,bm,pcache,1.0); rows.append(st)
    d=pd.DataFrame(rows); d.to_csv(OUT/'stock_stop_all_phase.csv',index=False); ss=summarize_stop(d); ss.to_csv(OUT/'stock_stop_summary.csv',index=False)
    # Select finalists by train-only robust Calmar, requiring no deterioration in train maximin vs same-N no-stop >2pp.
    finalists=[]
    for n in nsel:
        z=ss[ss.N==n].copy(); b=z[z.stop_rule=='none'].iloc[0]; ok=z[(z.train_maximin>=b.train_maximin-.02)&(z.train_mdd_median>=b.train_mdd_median-.02)]
        if len(ok)==0:ok=z
        finalists += [(int(n),str(r)) for r in ok.sort_values(['train_calmar_robust','train_maximin'],ascending=False).head(2).stop_rule]
    finalists=list(dict.fromkeys(finalists))[:4]
    stress=[]
    for n,rule in finalists:
      for h in H_STOP:
       for ph in range(max(1,round(h/5))):
        st,eq,tr,tm=stop_sim(q,h,ph,n,rule,cal,members,bm,pcache,2.0); stress.append(st)
    sd=pd.DataFrame(stress); sd.to_csv(OUT/'cost2_stock_stop_all_phase.csv',index=False); ssum=summarize_stop(sd); ssum.to_csv(OUT/'cost2_stock_stop_summary.csv',index=False)
    meta={'status':'REAL_STOCK_LEVEL_PIT_CHINA_EXECUTION_RECON_NOT_ORIGINAL_EXACT','candidate':'GEff-F10QV10','control_h90_phase0':float(ctrl.cagr),'control_expected':BASE_CONTROL_H90_PHASE0,'control_pass':control_pass,'source_identity_pass':False,
          'selection_rule':'N and stock-stop finalists selected using 2016-2021 only; 2022-2026 reported as untouched pseudo-OOS validation','H_concentration':H_ALL,'H_stop':H_STOP,'Ns':NS,'selected_N_train_only':nsel,'stop_rules':rules,'cost2_finalists':finalists,
          'china_execution':['signal close -> next trading day open','stock stop triggered by prior close -> earliest next open','T+1 respected by never exiting on entry day','suspension/missing quote blocks stop exit','one-price down day <=-4.5% blocks stop exit','100-share buy lots','5% buy volume participation cap','sell before buy','stopped position leaves cash idle until next scheduled review','20-session cooldown only in explicitly named cd20 variant'],
          'fundamental_pit':'next exchange day after announcement; period-corrected quarterly cumulative fields','market_factor':market_code,'universe_audit':ua}
    (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
    print('=== CONCENTRATION ==='); print(cs.to_string(index=False),flush=True)
    print('=== STOCK STOP ==='); print(ss.to_string(index=False),flush=True)
    print('=== COST2 ==='); print(ssum.to_string(index=False),flush=True)
    print('=== META ==='); print(json.dumps(meta,ensure_ascii=False,indent=2,default=str),flush=True)

if __name__=='__main__': main()
