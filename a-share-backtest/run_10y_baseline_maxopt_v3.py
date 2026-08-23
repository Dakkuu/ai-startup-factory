from __future__ import annotations
from pathlib import Path
import argparse, math, json
import numpy as np
import pandas as pd

# Install correctness patches before panel/executor use.
import run_10y_hard_executor_v2 as hv2
import run_10y_signal_pure_panel as sp
hv2.patch(); sp.patch()

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_factor_mine2 as mine2
import run_10y_grand_opt as grand
import run_10y_max_audit as ma

START=pd.Timestamp('2016-07-29')
TRAIN_END=pd.Timestamp('2021-12-31')
HALF1_END=pd.Timestamp('2019-12-31')
HALF2_START=pd.Timestamp('2020-01-01')
PSEUDO_START=pd.Timestamp('2022-01-01')
END=pd.Timestamp('2026-07-29')


def build_panel(out: Path, need_fwd=False):
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=out; v4.OUT=out; sp.OUT=out
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close)
    p=fq.add_factors(p,cal)
    p=sf.add_skews(p,cal,market_close)
    p=mine2.add_extra(p,cal,market_close)
    p=grand.add_grand_fields(p,cal,members)
    if need_fwd:
        p['fwd60']=np.nan
        groups=p.groupby('code').groups
        for i,(code,idx) in enumerate(groups.items(),1):
            c=base.qb.read_bin(code,'close',cal).loc[sim.WARM:sim.END]
            if c.empty: continue
            f=c.shift(-60)/c-1
            ds=pd.DatetimeIndex(p.loc[idx,'signal_date'])
            p.loc[idx,'fwd60']=f.reindex(ds).to_numpy(float)
            if i%1000==0: print('forward histories',i,'/',len(groups),flush=True)
    bm=market_close.loc[sim.START:sim.END].dropna()
    return p,cal,members,ua,market_code,bm


def pct_rank(q,m,col,ascending=True):
    return q.loc[m].groupby('signal_date')[col].rank(pct=True,method='average',ascending=ascending)


def eligible_mask(q,liq=.70,skew_keep=.80,age=0,exclude_growth=False):
    m=(q.liq_rank_pct<=liq)&np.isfinite(q.ivol60)&np.isfinite(q.eff120)&np.isfinite(q.skew40)
    if age>0: m &= q.age_days>=age
    if exclude_growth: m &= ~q.board.isin(['STAR','BJ'])
    if skew_keep<1:
        sm=m.copy(); sr=pct_rank(q,sm,'skew40',True); ok=pd.Series(False,index=q.index); ok.loc[sr.index]=sr<=skew_keep; m &= ok
    return m


def component_ranks(q,m):
    out={
      'iv':pct_rank(q,m,'ivol60',True),
      'ef':pct_rank(q,m,'eff120',False),
    }
    extras={
      'down':('dsemi60',True),'amax':('max20',True),'askew':('skew60',True),
      'rmom':('rmom126',False),'tstat':('tstat120',False),'dd':('dd120',False),
      'beta':('beta252',True),'capture':('capture120',False),'mom':('mom120',False),
      'volshock':('volshock',True),
    }
    for k,(c,a) in extras.items():
        if c in q.columns and np.isfinite(q.loc[m,c]).any(): out[k]=pct_rank(q,m,c,a)
    return out


def rerank(q,spec):
    x=q.copy(); x['rank_test']=np.nan
    m=eligible_mask(x,float(spec.get('liq',.70)),float(spec.get('skew',.80)),int(spec.get('age',0)),bool(spec.get('exgrowth',False)))
    req=spec.get('requires',[])
    for c in req: m &= np.isfinite(x[c])
    if not m.any(): return x
    R=component_ranks(x,m); kind=spec['kind']
    if kind=='linear':
        w=float(spec['w']); raw=w*R['iv']+(1-w)*R['ef']
    elif kind=='weighted':
        weights=spec['weights']; raw=None; sw=0.0
        for k,w in weights.items():
            if k not in R: continue
            raw=R[k]*float(w) if raw is None else raw+R[k]*float(w); sw+=float(w)
        raw=raw/sw
    elif kind=='power':
        pwr=float(spec['p']); a=R['iv']; b=R['ef']; raw=((a.pow(pwr)+b.pow(pwr))/2.0).pow(1.0/pwr)
    elif kind=='bottleneck': raw=pd.concat([R['iv'],R['ef']],axis=1).max(axis=1)
    elif kind=='meanmax': raw=.5*(R['iv']+R['ef'])/2+.5*pd.concat([R['iv'],R['ef']],axis=1).max(axis=1)
    elif kind=='product': raw=np.sqrt((R['iv']*R['ef']).clip(lower=1e-12))
    elif kind=='risk_bottleneck': raw=pd.concat([R['iv'],R['ef'],R['down'],R['amax']],axis=1).max(axis=1)
    elif kind=='risk_meanmax':
        z=pd.concat([R['iv'],R['ef'],R['down'],R['amax']],axis=1); raw=.5*z.mean(axis=1)+.5*z.max(axis=1)
    elif kind=='gate':
        ivq=float(spec.get('ivq',.5)); efq=float(spec.get('efq',.5)); keep=(R['iv']<=ivq)&(R['ef']<=efq)
        m2=pd.Series(False,index=x.index); m2.loc[keep.index]=keep; m &= m2
        R=component_ranks(x,m); raw=float(spec.get('w',.6))*R['iv']+(1-float(spec.get('w',.6)))*R['ef']
    else: raise ValueError(kind)
    x.loc[m,'rank_test']=raw.groupby(x.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return x


def signal_diag(q):
    z=q[(pd.to_datetime(q.signal_date)>=START)&(pd.to_datetime(q.signal_date)<=pd.Timestamp('2021-09-30'))][['signal_date','rank_test','fwd60']].dropna().copy()
    ics=[]; spreads=[]
    for d,g in z.groupby('signal_date',sort=True):
        if len(g)<300: continue
        a=-g.rank_test; ic=a.corr(g.fwd60,method='spearman')
        if np.isfinite(ic): ics.append((pd.Timestamp(d),float(ic)))
        rr=g.rank_test.rank(pct=True,method='average')
        hi=g.loc[rr<=.10,'fwd60']; lo=g.loc[rr>=.90,'fwd60']
        if len(hi)>20 and len(lo)>20: spreads.append(float(hi.mean()-lo.mean()))
    if not ics: return {'n_dates':0,'mean_ic':np.nan,'ic_t':np.nan,'positive_ic_share':np.nan,'positive_years':0,'years':0,'top_bottom_spread':np.nan}
    d=pd.DataFrame(ics,columns=['date','ic']); sd=d.ic.std(ddof=1); t=d.ic.mean()/sd*np.sqrt(len(d)) if sd>0 else np.nan
    ann=d.assign(year=d.date.dt.year).groupby('year').ic.mean();
    return {'n_dates':len(d),'mean_ic':float(d.ic.mean()),'ic_t':float(t),'positive_ic_share':float((d.ic>0).mean()),'positive_years':int((ann>0).sum()),'years':int(len(ann)),'top_bottom_spread':float(np.mean(spreads)) if spreads else np.nan}


def period_return(eq,a,b):
    z=eq[(pd.to_datetime(eq.trade_date)>=pd.Timestamp(a))&(pd.to_datetime(eq.trade_date)<=pd.Timestamp(b))]
    if len(z)<2:return np.nan
    return float(z.equity.iloc[-1]/z.equity.iloc[0]-1)


def train_run(q,hold,n,e,k,cal,members,bm):
    st,eq,tr,tm=ma.run_q(q,int(hold),0,cal,members,bm,n=int(n),entry=float(e),keep=float(k),start=START,end=TRAIN_END)
    st['half1_return']=period_return(eq,START,HALF1_END); st['half2_return']=period_return(eq,HALF2_START,TRAIN_END)
    st['min_half_return']=min(st['half1_return'],st['half2_return']) if np.isfinite(st['half1_return']) and np.isfinite(st['half2_return']) else np.nan
    return st,eq,tr,tm


def full_run(q,hold,n,e,k,cal,members,bm,cost=1.0,phase=0,initial_cash=1e6,vol_part=.05):
    st,eq,tr,tm=ma.run_q(q,int(hold),int(phase),cal,members,bm,n=int(n),entry=float(e),keep=float(k),cost=float(cost),initial_cash=float(initial_cash),vol_part=float(vol_part))
    st['train_2016_2021_return']=period_return(eq,START,TRAIN_END)
    st['pseudo_oos_2022_2026_return']=period_return(eq,PSEUDO_START,END)
    st['half1_return']=period_return(eq,START,HALF1_END); st['half2_return']=period_return(eq,HALF2_START,TRAIN_END)
    return st,eq,tr,tm


def robust_filter(df):
    z=df[(df.half1_return>0)&(df.half2_return>0)&(df.max_drawdown>-0.40)&(df.positions_max<=df.n_hold)].copy()
    return z if len(z) else df.copy()


def choose_two(df):
    z=robust_filter(df)
    growth=z.sort_values(['total_return','max_drawdown'],ascending=[False,False]).iloc[0]
    robust=z.sort_values(['min_half_return','total_return','max_drawdown'],ascending=[False,False,False]).iloc[0]
    rows=[growth]
    if str(robust.get('key'))!=str(growth.get('key')): rows.append(robust)
    return pd.DataFrame(rows)


def spec_id(s):
    return s.get('name') or json.dumps(s,sort_keys=True,separators=(',',':'))


def linear_specs():
    out=[]
    for liq in (.55,.65,.70,.80):
      for skew in (.65,.80,.90):
       for w in (.40,.50,.60,.70,.80):
        out.append({'name':f'lin_l{liq:.2f}_s{skew:.2f}_w{w:.2f}','kind':'linear','liq':liq,'skew':skew,'w':w})
    for age,exg in ((365,False),(730,False),(365,True)):
      for w in (.50,.60,.70): out.append({'name':f'lin_age{age}_ex{int(exg)}_w{w:.2f}','kind':'linear','liq':.70,'skew':.80,'w':w,'age':age,'exgrowth':exg})
    return out


def augmented_specs():
    formulas={
      'iv_eff_down':{'iv':.40,'ef':.30,'down':.30},
      'iv_eff_amax':{'iv':.45,'ef':.30,'amax':.25},
      'iv_eff_rmom':{'iv':.45,'ef':.35,'rmom':.20},
      'risk_lottery':{'iv':.30,'ef':.25,'down':.25,'amax':.20},
      'risk_momentum':{'iv':.30,'ef':.25,'down':.25,'rmom':.20},
      'lottery_momentum':{'iv':.30,'ef':.25,'amax':.20,'askew':.10,'rmom':.15},
      'trend_quality':{'iv':.40,'ef':.25,'tstat':.20,'rmom':.15},
      'drawdown_quality':{'iv':.40,'ef':.25,'down':.20,'dd':.15},
      'lowbeta_quality':{'iv':.35,'ef':.25,'down':.20,'beta':.20},
      'capture_quality':{'iv':.40,'ef':.25,'down':.15,'capture':.20},
    }
    out=[]
    for nm,w in formulas.items():
      req=[]
      mp={'down':'dsemi60','amax':'max20','askew':'skew60','rmom':'rmom126','tstat':'tstat120','dd':'dd120','beta':'beta252','capture':'capture120'}
      for k in w:
        if k in mp:req.append(mp[k])
      for liq,sk in ((.60,.80),(.70,.70),(.70,.80),(.70,.90),(.80,.80)):
        out.append({'name':f'{nm}_l{liq:.2f}_s{sk:.2f}','kind':'weighted','weights':w,'liq':liq,'skew':sk,'requires':req})
    return out


def nonlinear_specs():
    out=[]
    kinds=[('power2',{'kind':'power','p':2}),('power4',{'kind':'power','p':4}),('bottleneck',{'kind':'bottleneck'}),('meanmax',{'kind':'meanmax'}),('product',{'kind':'product'}),('risk_bottleneck',{'kind':'risk_bottleneck','requires':['dsemi60','max20']}),('risk_meanmax',{'kind':'risk_meanmax','requires':['dsemi60','max20']})]
    for nm,base_s in kinds:
      for liq,sk in ((.60,.80),(.70,.70),(.70,.80),(.70,.90),(.80,.80)):
        s={'name':f'{nm}_l{liq:.2f}_s{sk:.2f}','liq':liq,'skew':sk,**base_s}; out.append(s)
    for ivq,efq in ((.35,.35),(.45,.45),(.55,.40),(.40,.55),(.60,.60)):
      out.append({'name':f'gate_iv{ivq:.2f}_ef{efq:.2f}','kind':'gate','liq':.70,'skew':.80,'w':.60,'ivq':ivq,'efq':efq})
    return out


def run_signal_lane(lane):
    out=Path(f'results_baseline_maxopt_{lane}'); out.mkdir(exist_ok=True)
    p,cal,members,ua,market_code,bm=build_panel(out,need_fwd=True)
    specs={'linear':linear_specs,'augmented':augmented_specs,'nonlinear':nonlinear_specs}[lane]()
    screens=[]; qcache={}
    for s in specs:
        print('SCREEN',s['name'],flush=True); q=rerank(p,s); d=signal_diag(q); d.update({'signal':s['name'],'spec':json.dumps(s,sort_keys=True)}); screens.append(d); qcache[s['name']]=q
    sc=pd.DataFrame(screens); sc['pass_gate']=(sc.n_dates>=200)&(sc.mean_ic>0)&(sc.ic_t>=2)&(sc.top_bottom_spread>0)&(sc.positive_years>=4)
    sc=sc.sort_values(['pass_gate','ic_t','mean_ic'],ascending=[False,False,False]); sc.to_csv(out/'signal_screen.csv',index=False)
    shortlist=sc[sc.pass_gate].head(6)
    if len(shortlist)<4: shortlist=sc.head(6)
    anchors=[]
    for r in shortlist.itertuples(index=False):
        q=qcache[str(r.signal)]; st,eq,tr,tm=train_run(q,60,20,.10,.30,cal,members,bm); st.update({'signal':str(r.signal),'key':str(r.signal),'ic_t':float(r.ic_t),'mean_ic':float(r.mean_ic)}); anchors.append(st)
    adf=pd.DataFrame(anchors); adf.to_csv(out/'anchor_train.csv',index=False)
    seed=choose_two(adf)
    grid=[]; qkeep={}
    for rr in seed.itertuples(index=False):
        name=str(rr.signal); q=qcache[name]; qkeep[name]=q
        for n in (10,15,20,30):
          for h in (40,60,80,120):
           for e,k in ((.05,.20),(.10,.30)):
            print('TRAIN GRID',name,n,h,e,k,flush=True); st,eq,tr,tm=train_run(q,h,n,e,k,cal,members,bm); key=f'{name}|n{n}|h{h}|e{e}|k{k}'; st.update({'signal':name,'n_hold':n,'hold_days':h,'entry_pct':e,'keep_pct':k,'key':key}); grid.append(st)
    g=pd.DataFrame(grid); g.to_csv(out/'construction_train.csv',index=False)
    finalists=choose_two(g); finalists.to_csv(out/'train_selected.csv',index=False)
    full=[]; cache={}
    for r in finalists.itertuples(index=False):
        q=qkeep[str(r.signal)]; st,eq,tr,tm=full_run(q,r.hold_days,r.n_hold,r.entry_pct,r.keep_pct,cal,members,bm); st.update({'signal':str(r.signal),'key':str(r.key)}); full.append(st); cache[str(r.key)]=(q,eq,tr,tm,r)
    f=pd.DataFrame(full); f.to_csv(out/'finalists_full.csv',index=False)
    # Predeclared robust winner = maximin winner from train_selected, not pseudo-OOS.
    wr=finalists.sort_values(['min_half_return','total_return'],ascending=[False,False]).iloc[0]; wk=str(wr.key); q,eq,tr,tm,_=cache[wk]
    ann=sim.annual_returns(eq); ann.to_csv(out/'winner_annual.csv',index=False)
    pd.DataFrame([sim.robustness(eq,tr)]).to_csv(out/'winner_tail.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=full_run(q,wr.hold_days,wr.n_hold,wr.entry_pct,wr.keep_pct,cal,members,bm,cost=cm); st['cost_mult_test']=cm; costs.append(st)
    pd.DataFrame(costs).to_csv(out/'winner_costs.csv',index=False)
    phases=[]; step=max(1,round(float(wr.hold_days)/5))
    for ph in range(step):
        st,_,_,_=full_run(q,wr.hold_days,wr.n_hold,wr.entry_pct,wr.keep_pct,cal,members,bm,phase=ph); st['phase']=ph; phases.append(st)
    pd.DataFrame(phases).to_csv(out/'winner_phases.csv',index=False)
    bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    audit={**ua,'market_factor':market_code,'lane':lane,'signal_candidates':len(sc),'construction_points':len(g),'selection':'signal IC and construction use 2016-2021 only; maximin 2016-2019 vs 2020-2021; 2022-2026 pseudo-OOS only','signal_universe':'T-only signal-pure','volume_unit_shares':100,'timing_violations':bad}
    pd.DataFrame([audit]).to_csv(out/'audit.csv',index=False)
    print('=== SIGNAL TOP ==='); print(sc.head(20).to_string(index=False),flush=True)
    print('=== TRAIN SELECTED ==='); print(finalists.to_string(index=False),flush=True)
    print('=== FULL ==='); print(f.to_string(index=False),flush=True)
    print('=== PHASES ==='); print(pd.DataFrame(phases).to_string(index=False),flush=True)
    if bad: raise RuntimeError('timing violation')


def baseline_spec(w=.60): return {'name':f'baseline_w{w:.2f}','kind':'linear','liq':.70,'skew':.80,'w':w}


def run_surface(shard):
    out=Path(f'results_baseline_maxopt_surface_{shard}'); out.mkdir(exist_ok=True)
    p,cal,members,ua,market_code,bm=build_panel(out,need_fwd=False)
    weights={'short':(.50,.60,.70),'mid':(.45,.55,.60,.65,.75),'long':(.50,.60,.70)}[shard]
    holds={'short':(20,30,40,50),'mid':(60,70,80,100),'long':(120,140,160)}[shard]
    ns={'short':(8,12,16,20,25,30),'mid':(10,15,20,25,30,40),'long':(10,15,20,25,30)}[shard]
    rows=[]; qs={}
    for w in weights:
        q=rerank(p,baseline_spec(w)); qs[w]=q
        for n in ns:
          for h in holds:
           for e,k in ((.05,.20),(.10,.30),(.15,.40)):
            print('SURFACE TRAIN',shard,w,n,h,e,k,flush=True); st,eq,tr,tm=train_run(q,h,n,e,k,cal,members,bm); key=f'w{w}|n{n}|h{h}|e{e}|k{k}'; st.update({'ivol_weight':w,'n_hold':n,'hold_days':h,'entry_pct':e,'keep_pct':k,'key':key}); rows.append(st)
    g=pd.DataFrame(rows); g.to_csv(out/'surface_train.csv',index=False); finalists=choose_two(g); finalists.to_csv(out/'train_selected.csv',index=False)
    full=[]; cache={}
    for r in finalists.itertuples(index=False):
        q=qs[float(r.ivol_weight)]; st,eq,tr,tm=full_run(q,r.hold_days,r.n_hold,r.entry_pct,r.keep_pct,cal,members,bm); st.update({'ivol_weight':float(r.ivol_weight),'key':str(r.key)}); full.append(st); cache[str(r.key)]=(q,eq,tr,tm,r)
    f=pd.DataFrame(full); f.to_csv(out/'finalists_full.csv',index=False)
    wr=finalists.sort_values(['min_half_return','total_return'],ascending=[False,False]).iloc[0]; q,eq,tr,tm,_=cache[str(wr.key)]
    sim.annual_returns(eq).to_csv(out/'winner_annual.csv',index=False); pd.DataFrame([sim.robustness(eq,tr)]).to_csv(out/'winner_tail.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=full_run(q,wr.hold_days,wr.n_hold,wr.entry_pct,wr.keep_pct,cal,members,bm,cost=cm); st['cost_mult_test']=cm; costs.append(st)
    pd.DataFrame(costs).to_csv(out/'winner_costs.csv',index=False)
    bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    audit={**ua,'market_factor':market_code,'lane':f'surface_{shard}','points':len(g),'selection':'2016-2021 only; maximin halves plus train growth finalist; 2022-2026 pseudo-OOS only','signal_universe':'T-only signal-pure','volume_unit_shares':100,'timing_violations':bad}
    pd.DataFrame([audit]).to_csv(out/'audit.csv',index=False)
    print('=== SURFACE TOP ROBUST ==='); print(robust_filter(g).sort_values(['min_half_return','total_return'],ascending=[False,False]).head(30).to_string(index=False),flush=True)
    print('=== FULL ==='); print(f.to_string(index=False),flush=True)
    if bad: raise RuntimeError('timing violation')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('lane',choices=('linear','augmented','nonlinear','surface_short','surface_mid','surface_long')); args=ap.parse_args()
    if args.lane.startswith('surface_'): run_surface(args.lane.split('_',1)[1])
    else: run_signal_lane(args.lane)
