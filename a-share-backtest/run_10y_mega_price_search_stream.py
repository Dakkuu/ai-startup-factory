from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import run_10y_mega_price_search as m


def main():
    m.base.START=m.sim.START; m.base.WARM=m.sim.WARM; m.base.END=m.sim.END; m.base.OUT=m.OUT; m.v4.OUT=m.OUT
    cal,members,ua=m.base.load_base(); market_code,market_close,_=m.v4.pick_market(cal)
    p=m.v4.build_panel(cal,members,market_close); p=m.fq.add_factors(p,cal); p=m.sf.add_skews(p,cal,market_close); p=m.grand.add_grand_fields(p,cal,members); p=m.add_mega_fields(p,cal,market_close)
    state=m.regime_map(p,market_close); bm=market_close.loc[m.sim.START:m.sim.END].dropna()
    keep=['signal_date','trade_date','code','liq20','liq_rank_pct','exec_open','exec_high','exec_low','exec_volume','exec_factor','ivol60','eff120','skew40','rmom126','mom120raw','beta252','downbeta252','downsemivol60','max20','max60','gapvol60','range20','amtcv60','r2trend120','amount20']
    p=p[keep].copy()

    # Stage 1 streams 84 signal/filter panels; each is reused for five regime masks and then released.
    rows=[]
    for sig,flt in itertools.product(m.SIGNALS,m.FILTERS):
        qb=m.make_rank(p,sig,flt)
        for reg in m.REGIMES:
            q=m.apply_regime(qb,reg,state)
            st,_,_,_=m.run_fast(q,60,20,.10,.30,cal,members,bm); st.update({'signal':sig,'filter':flt,'regime':reg}); rows.append(st)
            print('S1',sig,flt,reg,'TRAIN',st.get('train_return'),'FULL',st.get('total_return'),flush=True)
        del qb
    s1=pd.DataFrame(rows); s1.to_csv(m.OUT/'stage1.csv',index=False)
    elig=s1[(s1.train_return>0)&(s1.train_mdd>-0.40)&(s1.positions_max<=20)].copy(); elig['train_score']=elig.train_cagr-.20*elig.train_mdd.abs(); elig=elig.sort_values(['train_score','train_cagr'],ascending=False)
    topkeys=[(str(r.signal),str(r['filter']),str(r.regime)) for _,r in elig.head(8).iterrows()]
    pd.DataFrame(topkeys,columns=['signal','filter','regime']).to_csv(m.OUT/'stage1_selected.csv',index=False)
    if not topkeys: raise RuntimeError('no stage1 candidates')

    # Rebuild only the eight selected structures.
    structures={}
    rank_cache={}
    for sig,flt,reg in topkeys:
        sfkey=(sig,flt)
        if sfkey not in rank_cache: rank_cache[sfkey]=m.make_rank(p,sig,flt)
        structures[(sig,flt,reg)]=m.apply_regime(rank_cache[sfkey],reg,state)

    rows2=[]
    for key in topkeys:
        q=structures[key]
        for n,h,(e,k) in itertools.product(m.NS,m.HOLDS,m.BUFFERS):
            st,_,_,_=m.run_fast(q,h,n,e,k,cal,members,bm); st.update({'signal':key[0],'filter':key[1],'regime':key[2]}); rows2.append(st)
    s2=pd.DataFrame(rows2); s2.to_csv(m.OUT/'stage2.csv',index=False)
    ok=s2[(s2.train_return>0)&(s2.train_mdd>-0.40)&(s2.positions_max<=s2.n_hold)].copy(); ok['train_score']=ok.train_cagr-.20*ok.train_mdd.abs(); ok=ok.sort_values(['train_score','train_cagr'],ascending=False)
    if ok.empty: raise RuntimeError('no stage2 candidates')

    exact=[]; cache={}; seen=set()
    for _,r in ok.head(m.TOP_EXACT).iterrows():
        cfg=(str(r.signal),str(r['filter']),str(r.regime),int(r.n_hold),int(r.hold_days),float(r.entry_pct),float(r.keep_pct))
        if cfg in seen: continue
        seen.add(cfg); sig,flt,reg,n,h,e,k=cfg
        st,eq,tr,tm=m.run_exact(structures[(sig,flt,reg)],h,n,e,k,cal,members,bm); st.update({'signal':sig,'filter':flt,'regime':reg}); exact.append(st); cache[cfg]=(eq,tr,tm)
        print('EXACT',cfg,'FULL',st['total_return'],'TRAIN',st['train_return'],'VAL',st['validation_return'],flush=True)
    ex=pd.DataFrame(exact); ex['train_score']=ex.train_cagr-.20*ex.train_mdd.abs(); ex=ex.sort_values(['train_score','train_cagr'],ascending=False); ex.to_csv(m.OUT/'exact_train_candidates.csv',index=False)
    win=ex.iloc[0]; wcfg=(str(win.signal),str(win['filter']),str(win.regime),int(win.n_hold),int(win.hold_days),float(win.entry_pct),float(win.keep_pct)); weq,wtr,wtm=cache[wcfg]

    annual=m.sim.annual_returns(weq); annual.to_csv(m.OUT/'winner_annual.csv',index=False)
    rob=m.sim.robustness(weq,wtr); pd.DataFrame([rob]).to_csv(m.OUT/'winner_tail.csv',index=False)
    costs=[]; q=structures[wcfg[:3]]
    for cm in (2.,4.,8.):
        st,_,_,_=m.run_exact(q,wcfg[4],wcfg[3],wcfg[5],wcfg[6],cal,members,bm,cost=cm); st.update({'cost_mult_test':cm}); costs.append(st)
    pd.DataFrame(costs).to_csv(m.OUT/'winner_costs.csv',index=False)
    s2full=s2.sort_values('total_return',ascending=False).head(20); s2full.to_csv(m.OUT/'exploratory_fullsample_top20_fast.csv',index=False)
    audit={**ua,'market_factor':market_code,'stage1_points':len(s1),'stage2_points':len(s2),'exact_points':len(ex),'selection':'ALL formal selection by 2016-2021 train score only; 2022-2026 reused validation; full-sample top exported exploratory only','winner':str(wcfg),'winner_total_return':float(win.total_return),'winner_cagr':float(win.cagr),'winner_mdd':float(win.max_drawdown),'winner_validation_return':float(win.validation_return),'five_x_target_met':int(float(win.total_return)>=4.0),'timing_violations':int((pd.to_datetime(wtm.signal_date)>=pd.to_datetime(wtm.trade_date)).sum()) if len(wtm) else 0,'streaming_memory_fix':1}
    pd.DataFrame([audit]).to_csv(m.OUT/'audit.csv',index=False)
    if audit['timing_violations']!=0: raise RuntimeError('timing violation')
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
    print('=== EXACT TRAIN SELECTED ==='); print(ex.to_string(index=False),flush=True)
    print('=== WINNER ANNUAL ==='); print(annual.to_string(index=False),flush=True)
    print('=== WINNER COSTS ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== WINNER TAIL ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== FULL SAMPLE EXPLORATORY FAST TOP20 ==='); print(s2full.to_string(index=False),flush=True)

if __name__=='__main__': main()
