from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

import run_10y_alpha_discovery_qv as qv
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_factor_mine2 as mine2
import run_10y_max_audit as ma

OUT=Path('results_alpha_ml_walkforward'); OUT.mkdir(exist_ok=True)
qv.sp.OUT=OUT
HORIZON=20
STEP=4                 # signal panel is every 5 sessions => rebalance about every 20 sessions
MATURITY_LAG=5         # >20 sessions so every training label is fully known before prediction
TRAIN_WINDOW=104       # about 2 years of signal dates
MIN_TRAIN_DATES=26
MAX_TRAIN_ROWS=60000
SEED=20260823
MODELS=('ridge1','ridge10','ridge100','lightgbm')
FEATURES=tuple(n for n,_,_ in qv.MANIFEST)


def prepare_features(p):
    q=p.copy(); liq=q.liq_rank_pct<=qv.LIQ_KEEP
    xcols=[]
    for f in FEATURES:
        c='x_'+f; xcols.append(c); q[c]=np.nan
        m=liq&np.isfinite(q[f])
        # all predictors become comparable [0,1] cross-sectional goodness ranks
        q.loc[m,c]=q.loc[m].groupby('signal_date')[f].rank(pct=True,method='average',ascending=True)
    q['y20']=np.nan
    m=liq&np.isfinite(q.fwd20)
    q.loc[m,'y20']=q.loc[m].groupby('signal_date').fwd20.rank(pct=True,method='average',ascending=True)
    return q,xcols


def fixed_fallback(g,xcols):
    wanted=['x_low_ivol','x_efficiency','x_residual_momentum','x_anti_max']
    cols=[c for c in wanted if c in xcols]
    return g[cols].mean(axis=1,skipna=True).fillna(.5).to_numpy(float)


def fit_predict(train,test,xcols,model_name,seed):
    tr=train[xcols+['y20']].copy(); tr[xcols]=tr[xcols].fillna(.5); tr=tr.dropna(subset=['y20'])
    if len(tr)>MAX_TRAIN_ROWS:
        tr=tr.sample(MAX_TRAIN_ROWS,random_state=seed,replace=False)
    X=tr[xcols].to_numpy(np.float32); y=tr.y20.to_numpy(np.float32); Xt=test[xcols].fillna(.5).to_numpy(np.float32)
    if model_name.startswith('ridge'):
        alpha=float(model_name.replace('ridge','')); model=Ridge(alpha=alpha,fit_intercept=True)
    else:
        model=LGBMRegressor(objective='regression_l2',n_estimators=120,learning_rate=.03,num_leaves=15,max_depth=-1,min_child_samples=100,subsample=.8,colsample_bytree=.8,reg_lambda=2.0,n_jobs=-1,random_state=seed,verbosity=-1)
    model.fit(X,y)
    return model.predict(Xt),len(tr)


def walkforward_scores(p,xcols,model_name):
    q=p.copy(); q['rank_test']=np.nan
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); datepos={d:i for i,d in enumerate(dates)}; chosen=list(dates[::STEP]); fit_rows=[]
    for ix,d in enumerate(chosen):
        j=datepos[d]; g=q[q.signal_date==d]
        end=j-MATURITY_LAG; start=max(0,end-TRAIN_WINDOW)
        train_dates=dates[start:max(0,end)]
        if end<=0 or len(train_dates)<MIN_TRAIN_DATES:
            pred=fixed_fallback(g,xcols); ntrain=0; mode='fixed_warmup'
        else:
            tr=q[q.signal_date.isin(train_dates)]
            try:
                pred,ntrain=fit_predict(tr,g,xcols,model_name,SEED+j); mode='model'
            except Exception as e:
                print('MODEL FALLBACK',model_name,d,repr(e),flush=True); pred=fixed_fallback(g,xcols); ntrain=0; mode='fit_failure_fallback'
        s=pd.Series(pred,index=g.index,dtype=float); finite=np.isfinite(s)
        if finite.any(): q.loc[s.index[finite],'rank_test']=s[finite].rank(pct=True,method='average',ascending=False)
        fit_rows.append({'signal_date':d,'model':model_name,'mode':mode,'train_start':train_dates[0] if len(train_dates) else pd.NaT,'train_end':train_dates[-1] if len(train_dates) else pd.NaT,'training_rows':ntrain,'maturity_lag_signal_steps':MATURITY_LAG,'feature_count':len(xcols)})
        if ix%20==0: print('WALK',model_name,ix,'/',len(chosen),d.date(),'rows',ntrain,mode,flush=True)
    return q,pd.DataFrame(fit_rows)


def exact(q,cal,members,bm,cost=1.0):
    # q already contains scores only on the pre-registered 20-session schedule; use hold=20 phase 0.
    st,eq,tr,tm=ma.run_q(q,HORIZON,0,cal,members,bm,n=20,entry=.10,keep=.30,cost=cost)
    st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31'); st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
    return st,eq,tr,tm


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=mine2.add_extra(p,cal,market_close); p=qv.add_qv_fields(p,cal); p=qv.attach_oriented_existing(p); p,xcols=prepare_features(p)
    bm=market_close.loc[sim.START:sim.END].dropna()
    pd.DataFrame([{'feature':f,'input':'cross-sectional goodness percentile','known_at':'signal close T'} for f in FEATURES]).to_csv(OUT/'feature_manifest.csv',index=False)
    rows=[]; logs=[]; cache={}
    for model_name in MODELS:
        print('MODEL',model_name,flush=True); rq,lg=walkforward_scores(p,xcols,model_name); logs.append(lg)
        st,eq,tr,tm=exact(rq,cal,members,bm); st.update({'model':model_name,'horizon':HORIZON,'train_window_signal_dates':TRAIN_WINDOW,'max_train_rows':MAX_TRAIN_ROWS,'maturity_lag_signal_steps':MATURITY_LAG}); rows.append(st); cache[model_name]=(rq,eq,tr,tm)
    grid=pd.DataFrame(rows).sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False]); grid.to_csv(OUT/'model_grid.csv',index=False); pd.concat(logs,ignore_index=True).to_csv(OUT/'fit_log.csv',index=False)
    w=grid.iloc[0]; winner=str(w.model); rq,eq,tr,tm=cache[winner]; pd.DataFrame([w]).to_csv(OUT/'winner.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=exact(rq,cal,members,bm,cm); st['model']=winner; costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'winner_costs.csv',index=False)
    a=sim.annual_returns(eq); a['model']=winner; a.to_csv(OUT/'winner_annual.csv',index=False)
    rr=sim.robustness(eq,tr); rr['model']=winner; pd.DataFrame([rr]).to_csv(OUT/'winner_robust.csv',index=False)
    allt=pd.concat([x[3].assign(model=k) for k,x in cache.items() if len(x[3])],ignore_index=True); bad=int((pd.to_datetime(allt.signal_date)>=pd.to_datetime(allt.trade_date)).sum()) if len(allt) else 0
    audit={**ua,'market_factor':market_code,'research_round':'strict rolling walk-forward ML','models':'|'.join(MODELS),'features':len(FEATURES),'target':'cross-sectional rank of forward 20-session return','maturity_lag_signal_steps':MATURITY_LAG,'train_window_signal_dates':TRAIN_WINDOW,'max_train_rows':MAX_TRAIN_ROWS,'warmup':'fixed equal lowIVOL/efficiency/residual-momentum/anti-MAX blend','selection':'reported model chosen on 2016-2021 only; each prediction fits only fully matured prior labels; 2022-2026 pseudo-OOS','signal_universe':'signal-pure T-only','volume_source_unit_shares':100,'target_500_hits':int((grid.total_return>=5.0).sum()),'timing_violations':bad}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad: raise RuntimeError('timing violation')
    print('=== MODEL GRID ==='); print(grid.to_string(index=False),flush=True)
    print('=== COSTS ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)

if __name__=='__main__': main()
