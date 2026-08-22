from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as a
import run_10y_alpha2f_v4 as v4

OUT=Path('results_signal_pure'); OUT.mkdir(exist_ok=True)


def active_mask(mm,dates):
    out=np.zeros(len(dates),dtype=bool); valid=~pd.isna(dates)
    for r in mm.itertuples(index=False): out |= valid & (dates>=r.start)&(dates<=r.end)
    return out


def build_panel(cal,members,market_close):
    """Build factor cross-sections using T and earlier information only.

    Execution fields are attached from T+1 but NEVER used to decide whether a row exists
    in the T signal cross-section. Missing/suspended next-open data remain NaN and are
    handled as failed orders by the execution engine.
    """
    trade_cal=cal[(cal>=a.START)&(cal<=a.END)]
    signal_dates=pd.DatetimeIndex(trade_cal[::5])
    alltrade=cal[cal<=a.END]
    exec_dates=[]
    for s in signal_dates:
        k=alltrade.searchsorted(s,side='right')
        exec_dates.append(alltrade[k] if k<len(alltrade) else pd.NaT)
    exec_dates=pd.DatetimeIndex(exec_dates)

    bm_ret=market_close.reindex(cal[(cal>=a.WARM)&(cal<=a.END)]).pct_change(fill_method=None)
    bm_mu=bm_ret.rolling(a.BETA_LOOKBACK,min_periods=a.MIN_BETA_OBS).mean().shift(1)
    bm_var=bm_ret.rolling(a.BETA_LOOKBACK,min_periods=a.MIN_BETA_OBS).var().shift(1)
    factor_cols=[f'rmom{w}' for w in a.RMOM_WINDOWS]+[f'ivol{w}' for w in a.IVOL_WINDOWS]
    frames=[]; codes=sorted(members.code.unique())
    rows_with_missing_exec=0
    for i,code in enumerate(codes,1):
        mm=members[members.code==code]; cols={}
        for f in ['open','high','low','close','volume','factor']:
            s=base.qb.read_bin(code,f,cal)
            if not s.empty: cols[f]=s
        if not all(f in cols for f in ['open','high','low','close','volume']): continue
        z=pd.concat(cols,axis=1).loc[a.WARM:a.END].copy()
        if z.empty: continue
        if 'factor' not in z: z['factor']=1.0
        # Preserve missing factor on suspended days for execution; only fill non-suspended
        # historical factor gaps when a price is present.
        z.loc[z.factor==0,'factor']=np.nan
        r=z.close.pct_change(fill_method=None)
        count120=z.close.notna().rolling(a.MIN_LIST_DAYS).sum()
        liq20=(z.close.abs()*z.volume.abs()).rolling(20).mean()
        m=bm_ret.reindex(z.index)
        smu=r.rolling(a.BETA_LOOKBACK,min_periods=a.MIN_BETA_OBS).mean().shift(1)
        cov=r.rolling(a.BETA_LOOKBACK,min_periods=a.MIN_BETA_OBS).cov(m).shift(1)
        beta=cov/bm_var.reindex(z.index); alpha=smu-beta*bm_mu.reindex(z.index); resid=r-alpha-beta*m
        fac={}
        for w in a.RMOM_WINDOWS:
            n=w-a.SKIP_RECENT+1
            fac[f'rmom{w}']=resid.shift(a.SKIP_RECENT).rolling(n,min_periods=max(30,int(n*.8))).sum()
        for w in a.IVOL_WINDOWS:
            fac[f'ivol{w}']=resid.rolling(w,min_periods=max(25,int(w*.8))).std()

        sig={'count120':count120.reindex(signal_dates).to_numpy(),'liq20':liq20.reindex(signal_dates).to_numpy()}
        for k,s in fac.items(): sig[k]=s.reindex(signal_dates).to_numpy()
        sig=pd.DataFrame(sig)
        ex=z.reindex(exec_dates).reset_index(drop=True)

        # CRITICAL: signal eligibility uses signal-date information only.
        active_s=active_mask(mm,signal_dates)
        valid=active_s & (~pd.isna(signal_dates))
        valid &= np.asarray(sig.count120>=a.MIN_LIST_DAYS)
        valid &= np.isfinite(sig[['liq20']].to_numpy()).all(axis=1)
        if not valid.any(): continue
        idx=np.flatnonzero(valid)

        rec={'signal_date':signal_dates[idx],'trade_date':exec_dates[idx],'code':code,
             'liq20':sig.liq20.to_numpy()[idx].astype(float)}
        for c in factor_cols: rec[c]=sig[c].to_numpy()[idx].astype(float)
        # T+1 fields may be NaN. Do not filter them here.
        for csrc,cdst in [('open','exec_open'),('high','exec_high'),('low','exec_low'),('volume','exec_volume'),('factor','exec_factor')]:
            if csrc in ex:
                rec[cdst]=ex[csrc].to_numpy()[idx].astype(float)
            else:
                rec[cdst]=np.full(len(idx),np.nan)
        fr=pd.DataFrame(rec)
        rows_with_missing_exec += int((~np.isfinite(fr[['exec_open','exec_high','exec_low','exec_volume']]).all(axis=1)).sum())
        frames.append(fr)
        if i%500==0: print('signal-pure histories',i,'/',len(codes),flush=True)
    if not frames: raise RuntimeError('no signal-pure panel')
    p=pd.concat(frames,ignore_index=True)
    if not (pd.to_datetime(p.signal_date)<pd.to_datetime(p.trade_date)).all(): raise RuntimeError('signal/trade date construction violation')

    p['liq_rank_pct']=p.groupby('signal_date').liq20.rank(pct=True,method='average',ascending=False)
    liq_ok=p.liq_rank_pct<=a.LIQ_KEEP_PCT
    for w in a.RMOM_WINDOWS:
        col=f'rmom{w}'; out=f'rmom{w}_pct'; p[out]=np.nan; mask=liq_ok&p[col].notna()
        p.loc[mask,out]=p.loc[mask].groupby('signal_date')[col].rank(pct=True,method='average',ascending=False)
    for w in a.IVOL_WINDOWS:
        col=f'ivol{w}'; out=f'ivol{w}_pct'; p[out]=np.nan; mask=liq_ok&p[col].notna()
        p.loc[mask,out]=p.loc[mask].groupby('signal_date')[col].rank(pct=True,method='average',ascending=True)
    for rw in a.RMOM_WINDOWS:
        for vw in a.IVOL_WINDOWS:
            rc=f'rmom{rw}_pct'; vc=f'ivol{vw}_pct'; sc=f'score_{rw}_{vw}'; pc=sc+'_pct'
            p[sc]=np.where(p[rc].notna()&p[vc].notna(),.5*(1-p[rc])+.5*(1-p[vc]),np.nan)
            p[pc]=np.nan; mask=p[sc].notna(); p.loc[mask,pc]=p.loc[mask].groupby('signal_date')[sc].rank(pct=True,method='average',ascending=False)
    p['core_eligible']=p[f'rmom{a.CORE_RMOM}_pct'].notna()&p[f'ivol{a.CORE_IVOL}_pct'].notna()
    core=p[p.core_eligible]; gs=core.groupby('signal_date').size(); full=p.groupby('signal_date').size()
    audit={'rows':len(p),'signal_dates':int(p.signal_date.nunique()),'full_min':int(full.min()),'full_median':float(full.median()),'full_max':int(full.max()),'core_min':int(gs.min()),'core_median':float(gs.median()),'core_max':int(gs.max()),'rows_missing_next_exec':rows_with_missing_exec,'missing_next_exec_share':float(rows_with_missing_exec/len(p))}
    pd.DataFrame([audit]).to_csv(OUT/'signal_pure_panel_audit.csv',index=False)
    print('SIGNAL PURE PANEL',audit,flush=True)
    if core.signal_date.nunique()<480 or gs.median()<1500 or gs.min()<500: raise RuntimeError(f'FAIL-CLOSED signal-pure core {audit}')
    # Positive-control audit: there should normally be at least some T-eligible rows with no T+1 quote.
    # If zero, we cannot demonstrate that future tradability is no longer conditioning the universe.
    if rows_with_missing_exec<=0: raise RuntimeError('FAIL-CLOSED no missing-next-exec rows retained; lookahead-removal audit inconclusive')
    return p


def patch():
    v4.build_panel=build_panel
    return v4

if __name__=='__main__':
    print('signal-pure panel module; import patch() before audits')
