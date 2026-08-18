from __future__ import annotations

import numpy as np
import pandas as pd
import run_10y_era_backtest as base


def _active_mask(mm: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    out=np.zeros(len(dates),dtype=bool)
    valid=~pd.isna(dates)
    for r in mm.itertuples(index=False):
        out |= valid & (dates>=r.start) & (dates<=r.end)
    return out


def build_weekly_panel_fast(cal,m):
    test=cal[(cal>=base.START)&(cal<=base.END)]
    signal_dates=pd.DatetimeIndex(test[::5])
    alltrade=cal[cal<=base.END]
    exec_dates=[]
    for s in signal_dates:
        k=alltrade.searchsorted(s,side='right')
        exec_dates.append(alltrade[k] if k<len(alltrade) else pd.NaT)
    exec_dates=pd.DatetimeIndex(exec_dates)
    next_exec_dates=pd.DatetimeIndex(list(exec_dates[1:])+[pd.NaT])
    frames=[]
    needed=['ret1','mom5','mom20','mom60','mom120','vol20','vol60','vol_ratio','liq_ma20','ma20gap','ma60gap','high20']
    codes=sorted(m.code.unique())
    for i,code in enumerate(codes,1):
        cols={}
        for f in ['open','high','low','close','volume','factor']:
            s=base.qb.read_bin(code,f,cal)
            if not s.empty: cols[f]=s
        if not all(f in cols for f in ['open','high','low','close','volume']): continue
        z=pd.concat(cols,axis=1).loc[base.WARM:base.END].copy()
        if z.empty: continue
        if 'factor' not in z: z['factor']=1.0
        z['factor']=z.factor.replace(0,np.nan).fillna(1.0)
        r1=z.close.pct_change()
        z['ret1']=r1; z['mom5']=z.close.pct_change(5); z['mom20']=z.close.pct_change(20); z['mom60']=z.close.pct_change(60); z['mom120']=z.close.pct_change(120)
        z['vol20']=r1.rolling(20).std(); z['vol60']=r1.rolling(60).std()
        z['vol_ma20']=z.volume.rolling(20).mean(); z['vol_ratio']=z.volume/z.vol_ma20
        z['liq_ma20']=(z.close.abs()*z.volume.abs()).rolling(20).mean()
        z['ma20gap']=z.close/z.close.rolling(20).mean()-1; z['ma60gap']=z.close/z.close.rolling(60).mean()-1
        z['high20']=z.close/z.high.shift(1).rolling(20).max()-1
        z['drawdown120']=z.close/z.close.rolling(120).max()-1
        mm=m[m.code==code]
        active_s=_active_mask(mm,signal_dates); active_e=_active_mask(mm,exec_dates); active_ne=_active_mask(mm,next_exec_dates)
        sig=z.reindex(signal_dates)
        ex=z.reindex(exec_dates).reset_index(drop=True)
        nex=z.reindex(next_exec_dates).reset_index(drop=True)
        valid=active_s & active_e & (~pd.isna(exec_dates))
        valid &= np.isfinite(sig[needed].to_numpy()).all(axis=1)
        valid &= np.isfinite(ex[['open','high','low','close','volume']].to_numpy()).all(axis=1)
        if not valid.any(): continue
        idx=np.flatnonzero(valid)
        rec=pd.DataFrame({
            'signal_date':signal_dates[idx], 'trade_date':exec_dates[idx], 'code':code,
            **{x:sig[x].to_numpy()[idx].astype(float) for x in needed},
            'drawdown120':sig['drawdown120'].to_numpy()[idx].astype(float),
            'signal_close':sig['close'].to_numpy()[idx].astype(float),
            'exec_open':ex['open'].to_numpy()[idx].astype(float),
            'exec_high':ex['high'].to_numpy()[idx].astype(float),
            'exec_low':ex['low'].to_numpy()[idx].astype(float),
            'exec_close':ex['close'].to_numpy()[idx].astype(float),
            'exec_volume':ex['volume'].to_numpy()[idx].astype(float),
            'exec_factor':ex['factor'].to_numpy()[idx].astype(float),
        })
        label=np.full(len(idx),np.nan,dtype=float)
        ne_open=nex['open'].to_numpy()
        e_open=ex['open'].to_numpy()
        good_label=active_ne[idx] & np.isfinite(ne_open[idx]) & np.isfinite(e_open[idx]) & (e_open[idx]>0)
        label[good_label]=ne_open[idx][good_label]/e_open[idx][good_label]-1
        rec['label']=label
        rec['label_exit_date']=next_exec_dates[idx]
        frames.append(rec)
        if i%500==0: print('vector weekly features',i,'/',len(codes),flush=True)
    if not frames: raise RuntimeError('no weekly features')
    p=pd.concat(frames,ignore_index=True)
    p=p[(p.trade_date<=base.END)&(p.signal_date>=base.START)].copy()
    rank_map={
        'r_ret1':('ret1',True),'r_mom5':('mom5',True),'r_mom20':('mom20',True),'r_mom60':('mom60',True),'r_mom120':('mom120',True),
        'r_lowvol20':('vol20',False),'r_lowvol60':('vol60',False),'r_volratio':('vol_ratio',True),'r_liq':('liq_ma20',True),
        'r_ma20gap':('ma20gap',True),'r_ma60gap':('ma60gap',True),'r_high20':('high20',True),
    }
    for outcol,(incol,ascending) in rank_map.items():
        p[outcol]=p.groupby('signal_date')[incol].rank(pct=True,method='average',ascending=ascending)
    # fail closed: signal date must precede trade date, labels must mature after trade date.
    if not (pd.to_datetime(p.signal_date)<pd.to_datetime(p.trade_date)).all(): raise RuntimeError('panel signal/trade timing violation')
    lab=p[p.label.notna()]
    if len(lab) and not (pd.to_datetime(lab.trade_date)<pd.to_datetime(lab.label_exit_date)).all(): raise RuntimeError('panel label timing violation')
    p.to_pickle(base.OUT/'weekly_panel.pkl')
    print('panel rows',len(p),'signal dates',p.signal_date.nunique(),flush=True)
    return p,signal_dates

base.build_weekly_panel=build_weekly_panel_fast
base.main()
