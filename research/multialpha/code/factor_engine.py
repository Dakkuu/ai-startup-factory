import numpy as np
import pandas as pd

EPS=1e-12

def _comp_ret(x, w):
    return (1+x).rolling(w,min_periods=w).apply(np.prod,raw=True)-1

def _roll_slope_rsq(logp, w):
    x=np.arange(w,dtype=float); x=x-x.mean(); den=(x*x).sum()
    def slope(a):
        y=a-a.mean(); return float((x*y).sum()/den)
    def rsq(a):
        y=a-a.mean(); ss=(y*y).sum()
        if ss<=0: return np.nan
        b=(x*y).sum()/den; fit=b*x
        return float((fit*fit).sum()/ss)
    return logp.rolling(w,min_periods=w).apply(slope,raw=True), logp.rolling(w,min_periods=w).apply(rsq,raw=True)

def compute_factors(bars: pd.DataFrame) -> pd.DataFrame:
    df=bars.copy().sort_values(['ts_code','trade_date']).reset_index(drop=True)
    df['trade_date']=pd.to_datetime(df.trade_date)
    for c in ['open','high','low','close','volume']:
        if c not in df: raise ValueError(f'missing {c}')
    if 'amount' not in df: df['amount']=df['close']*df['volume']
    if 'turnover_rate' not in df: df['turnover_rate']=np.nan
    if 'industry' not in df: df['industry']='UNKNOWN'
    if 'up_limit' not in df: df['up_limit']=np.nan
    g=df.groupby('ts_code',group_keys=False)
    prev=g.close.shift(1); ret=df.close/prev-1; df['_ret']=ret
    df['_mkt']=df.groupby('trade_date')['_ret'].transform('mean')
    for w,n in [(1,'RET_1'),(3,'RET_3'),(5,'RET_5'),(10,'RET_10'),(20,'RET_20'),(60,'RET_60')]: df[n]=g.close.pct_change(w,fill_method=None)
    df['MOM_20_5']=g.close.shift(5)/g.close.shift(20)-1; df['MOM_60_10']=g.close.shift(10)/g.close.shift(60)-1
    for n in [1,3,5]: df[f'REV_{n}']=-df[f'RET_{n}']
    df['MAXRET_5']=g['_ret'].transform(lambda s:s.rolling(5,min_periods=5).max()); df['MINRET_5']=g['_ret'].transform(lambda s:s.rolling(5,min_periods=5).min())
    df['MAXRET_20']=g['_ret'].transform(lambda s:s.rolling(20,min_periods=20).max()); df['MINRET_20']=g['_ret'].transform(lambda s:s.rolling(20,min_periods=20).min())
    df['STREAK_5']=g['_ret'].transform(lambda s:np.sign(s).rolling(5,min_periods=5).sum())
    for w in [5,10,20,60]: df[f'VOL_{w}']=g['_ret'].transform(lambda s:s.rolling(w,min_periods=w).std())*np.sqrt(252)
    for w in [20,60]: df[f'DOWNSIDE_{w}']=g['_ret'].transform(lambda s:s.clip(upper=0).pow(2).rolling(w,min_periods=w).mean().pow(.5))*np.sqrt(252)
    tr=pd.concat([(df.high-df.low).abs(),(df.high-prev).abs(),(df.low-prev).abs()],axis=1).max(axis=1); df['_tr']=tr
    df['ATR_14']=g['_tr'].transform(lambda s:s.rolling(14,min_periods=14).mean())/(df.close.abs()+EPS)
    df['_range']=(df.high-df.low)/(df.close.abs()+EPS); df['RANGEVOL_20']=g['_range'].transform(lambda s:s.rolling(20,min_periods=20).mean())
    for col,short,long,name in [('volume',5,20,'VOL_RATIO_5_20'),('volume',20,60,'VOL_RATIO_20_60'),('amount',5,20,'AMOUNT_RATIO_5_20'),('amount',20,60,'AMOUNT_RATIO_20_60')]:
        num=g[col].transform(lambda s:s.rolling(short,min_periods=short).mean()); den=g[col].transform(lambda s:s.rolling(long,min_periods=long).mean()); df[name]=num/(den+EPS)
    df['_illiq']=df['_ret'].abs()/(df.amount.abs()+EPS)
    for w in [20,60]: df[f'AMIHUD_{w}']=g['_illiq'].transform(lambda s:s.rolling(w,min_periods=w).mean())
    df['_vchg']=g.volume.pct_change(fill_method=None)
    pv=[]
    for _,x in df.groupby('ts_code',sort=False): pv.append(x['_ret'].rolling(20,min_periods=20).corr(x['_vchg']))
    df['PV_CORR_20']=pd.concat(pv).sort_index()
    vm=g.volume.transform(lambda s:s.rolling(20,min_periods=20).mean()); vs=g.volume.transform(lambda s:s.rolling(20,min_periods=20).std()); df['VOLUME_CV_20']=vs/(vm.abs()+EPS)
    rng=(df.high-df.low).abs()+EPS
    df['BODY_FRAC']=(df.close-df.open)/rng; df['ABS_BODY_FRAC']=(df.close-df.open).abs()/rng; df['UPPER_SHADOW']=(df.high-df[['open','close']].max(axis=1))/rng; df['LOWER_SHADOW']=(df[['open','close']].min(axis=1)-df.low)/rng; df['CLOSE_LOCATION']=(df.close-df.low)/rng; df['OPEN_GAP']=df.open/prev-1
    hh=g.high.transform(lambda s:s.rolling(20,min_periods=20).max()); ll=g.low.transform(lambda s:s.rolling(20,min_periods=20).min()); df['HIGH_BREAK_20']=df.close/(hh+EPS)-1; df['LOW_DIST_20']=df.close/(ll+EPS)-1
    df['OVERNIGHT_1']=df['OPEN_GAP']; df['INTRADAY_1']=df.close/df.open-1
    for w in [5,20]:
        df[f'OVERNIGHT_{w}']=g['OVERNIGHT_1'].transform(lambda s:_comp_ret(s,w)); df[f'INTRADAY_{w}']=g['INTRADAY_1'].transform(lambda s:_comp_ret(s,w)); df[f'ON_MINUS_ID_{w}']=df[f'OVERNIGHT_{w}']-df[f'INTRADAY_{w}']
    for w in [1,5,20,60]:
        r=df[f'RET_{w}']; df[f'MKT_REL_{w}']=r-r.groupby(df.trade_date).transform('mean'); df[f'SECTOR_REL_{w}']=r-r.groupby([df.trade_date,df.industry]).transform('mean')
    slopes20=[]; rsq20=[]; slopes60=[]; rsq60=[]
    for _,x in df.groupby('ts_code',sort=False):
        lp=np.log(x.close.clip(lower=EPS)); s20,r20=_roll_slope_rsq(lp,20); s60,r60=_roll_slope_rsq(lp,60); slopes20.append(s20);rsq20.append(r20);slopes60.append(s60);rsq60.append(r60)
    df['SLOPE_20']=pd.concat(slopes20).sort_index(); df['SLOPE_60']=pd.concat(slopes60).sort_index(); df['RSQ_20']=pd.concat(rsq20).sort_index(); df['RSQ_60']=pd.concat(rsq60).sort_index()
    beta=[]
    for _,x in df.groupby('ts_code',sort=False): beta.append(x['_ret'].rolling(60,min_periods=60).cov(x['_mkt'])/(x['_mkt'].rolling(60,min_periods=60).var()+EPS))
    df['BETA_60']=pd.concat(beta).sort_index(); df['_resid']=df['_ret']-df['BETA_60']*df['_mkt']
    df['IDIO_VOL_20']=df.groupby('ts_code')['_resid'].transform(lambda s:s.rolling(20,min_periods=20).std())*np.sqrt(252); df['RESID_MOM_20']=df.groupby('ts_code')['_resid'].transform(lambda s:s.rolling(20,min_periods=20).sum()); df['RESID_MOM_60']=df.groupby('ts_code')['_resid'].transform(lambda s:s.rolling(60,min_periods=60).sum())
    for w in [20,60]:
        m=g.turnover_rate.transform(lambda s:s.rolling(w,min_periods=w).mean()); sd=g.turnover_rate.transform(lambda s:s.rolling(w,min_periods=w).std()); df[f'TURNOVER_Z_{w}']=(df.turnover_rate-m)/(sd+EPS)
    df['UP_LIMIT_PROX']=1-(df.up_limit-df.close)/(df.up_limit.abs()+EPS); df['LIMIT_HIT_UP']=(df.high>=0.999*df.up_limit).astype(float).where(df.up_limit.notna()); df['LIMIT_CLOSE_UP']=(df.close>=0.999*df.up_limit).astype(float).where(df.up_limit.notna()); df['ABS_GAP']=df['OPEN_GAP'].abs(); df['ZERO_RET_FRAC_20']=g['_ret'].transform(lambda s:(s.abs()<1e-6).rolling(20,min_periods=20).mean()); df['ILLIQ_SHOCK']=df['AMIHUD_20']/(df['AMIHUD_60']+EPS)
    mapping={'F01':'RET_1','F02':'RET_3','F03':'RET_5','F04':'RET_10','F05':'RET_20','F06':'RET_60','F07':'MOM_20_5','F08':'MOM_60_10','F09':'REV_1','F10':'REV_3','F11':'REV_5','F12':'MAXRET_5','F13':'MINRET_5','F14':'MAXRET_20','F15':'MINRET_20','F16':'STREAK_5','F17':'VOL_5','F18':'VOL_10','F19':'VOL_20','F20':'VOL_60','F21':'DOWNSIDE_20','F22':'DOWNSIDE_60','F23':'ATR_14','F24':'RANGEVOL_20','F25':'VOL_RATIO_5_20','F26':'VOL_RATIO_20_60','F27':'AMOUNT_RATIO_5_20','F28':'AMOUNT_RATIO_20_60','F29':'AMIHUD_20','F30':'AMIHUD_60','F31':'PV_CORR_20','F32':'VOLUME_CV_20','F33':'BODY_FRAC','F34':'ABS_BODY_FRAC','F35':'UPPER_SHADOW','F36':'LOWER_SHADOW','F37':'CLOSE_LOCATION','F38':'OPEN_GAP','F39':'HIGH_BREAK_20','F40':'LOW_DIST_20','F41':'OVERNIGHT_1','F42':'OVERNIGHT_5','F43':'OVERNIGHT_20','F44':'INTRADAY_1','F45':'INTRADAY_5','F46':'INTRADAY_20','F47':'ON_MINUS_ID_5','F48':'ON_MINUS_ID_20','F49':'MKT_REL_1','F50':'MKT_REL_5','F51':'MKT_REL_20','F52':'MKT_REL_60','F53':'SECTOR_REL_1','F54':'SECTOR_REL_5','F55':'SECTOR_REL_20','F56':'SECTOR_REL_60','F57':'SLOPE_20','F58':'SLOPE_60','F59':'RSQ_20','F60':'RSQ_60','F61':'BETA_60','F62':'IDIO_VOL_20','F63':'RESID_MOM_20','F64':'RESID_MOM_60','F65':'TURNOVER_Z_20','F66':'TURNOVER_Z_60','F67':'UP_LIMIT_PROX','F68':'LIMIT_HIT_UP','F69':'LIMIT_CLOSE_UP','F70':'ABS_GAP','F71':'ZERO_RET_FRAC_20','F72':'ILLIQ_SHOCK'}
    out=df[['trade_date','ts_code','industry']].copy()
    for fid,n in mapping.items(): out[fid]=df[n]
    return out
