from __future__ import annotations
from pathlib import Path
import argparse, time
import numpy as np
import pandas as pd
import baostock as bs

FIELDS='date,code,close,volume,turn,tradestatus,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST'
START='2015-01-01'; END='2026-07-29'

def bs_code(q):
    q=str(q).upper()
    if q.startswith('SH'): return 'sh.'+q[2:]
    if q.startswith('SZ'): return 'sz.'+q[2:]
    if q.startswith('BJ'): return 'bj.'+q[2:]
    return q.lower()

def read_rs(rs):
    rows=[]
    while rs.error_code=='0' and rs.next(): rows.append(rs.get_row_data())
    return pd.DataFrame(rows,columns=rs.fields) if rows else pd.DataFrame(columns=rs.fields)

def one(code,signals):
    bc=bs_code(code)
    rs=bs.query_history_k_data_plus(bc,FIELDS,start_date=START,end_date=END,frequency='d',adjustflag='3')
    if rs.error_code!='0': return pd.DataFrame(),rs.error_code+':'+rs.error_msg
    d=read_rs(rs)
    if d.empty: return d,'EMPTY'
    d['date']=pd.to_datetime(d['date'],errors='coerce')
    for c in ['close','volume','turn','tradestatus','peTTM','pbMRQ','psTTM','pcfNcfTTM','isST']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['date']).sort_values('date').drop_duplicates('date',keep='last').set_index('date')
    # turnover is percentage. Infer PIT float shares from raw volume / turnover rate; smooth shares to suppress one-day data glitches.
    fshares=d['volume']/(d['turn']/100.0).replace(0,np.nan)
    bad=(fshares<=0)|(~np.isfinite(fshares)); fshares=fshares.mask(bad)
    fshares20=fshares.rolling(20,min_periods=5).median()
    d['float_mv_proxy']=d['close']*fshares20
    d['turn20']=d['turn'].rolling(20,min_periods=10).mean()
    d['turn60']=d['turn'].rolling(60,min_periods=30).mean()
    d['float_shares20']=fshares20
    # sample weekly signal grid causally. Up to 4-session forward-fill is allowed from the latest PRIOR observation only.
    idx=d.index.union(signals).sort_values()
    x=d[['close','volume','turn','turn20','turn60','float_shares20','float_mv_proxy','tradestatus','peTTM','pbMRQ','psTTM','pcfNcfTTM','isST']].reindex(idx).ffill(limit=4).reindex(signals)
    x=x.reset_index().rename(columns={'index':'signal_date'}); x['code']=code
    # Flag exact-date observation versus causal carry-forward.
    x['exact_obs']=x['signal_date'].isin(d.index).astype(int)
    return x,''

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,required=True); ap.add_argument('--nshards',type=int,required=True); a=ap.parse_args()
    prep=Path('baostock_prep'); out=Path('baostock_shard_out'); out.mkdir(exist_ok=True)
    codes=[x.strip() for x in (prep/'codes.txt').read_text(encoding='utf-8').splitlines() if x.strip()]
    signals=pd.DatetimeIndex(pd.to_datetime([x.strip() for x in (prep/'signal_dates.txt').read_text(encoding='utf-8').splitlines() if x.strip()]))
    mine=[c for i,c in enumerate(codes) if i%a.nshards==a.shard]
    lg=bs.login()
    if lg.error_code!='0': raise RuntimeError('baostock login '+lg.error_code+' '+lg.error_msg)
    frames=[]; errs=[]
    try:
        for j,c in enumerate(mine,1):
            try:
                z,err=one(c,signals)
                if len(z): frames.append(z)
                if err: errs.append({'code':c,'error':err})
            except Exception as e:
                errs.append({'code':c,'error':repr(e)})
            if j%25==0: print('SHARD',a.shard,j,'/',len(mine),'rows',sum(len(x) for x in frames),'errs',len(errs),flush=True)
            time.sleep(.02)
    finally:
        bs.logout()
    data=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    data.to_csv(out/f'factor_shard_{a.shard:02d}.csv.gz',index=False,compression='gzip')
    pd.DataFrame(errs).to_csv(out/f'errors_{a.shard:02d}.csv',index=False)
    pd.DataFrame([{'shard':a.shard,'nshards':a.nshards,'requested_codes':len(mine),'returned_codes':int(data.code.nunique()) if len(data) else 0,'rows':len(data),'errors':len(errs)}]).to_csv(out/f'manifest_{a.shard:02d}.csv',index=False)
    print('DONE SHARD',a.shard,'codes',len(mine),'returned',int(data.code.nunique()) if len(data) else 0,'rows',len(data),'errors',len(errs),flush=True)

if __name__=='__main__': main()
