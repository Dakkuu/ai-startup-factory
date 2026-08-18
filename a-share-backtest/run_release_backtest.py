from __future__ import annotations

import os, time
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import baostock as bs

import run_backtest as core

START=os.getenv('BT_START','2025-08-01')
END=os.getenv('BT_END','2026-08-18')
WARM=(pd.Timestamp(START)-pd.Timedelta(days=240)).strftime('%Y-%m-%d')
DATA=Path('data'); OUT=Path('results'); DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)


def qdf(rs):
    rows=[]
    while rs.error_code=='0' and rs.next(): rows.append(rs.get_row_data())
    if rs.error_code!='0': raise RuntimeError(rs.error_msg)
    return pd.DataFrame(rows,columns=rs.fields)


def download_cn_release():
    meta_url='https://api.github.com/repos/irachex/open-stock-data/releases/tags/data-cn-bars'
    r=requests.get(meta_url,timeout=30,headers={'Accept':'application/vnd.github+json'})
    r.raise_for_status(); rel=r.json()
    assets={a['name']:a for a in rel.get('assets',[])}
    if 'cn_bars.parquet' not in assets:
        raise RuntimeError('cn_bars.parquet not found in release')
    a=assets['cn_bars.parquet']; url=a['browser_download_url']; path=DATA/'cn_bars.parquet'
    print('Downloading release asset',a.get('size'),'bytes from',url)
    with requests.get(url,stream=True,timeout=300) as resp:
        resp.raise_for_status()
        with path.open('wb') as f:
            for chunk in resp.iter_content(1024*1024):
                if chunk:f.write(chunk)
    df=pd.read_parquet(path)
    print('Release bars rows=',len(df),'cols=',list(df.columns))
    return df


def historical_membership_and_index():
    lg=bs.login()
    if lg.error_code!='0': raise RuntimeError('BaoStock login failed '+lg.error_msg)
    try:
        cal=qdf(bs.query_trade_dates(start_date=WARM,end_date=END))
        days=pd.to_datetime(cal.loc[cal.is_trading_day=='1','calendar_date']).sort_values().tolist()
        weekly=[]; seen=set()
        for d in days:
            k=d.to_period('W-MON')
            if k not in seen: weekly.append(d); seen.add(k)
        snapshots={}; union=set()
        for i,d in enumerate(weekly):
            z=qdf(bs.query_zz500_stocks(date=d.strftime('%Y-%m-%d')))
            if not z.empty:
                codes=set(z.code.astype(str).str.split('.').str[-1].str.zfill(6)); snapshots[pd.Timestamp(d)]=codes; union |= codes
            if (i+1)%15==0: print('membership',i+1,'/',len(weekly),'union=',len(union))
            time.sleep(.02)
        idx=qdf(bs.query_history_k_data_plus('sh.000905','date,code,open,high,low,close,preclose,volume,amount,pctChg',start_date=WARM,end_date=END,frequency='d',adjustflag='3'))
        return snapshots,union,idx
    finally:
        bs.logout()


def normalize_release(df,union):
    x=df.copy(); x['date']=pd.to_datetime(x['date']); x['code']=x['code'].astype(str).str.extract(r'(\d{6})',expand=False).fillna(x['code'].astype(str).str[-6:]).str.zfill(6)
    x=x[(x['date']>=pd.Timestamp(WARM))&(x['date']<=pd.Timestamp(END))&x['code'].isin(union)].copy()
    for c in ['open','high','low','close','volume','amount','turnover']:
        if c in x:x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x.sort_values(['code','date']); x['preclose']=x.groupby('code')['close'].shift(1); x['pctChg']=(x.close/x.preclose-1)*100
    x['turn']=x['turnover'] if 'turnover' in x else np.nan; x['tradestatus']=1; x['isST']=0
    cols=['date','code','open','high','low','close','preclose','volume','amount','turn','tradestatus','pctChg','isST']
    x=x[cols].dropna(subset=['date','code','open','close'])
    print('Filtered bars=',len(x),'codes=',x.code.nunique(),'dates',x.date.min(),x.date.max())
    if x.code.nunique()<300: raise RuntimeError('Release asset lacks enough historical CSI500 codes')
    if x.date.max()<pd.Timestamp(END)-pd.Timedelta(days=7): raise RuntimeError(f'Release stale: max={x.date.max()} expected near {END}')
    return x


def run_all(prices,idx,snapshots):
    prices,idx=core.clean(prices,idx); f=core.add_features(prices)
    names=['trend_breakout','relative_momentum','mean_reversion','lowvol_trend','volume_breakout','multifactor']
    rows=[]; eqs=[]; trs=[]; bm,bmeq=core.benchmark(idx); bmret=bm['total_return']
    for s in names:
        print('Running',s); eq,tr=core.backtest(f,idx,snapshots,s); m=core.stats(eq,tr); m.update(strategy=s,excess_vs_csi500=m['total_return']-bmret); rows.append(m); eqs.append(eq)
        if not tr.empty: trs.append(tr)
    b=bm.copy(); b.update(strategy='CSI500_buy_hold',excess_vs_csi500=0.0); rows.append(b)
    summary=pd.DataFrame(rows)[['strategy','total_return','excess_vs_csi500','cagr','max_drawdown','sharpe','sortino','trades','win_rate','avg_win','avg_loss','profit_factor','max_losing_streak']]
    summary.to_csv(OUT/'summary.csv',index=False); pd.concat(eqs).to_csv(OUT/'equity.csv',index=False); (pd.concat(trs) if trs else pd.DataFrame()).to_csv(OUT/'trades.csv',index=False); bmeq.to_csv(OUT/'benchmark_equity.csv',index=False)
    pd.DataFrame([{'start':START,'end':END,'data_source':'irachex/open-stock-data GitHub Release cn_bars.parquet + BaoStock historical CSI500 membership','price_adjustment':'raw/unadjusted','signal_execution':'T close -> T+1 open','historical_membership_snapshots':len(snapshots),'unique_codes':prices.code.nunique(),'price_rows':len(prices),'limitations':'Raw prices do not total-return-adjust dividends/rights. Historical ST flag not supplied by release; CSI500 membership largely screens risk-warning stocks. Missing trading rows are treated as non-tradable/held.'}]).to_csv(OUT/'metadata.csv',index=False)
    print('\n=== RESULTS ===\n'+summary.sort_values('total_return',ascending=False).to_string(index=False))


def main():
    bars=download_cn_release(); snapshots,union,idx=historical_membership_and_index(); prices=normalize_release(bars,union); run_all(prices,idx,snapshots)

if __name__=='__main__': main()
