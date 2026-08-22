from pathlib import Path
import re
import pandas as pd
import run_backtest_qlib as qb

OUT=Path('baostock_prep'); OUT.mkdir(exist_ok=True)
RELEASE_TAG='2026-07-29'
STOCK_RE=re.compile(r'^(?:SH(?:600|601|603|605|688)\d{3}|SZ(?:000|001|002|003|300|301)\d{3}|BJ\d{6})$')
FETCH_START=pd.Timestamp('2015-01-01'); SIGNAL_START=pd.Timestamp('2016-07-29'); END=pd.Timestamp('2026-07-29')

qb.RELEASE_TAG=RELEASE_TAG; qb.ROOT=Path('qlib_data'); qb.download_and_extract()
cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(qb.ROOT/'calendars'/'day.txt',header=None)[0]))
m=pd.read_csv(qb.ROOT/'instruments'/'all.txt',sep='\t',header=None,names=['code','start','end'],usecols=[0,1,2])
m['code']=m.code.astype(str).str.upper(); m['start']=pd.to_datetime(m.start); m['end']=pd.to_datetime(m.end)
m=m[m.code.str.match(STOCK_RE)].copy(); m=m[(m.end>=FETCH_START)&(m.start<=END)]
codes=sorted(m.code.unique())
(OUT/'codes.txt').write_text('\n'.join(codes),encoding='utf-8')
sig=cal[(cal>=SIGNAL_START)&(cal<=END)][::5]
(OUT/'signal_dates.txt').write_text('\n'.join(pd.Series(sig).dt.strftime('%Y-%m-%d')),encoding='utf-8')
pd.DataFrame([{'release_tag':RELEASE_TAG,'fetch_start':str(FETCH_START.date()),'signal_start':str(SIGNAL_START.date()),'end':str(END.date()),'codes':len(codes),'signal_dates':len(sig),'min_member_start':str(m.start.min().date()),'max_member_end':str(m.end.max().date())}]).to_csv(OUT/'manifest.csv',index=False)
print('PREP',len(codes),'codes',len(sig),'signal dates',flush=True)
