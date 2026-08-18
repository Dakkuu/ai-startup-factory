from pathlib import Path
import urllib.request
import pandas as pd

URL='https://raw.githubusercontent.com/JinanZou/Astock/main/data/df_all_year_srl.csv'
P=Path('astock_news.tsv')
if not P.exists():
    print('downloading news...', flush=True)
    urllib.request.urlretrieve(URL,P)
use=['CODE','NAME','DATE','CREATED_DATE','text_a','DESCRIPTION','READ','MARKET']
df=pd.read_csv(P,sep='\t',usecols=use,low_memory=False)
for c in ['DATE','CREATED_DATE']:
    df[c]=pd.to_datetime(df[c],errors='coerce')
df['CODE']=pd.to_numeric(df.CODE,errors='coerce').astype('Int64')
print('rows',len(df),'codes',df.CODE.nunique(),flush=True)
print('DATE range',df.DATE.min(),df.DATE.max(),flush=True)
print('CREATED range',df.CREATED_DATE.min(),df.CREATED_DATE.max(),flush=True)
print('missing created',df.CREATED_DATE.isna().mean(),flush=True)
print('daily news quantiles')
print(df.groupby(df.CREATED_DATE.dt.date).size().quantile([0,.1,.25,.5,.75,.9,1]).to_string(),flush=True)
print('year counts')
print(df.groupby(df.CREATED_DATE.dt.year).agg(rows=('CODE','size'),codes=('CODE','nunique')).to_string(),flush=True)
print('read stats')
print(pd.to_numeric(df.READ,errors='coerce').describe(percentiles=[.1,.5,.9,.99]).to_string(),flush=True)
print('sample latest')
print(df.sort_values('CREATED_DATE').tail(10)[['CODE','NAME','CREATED_DATE','READ','text_a']].to_string(index=False),flush=True)
