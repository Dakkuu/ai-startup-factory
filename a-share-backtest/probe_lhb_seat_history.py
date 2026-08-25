import akshare as ak
import pandas as pd

DATES=['20160729','20190104','20210104','20240104','20260728']
for d in DATES:
    print('\n===',d,'===',flush=True)
    try:
        df=ak.stock_lhb_hyyyb_em(start_date=d,end_date=d)
        print('HYYB shape',df.shape,'cols',list(df.columns),flush=True)
        if len(df): print(df.head(12).to_string(index=False),flush=True)
    except Exception as e:
        print('HYYB ERROR',type(e).__name__,str(e)[:500],flush=True)
    try:
        df=ak.stock_lhb_jgmmtj_em(start_date=d,end_date=d)
        print('INST shape',df.shape,'cols',list(df.columns),flush=True)
        if len(df): print(df.head(5).to_string(index=False),flush=True)
    except Exception as e:
        print('INST ERROR',type(e).__name__,str(e)[:500],flush=True)
