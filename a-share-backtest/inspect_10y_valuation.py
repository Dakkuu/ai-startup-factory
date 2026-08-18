import pandas as pd
import akshare as ak

x=ak.stock_a_ttm_lyr()
print('columns',list(x.columns),flush=True)
print('rows',len(x),flush=True)
print(x.head(3).to_string(index=False),flush=True)
print(x.tail(5).to_string(index=False),flush=True)
print('date range',x['date'].min(),x['date'].max(),flush=True)
x.to_csv('valuation_raw.csv',index=False)
