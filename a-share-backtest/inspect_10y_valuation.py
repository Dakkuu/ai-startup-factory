import pandas as pd
import akshare as ak

# Valuation history: all-A median and equal-weight PE.
x=ak.stock_a_ttm_lyr()
print('columns',list(x.columns),flush=True)
print('rows',len(x),flush=True)
print(x.head(3).to_string(index=False),flush=True)
print(x.tail(5).to_string(index=False),flush=True)
print('date range',x['date'].min(),x['date'].max(),flush=True)
x.to_csv('valuation_raw.csv',index=False)

# Demand/leverage proxy: SSE full history + SZSE at each year's final SSE trading date.
sse=ak.stock_margin_sse(start_date='20160101',end_date='20260729')
sse['信用交易日期']=pd.to_datetime(sse['信用交易日期'])
sse=sse.sort_values('信用交易日期')
print('SSE margin rows',len(sse),'range',sse['信用交易日期'].min(),sse['信用交易日期'].max(),flush=True)
rows=[]
for y in range(2016,2027):
    z=sse[sse['信用交易日期'].dt.year==y]
    if z.empty: continue
    rr=z.iloc[-1]; d=pd.Timestamp(rr['信用交易日期'])
    sz=None
    # Try final SSE date and previous four calendar days in case exchange calendars differ.
    for lag in range(5):
        ds=(d-pd.Timedelta(days=lag)).strftime('%Y%m%d')
        try:
            q=ak.stock_margin_szse(date=ds)
            if q is not None and len(q):
                sz=q.iloc[0]; dsz=ds; break
        except Exception:
            pass
    sse_bal=float(rr['融资余额']) if pd.notna(rr['融资余额']) else float('nan')
    sz_bal=float(sz['融资余额']) if sz is not None and pd.notna(sz['融资余额']) else float('nan')
    rows.append({'year':y,'date_sse':d.date(),'date_szse':dsz if sz is not None else None,'sse_financing_balance':sse_bal,'szse_financing_balance':sz_bal,'combined_financing_balance':sse_bal+sz_bal if pd.notna(sse_bal) and pd.notna(sz_bal) else float('nan')})
m=pd.DataFrame(rows)
print('=== YEAR-END MARGIN DEMAND PROXY ===',flush=True)
print(m.to_string(index=False),flush=True)
m.to_csv('margin_yearend.csv',index=False)
