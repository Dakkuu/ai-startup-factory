import inspect, json, traceback
import pandas as pd
import akshare as ak

DATES = ['20160729','20190104','20210104','20240104','20260728']

def show(name, fn):
    try:
        print(f'\n## {name} signature: {inspect.signature(fn)}')
    except Exception as e:
        print('signature err', e)

for n in ['stock_zt_pool_em','stock_zt_pool_previous_em','stock_lhb_detail_em','stock_lhb_hyyyb_em','stock_lhb_jgmmtj_em','stock_zh_a_disclosure_report_cninfo']:
    if hasattr(ak,n): show(n,getattr(ak,n))

for d in DATES:
    print('\n=== DATE', d, '===')
    for name, call in [
        ('zt_pool', lambda d=d: ak.stock_zt_pool_em(date=d)),
        ('zt_prev', lambda d=d: ak.stock_zt_pool_previous_em(date=d)),
        ('lhb_detail', lambda d=d: ak.stock_lhb_detail_em(start_date=d,end_date=d)),
    ]:
        try:
            df=call()
            print(name, 'shape=', getattr(df,'shape',None), 'cols=', list(df.columns) if hasattr(df,'columns') else None)
            if hasattr(df,'head') and not df.empty:
                print(df.head(3).to_string(index=False))
        except Exception as e:
            print(name,'ERROR',type(e).__name__,str(e)[:500])

# Probe CNINFO all-market keyword query without a symbol.
for start,end,kw in [('20190101','20190131','回购'),('20240101','20240107','中标')]:
    try:
        df=ak.stock_zh_a_disclosure_report_cninfo(symbol='', market='沪深京', keyword=kw, category='', start_date=start,end_date=end)
        print('\nCNINFO',start,end,kw,'shape=',df.shape,'cols=',list(df.columns))
        print(df.head(5).to_string(index=False))
    except Exception as e:
        print('\nCNINFO ERROR',start,end,kw,type(e).__name__,str(e)[:800])
