from pathlib import Path
import pandas as pd
import run_10y_alpha2f as a

IN=Path('failed_artifact/panel_core.pkl')
OUT=Path('results_alpha2f_recovered'); OUT.mkdir(exist_ok=True)
p=pd.read_pickle(IN)
if 'signal_date' not in p.columns:
    p['signal_date']=pd.to_datetime(p['trade_date'])
print('PANEL', p.shape, 'groups', p.signal_date.nunique(), 'median_group', p.groupby('signal_date').size().median(), 'min', p.groupby('signal_date').size().min(), 'max', p.groupby('signal_date').size().max(), flush=True)

rows=[]; annual=[]; eqs=[]; trs=[]
for v in ['rmom','ivol','2f']:
    eq,tr,tm,to=a.simulate(p,v,1.0)
    st=a.perf(eq,tr,to); st['variant']=v
    st['train_2016_2021_return']=a.period_return(eq,'2016-07-29','2021-12-31')
    st['sealed_2022_2026_return']=a.period_return(eq,'2022-01-01','2026-07-29')
    rows.append(st); eq['variant']=v; tr['variant']=v; eqs.append(eq); trs.append(tr)
    ar=a.annual_returns(eq); ar['variant']=v; annual.append(ar)
eq2,tr2,tm2,to2=a.simulate(p,'2f',2.0)
stress=a.perf(eq2,tr2,to2); stress['variant']='2f_double_cost'
sm=pd.DataFrame(rows)
rob=pd.DataFrame([a.robustness(eqs[2],trs[2])])
sm.to_csv(OUT/'summary.csv',index=False)
pd.DataFrame([stress]).to_csv(OUT/'double_cost.csv',index=False)
rob.to_csv(OUT/'robustness.csv',index=False)
pd.concat(annual,ignore_index=True).to_csv(OUT/'annual_returns.csv',index=False)
pd.concat(eqs,ignore_index=True).to_csv(OUT/'equity.csv',index=False)
pd.concat(trs,ignore_index=True).to_csv(OUT/'trades.csv',index=False)
print('=== SUMMARY ==='); print(sm.to_string(index=False))
print('=== DOUBLE COST ==='); print(pd.DataFrame([stress]).to_string(index=False))
print('=== ROBUSTNESS ==='); print(rob.to_string(index=False))
print('=== ANNUAL ==='); print(pd.concat(annual,ignore_index=True).pivot(index='year',columns='variant',values='return').to_string())
