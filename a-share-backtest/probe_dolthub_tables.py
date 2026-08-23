from __future__ import annotations
from pathlib import Path
import json, requests, pandas as pd

OUT=Path('results_alpha500_probe'); OUT.mkdir(exist_ok=True)
ROOT='https://www.dolthub.com/api/v1alpha1/chenditc/investment_data/master'

def q(sql):
    r=requests.get(ROOT,params={'q':sql},timeout=120)
    r.raise_for_status()
    z=r.json()
    if z.get('query_execution_status')!='Success':
        raise RuntimeError(z)
    return z.get('rows',[])

rows=q('SHOW TABLES')
pd.DataFrame(rows).to_csv(OUT/'tables.csv',index=False)
print('TABLES')
print(pd.DataFrame(rows).to_string(index=False),flush=True)

schema=q("SELECT table_name,column_name,data_type FROM information_schema.columns WHERE table_schema=DATABASE() ORDER BY table_name,ordinal_position")
s=pd.DataFrame(schema)
s.to_csv(OUT/'schema_all.csv',index=False)
keys=('basic','daily','stock','market','value','valuation','indicator','factor','financial','share')
if len(s):
    hit=s[s.table_name.astype(str).str.lower().apply(lambda x:any(k in x for k in keys)) | s.column_name.astype(str).str.lower().isin(['total_mv','circ_mv','pe','pe_ttm','pb','ps','turnover_rate','turnover_rate_f','total_share','float_share','free_share'])]
else:
    hit=s
hit.to_csv(OUT/'schema_hits.csv',index=False)
print('SCHEMA HITS')
print(hit.to_string(index=False),flush=True)

for t in sorted(set(hit.table_name.astype(str)))[:30]:
    try:
        sample=pd.DataFrame(q(f'SELECT * FROM `{t}` LIMIT 3'))
        print('\nSAMPLE',t)
        print(sample.to_string(index=False),flush=True)
    except Exception as e:
        print('SAMPLE FAIL',t,repr(e),flush=True)
