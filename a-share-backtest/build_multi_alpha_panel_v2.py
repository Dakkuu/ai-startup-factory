from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

import run_multi_alpha_system_v1 as v1
import run_geff_fundamental_integrated_v3 as iv3
import run_10y_geff55_strict_audit_v2 as strict

OUT=Path('results_multi_alpha_panel_v2'); OUT.mkdir(exist_ok=True)


def main():
    p,z,cal,members,ua,market_code,bm=v1.load_base_panel()
    p=v1.add_short_features(p,cal)
    p=v1.add_long_ranks(p,z)

    basecols=['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor']
    flagcols=[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in p.columns]
    s=p[basecols+flagcols].copy()

    for fam in v1.SHORT_FAMILIES:
        q=v1.make_short_q(p,fam)
        s[f'rank_short_{fam}']=q.rank_test.to_numpy(float)
    for fam in v1.LONG_FAMILIES:
        q=v1.make_long_q(p,fam)
        s[f'rank_long_{fam}']=q.rank_test.to_numpy(float)
    qm=iv3.build_candidates(p)['mom_cfo10_qv10']
    s['rank_medium']=qm.rank_test.to_numpy(float)

    # compact types; dates retained as timestamps, code dictionary-encodes well in parquet.
    for c in [x for x in s.columns if x.startswith('rank_')]: s[c]=pd.to_numeric(s[c],errors='coerce').astype('float32')
    for c in ['exec_open','exec_high','exec_low','exec_volume','exec_factor','exec_open_gap','exec_limit_proxy','liq20']:
        if c in s: s[c]=pd.to_numeric(s[c],errors='coerce').astype('float32')
    s.to_parquet(OUT/'signals.parquet',compression='zstd',index=False)
    bm.rename('close').to_csv(OUT/'benchmark.csv',header=True)
    meta={'status':'REUSABLE_SIGNAL_PANEL_FOR_PREREG_MULTI_ALPHA_V2','rows':len(s),'signal_dates':int(s.signal_date.nunique()),'market_factor':market_code,'release_tag':'2026-07-29','prereg':'MULTI_ALPHA_PREREG_2026-09-03.md','rank_columns':[c for c in s if c.startswith('rank_')],'universe_audit':ua}
    (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
    print(json.dumps(meta,ensure_ascii=False,indent=2,default=str),flush=True)
    print('PARQUET_MB',round((OUT/'signals.parquet').stat().st_size/1024/1024,2),flush=True)

if __name__=='__main__': main()
