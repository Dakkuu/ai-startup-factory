from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_geff_factor_attribution_v1'); OUT.mkdir(exist_ok=True)
TRAIN_END=pd.Timestamp('2021-12-31'); PSEUDO_START=pd.Timestamp('2022-01-01')
BASE={'iv':.25,'down':.15,'rmom':.35,'tstat':.25}
FACTORS=['iv','down','rmom','tstat','ef','beta','capture','amax','askew','dd','mom','volshock']

def cross_metrics(q,label,a,b):
    z=q.copy(); z['signal_date']=pd.to_datetime(z.signal_date)
    z=z[(z.signal_date>=pd.Timestamp(a))&(z.signal_date<=pd.Timestamp(b))][['signal_date','rank_test','fwd60']].dropna()
    ics=[]; spreads=[]; top=[]; bot=[]; rows=[]
    for d,g in z.groupby('signal_date',sort=True):
        if len(g)<200: continue
        # lower rank_test is better; convert to higher-good for IC
        good=-g.rank_test.astype(float)
        ic=good.corr(g.fwd60.astype(float),method='spearman')
        rr=g.rank_test.rank(pct=True,method='average')
        hi=g.loc[rr<=.10,'fwd60']; lo=g.loc[rr>=.90,'fwd60']
        sp=float(hi.mean()-lo.mean()) if len(hi)>=10 and len(lo)>=10 else np.nan
        if np.isfinite(ic): ics.append(float(ic))
        if np.isfinite(sp): spreads.append(sp)
        if len(hi): top.append(float(hi.mean()))
        if len(lo): bot.append(float(lo.mean()))
        rows.append({'signal':label,'period':f'{a}_{b}','date':d,'n':len(g),'ic':ic,'top_bottom_spread':sp})
    arr=np.asarray(ics,float)
    t=float(arr.mean()/arr.std(ddof=1)*np.sqrt(len(arr))) if len(arr)>1 and arr.std(ddof=1)>0 else np.nan
    return {
      'signal':label,'period':f'{a}_{b}','dates':len(arr),'mean_ic':float(np.mean(arr)) if len(arr) else np.nan,
      'ic_t':t,'positive_ic_share':float((arr>0).mean()) if len(arr) else np.nan,
      'top_bottom_spread60':float(np.mean(spreads)) if spreads else np.nan,
      'top_decile_fwd60':float(np.mean(top)) if top else np.nan,
      'bottom_decile_fwd60':float(np.mean(bot)) if bot else np.nan,
    },rows

def single_spec(k):
    return {'name':f'single_{k}','kind':'gate','g':{'ef':.55},'w':{k:1.0}}

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=True)
    p=strict.attach_gap_flags(p,cal,'board')
    specs=[{'name':'base_mom','kind':'gate','g':{'ef':.55},'w':BASE}] + [single_spec(k) for k in FACTORS]
    summaries=[]; dates=[]
    for sp in specs:
        print('SIGNAL',sp['name'],flush=True)
        q=mega.make_rank(p,sp)
        for label,a,b in [('train','2016-08-02','2021-12-31'),('pseudo','2022-01-01','2026-07-29')]:
            s,r=cross_metrics(q,sp['name'],a,b); s['split']=label; summaries.append(s); dates.extend(r)
    df=pd.DataFrame(summaries)
    # Component robustness score: positive train and pseudo IC/spread; no performance-based weight tuning.
    piv=df.pivot(index='signal',columns='split',values=['mean_ic','ic_t','top_bottom_spread60'])
    rows=[]
    for sig in df.signal.unique():
        tr=df[(df.signal==sig)&(df.split=='train')].iloc[0]; ps=df[(df.signal==sig)&(df.split=='pseudo')].iloc[0]
        rows.append({'signal':sig,'train_mean_ic':tr.mean_ic,'train_ic_t':tr.ic_t,'train_spread60':tr.top_bottom_spread60,'pseudo_mean_ic':ps.mean_ic,'pseudo_ic_t':ps.ic_t,'pseudo_spread60':ps.top_bottom_spread60,'both_ic_positive':int(tr.mean_ic>0 and ps.mean_ic>0),'both_spread_positive':int(tr.top_bottom_spread60>0 and ps.top_bottom_spread60>0)})
    rob=pd.DataFrame(rows).sort_values(['both_ic_positive','both_spread_positive','train_ic_t','pseudo_ic_t'],ascending=[False,False,False,False])
    df.to_csv(OUT/'factor_split_metrics.csv',index=False); pd.DataFrame(dates).to_csv(OUT/'factor_date_metrics.csv',index=False); rob.to_csv(OUT/'factor_robustness.csv',index=False)
    meta={'market_factor':market_code,'universe_audit':ua,'horizon':'forward 60 trading-session adjusted-close return from signal close; diagnostic only','factors':FACTORS}
    (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
    print('\n=== FACTOR ROBUSTNESS ==='); print(rob.to_string(index=False),flush=True)
    print('\n=== SPLITS ==='); print(df.to_string(index=False),flush=True)
if __name__=='__main__': main()
