from __future__ import annotations
from pathlib import Path
import glob,json
import numpy as np
import pandas as pd

import run_multi_alpha_system_v1 as v1
import run_geff_fundamental_ranktilt_v1 as rt

OUT=Path('results_multi_alpha_system_v2'); OUT.mkdir(exist_ok=True)


def one(pat):
    x=glob.glob(pat,recursive=True)
    if not x: raise FileNotFoundError(pat)
    return x[0]


def eqread(path):
    x=pd.read_csv(path); x['trade_date']=pd.to_datetime(x.trade_date); return x


def stats(eq):
    s=v1.series_from_eq(eq); f=v1.perf_series(s); tr=v1.perf_series(s,v1.START,v1.TRAIN_END); ps=v1.perf_series(s,v1.PSEUDO,v1.END)
    return {**f,'train_cagr':tr['cagr'],'train_mdd':tr['max_drawdown'],'train_sharpe':tr['sharpe'],'train_calmar':tr['calmar'],'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['max_drawdown'],'pseudo_sharpe':ps['sharpe']}


def pick_global(kind):
    fs=glob.glob(f'shard_artifacts/multi-alpha-{kind}-*-v2/**/grid.csv',recursive=True)
    if not fs: raise FileNotFoundError(f'no {kind} grids')
    d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
    d.to_csv(OUT/f'{kind}_grid_all.csv',index=False)
    ok=d[(d.train_cagr>0)&(d.train_mdd>-0.45)].copy()
    if len(ok)==0: ok=d.copy()
    win=ok.sort_values(['train_calmar','train_sharpe'],ascending=[False,False],kind='stable').iloc[0]
    return d,win


def family_dir(kind,fam):
    return Path(one(f'shard_artifacts/multi-alpha-{kind}-{fam}-v2/**/metadata.json')).parent


def correlations(eqs,bm):
    r={k:v1.series_from_eq(e).pct_change() for k,e in eqs.items()}
    R=pd.concat(r,axis=1).dropna(how='all').fillna(0.0)
    train=R.loc[v1.START:v1.TRAIN_END].corr(); pseudo=R.loc[v1.PSEUDO:v1.END].corr(); full=R.corr()
    br=bm.pct_change(fill_method=None).reindex(R.index); down=R.loc[br<0].corr()
    roll=[]; names=list(eqs)
    for i,a in enumerate(names):
      for b in names[i+1:]:
        z=R[a].rolling(252,min_periods=126).corr(R[b]).dropna()
        if len(z): roll.append({'pair':f'{a}-{b}','median':z.median(),'p10':z.quantile(.10),'p90':z.quantile(.90),'min':z.min(),'max':z.max()})
    return R,train,pseudo,full,down,pd.DataFrame(roll)


def annual(eqs):
    rows=[]
    for name,e in eqs.items():
        s=v1.series_from_eq(e)
        for y,g in s.groupby(s.index.year):
            before=s[s.index<pd.Timestamp(f'{y}-01-01')]
            st=float(before.iloc[-1]) if len(before) else float(g.iloc[0])
            rows.append({'sleeve':name,'year':int(y),'return':float(g.iloc[-1]/st-1)})
    return pd.DataFrame(rows)


def main():
    sd,sw=pick_global('short'); ld,lw=pick_global('long')
    short_key=str(sw.key); long_key=str(lw.key); sfam=str(sw.family); lfam=str(lw.family)
    sdir=family_dir('short',sfam); ldir=family_dir('long',lfam)
    mdir=Path(one('shard_artifacts/multi-alpha-medium-v2/**/metadata.json')).parent

    # Assert selected candidate is the train-only family winner saved by its shard.
    saved_s=str(pd.read_csv(sdir/'winner.csv').iloc[0].key); saved_l=str(pd.read_csv(ldir/'winner.csv').iloc[0].key)
    if saved_s!=short_key: raise RuntimeError(f'short winner mismatch {saved_s} {short_key}')
    if saved_l!=long_key: raise RuntimeError(f'long winner mismatch {saved_l} {long_key}')

    eqs={
      'short':eqread(sdir/'winner_equity_cost1.csv'),
      'medium':eqread(mdir/'equity_cost1.csv'),
      'long':eqread(ldir/'winner_equity_cost1.csv'),
    }
    sleeve_rows=[]; standalone={}
    for name,e in eqs.items():
        st=stats(e); sleeve_rows.append({'sleeve':name,**st}); standalone[name]=st
    pd.DataFrame(sleeve_rows).to_csv(OUT/'selected_sleeves.csv',index=False)

    bpath=one('shard_artifacts/multi-alpha-panel-v2/**/benchmark.csv')
    bdf=pd.read_csv(bpath,index_col=0); bdf.index=pd.to_datetime(bdf.index); bm=pd.to_numeric(bdf.iloc[:,0],errors='coerce').dropna()
    R,ctr,cp,cf,cd,roll=correlations(eqs,bm)
    R.to_csv(OUT/'selected_daily_returns.csv'); ctr.to_csv(OUT/'correlation_train.csv'); cp.to_csv(OUT/'correlation_pseudo.csv'); cf.to_csv(OUT/'correlation_full.csv'); cd.to_csv(OUT/'correlation_down_days.csv'); roll.to_csv(OUT/'rolling_corr_252.csv',index=False)

    rows=[]; pcache={}
    for aname,w in v1.ALLOCATIONS.items():
        pe=rt.weighted_mix([eqs['short'],eqs['medium'],eqs['long']],list(w)); st=stats(pe)
        ref=sum(wi*standalone[n]['train_cagr'] for wi,n in zip(w,['short','medium','long']))
        rows.append({'allocation':aname,'w_short':w[0],'w_medium':w[1],'w_long':w[2],**st,'weighted_train_cagr_reference':ref,'constraint_pass':int(st['train_cagr']>=ref-.02)})
        pcache[aname]=pe
    ad=pd.DataFrame(rows); ad.to_csv(OUT/'allocation_grid.csv',index=False)
    ok=ad[ad.constraint_pass==1].copy();
    if len(ok)==0: ok=ad.copy()
    aw=ok.sort_values(['train_calmar','train_sharpe'],ascending=[False,False],kind='stable').iloc[0]
    akey=str(aw.allocation); weights=v1.ALLOCATIONS[akey]; portfolio=pcache[akey]
    portfolio.to_csv(OUT/'equity_portfolio.csv',index=False)
    for n,e in eqs.items(): e.to_csv(OUT/f'equity_{n}.csv',index=False)

    marg=[]; pst=stats(portfolio)
    names=['short','medium','long']
    for i,name in enumerate(names):
        keep=[j for j in range(3) if j!=i]; ww=np.array([weights[j] for j in keep],float); ww=ww/ww.sum(); kn=[names[j] for j in keep]
        ee=rt.weighted_mix([eqs[n] for n in kn],ww.tolist()); ss=stats(ee)
        marg.append({'removed':name,'portfolio_cagr':pst['cagr'],'without_cagr':ss['cagr'],'delta_cagr':pst['cagr']-ss['cagr'],'portfolio_sharpe':pst['sharpe'],'without_sharpe':ss['sharpe'],'delta_sharpe':pst['sharpe']-ss['sharpe'],'portfolio_mdd':pst['max_drawdown'],'without_mdd':ss['max_drawdown']})
    md=pd.DataFrame(marg); md.to_csv(OUT/'marginal_contribution.csv',index=False)

    # Cost stress uses each selected sleeve rerun at the same frozen parameters.
    crows=[]
    for cm in (2,4):
        ceqs={'short':eqread(sdir/f'winner_equity_cost{cm}.csv'),'medium':eqread(mdir/f'equity_cost{cm}.csv'),'long':eqread(ldir/f'winner_equity_cost{cm}.csv')}
        pe=rt.weighted_mix([ceqs[n] for n in names],list(weights)); ss=stats(pe); crows.append({'cost_mult':cm,**ss})
    cdf=pd.DataFrame(crows); cdf.to_csv(OUT/'allocation_cost_stress.csv',index=False)

    annual({**eqs,'portfolio':portfolio}).to_csv(OUT/'annual_selected.csv',index=False)

    med=standalone['medium']; mincorr=min(float(cf.loc['short','medium']),float(cf.loc['long','medium']))
    gates={
      'all_sleeves_pseudo_positive':int(all(standalone[n]['pseudo_cagr']>0 for n in names)),
      'portfolio_pseudo_positive':int(pst['pseudo_cagr']>0),
      'portfolio_train_calmar_gt_medium':int(pst['train_calmar']>med['train_calmar']),
      'portfolio_mdd_not_gt_medium_by_5pp':int(pst['max_drawdown']>=med['max_drawdown']-.05),
      'one_nonmedium_corr_le_060':int(mincorr<=.60),
      'no_sleeve_removal_sharpe_improves_gt_010':int((md.without_sharpe<=md.portfolio_sharpe+.10).all()),
      'cost2_positive_cagr':int(float(cdf[cdf.cost_mult==2].cagr.iloc[0])>0),
    }
    gd=pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]); gd.to_csv(OUT/'gates.csv',index=False)
    status='PROMOTED_SERIOUS_SHADOW_CANDIDATE' if sum(gates.values())==len(gates) else 'NOT_PROMOTED'
    meta={'status':status,'research_label':'NEW_STOCK_LEVEL_MULTI_ALPHA_RESEARCH_NOT_ORIGINAL_EXACT','prereg_committed_before_search':'MULTI_ALPHA_PREREG_2026-09-03.md','short_selected_train_only':short_key,'medium_fixed':'mom_cfo10_qv10|H60|75%N10+25%N5|phases0_4_8','long_selected_train_only':long_key,'allocation_selected_train_only':akey,'weights':{'short':weights[0],'medium':weights[1],'long':weights[2]},'pseudo_oos_warning':'2022-2026 is research-contaminated diagnostic, not clean OOS','gates_passed':int(sum(gates.values())),'gates_total':len(gates)}
    (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
    summary={'portfolio':pst,'sleeves':standalone,'correlation_full':cf.to_dict(),'allocation':akey,'weights':meta['weights'],'cost_stress':cdf.to_dict(orient='records'),'gates':gates,'status':status}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
    print('SHORT',short_key,flush=True); print('LONG',long_key,flush=True); print(pd.DataFrame(sleeve_rows).to_string(index=False),flush=True); print('CORR\n',cf.to_string(),flush=True); print('ALLOCATIONS\n',ad.to_string(index=False),flush=True); print('SELECTED',akey,weights,flush=True); print('PORTFOLIO',pst,flush=True); print('COST\n',cdf.to_string(index=False),flush=True); print('GATES\n',gd.to_string(index=False),flush=True); print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
