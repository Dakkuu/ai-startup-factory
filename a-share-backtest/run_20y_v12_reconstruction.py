from __future__ import annotations
import os, tarfile, urllib.request, json
from pathlib import Path
import numpy as np
import pandas as pd

TAG=os.getenv('QLIB_RELEASE_TAG','2026-07-29')
ROOT=Path('qlib_data_v12_20y')
OUT=Path('results_v12_20y_reconstruction'); OUT.mkdir(exist_ok=True)
ANCHOR=pd.Timestamp('2016-07-29')
REPORT_START=pd.Timestamp('2006-01-04')
REPORT_END=pd.Timestamp('2026-07-29')
REBALANCE=90
NMAX=10
GATE_Q=.55
ENTRY_Q=.90
KEEP_Q=.70
ONEWAY_COST=0.00155
ANNUAL_DRAG=.003

RECON_SPEC={
  'universe':'pit_all_txt_no_price_floor',
  'history_min_sessions':126,
  'efficiency':'abs(sum(residual60))/sum(abs(residual60)) vs CSI300; gate <=55pct',
  'momentum':'raw adjusted 6-1 log return t-21/t-126',
  'rank_scope':'pre_gate eligible universe',
  'weights':{'low_downside':.15,'low_resvol':.25,'momentum':.35,'trend_t':.25},
  'rebalance_sessions':90,'anchor_signal_date':'2016-07-29','max_names':10,
  'entry_top':.10,'retain_top':.30,'fixed_denominator':10,
  'base_effective_lag_sessions':2,
  'oneway_cost_proxy':ONEWAY_COST,
  'risk_overlay':'exact recovered V12-RC return-stream engine'
}
REFERENCE_RAW={
 'period':['2016-08-02','2026-07-29'],'cagr':0.22650844794073488,'mdd':-0.41800289801993573,
 'sharpe':1.0507753133872526,'total_return':6.684775311490827,
 'annual':{'2016':-0.0024090575383427515,'2017':0.6879143948394355,'2018':-0.3088409792839253,
 '2019':0.16134374438089116,'2020':1.0508121036610651,'2021':0.38695880662670223,
 '2022':-0.02214903411709035,'2023':0.1085247533429099,'2024':0.278475358398659,
 '2025':0.3162160424439737,'2026':0.09587861904940587}
}

RIDGE_ALPHA=1.0; FORECAST_HORIZON=20; REFIT_EVERY=21; MODEL_START=300; MIN_TRAIN=200
HAR_THRESHOLD=.26; TREND_STRESS=-.03
CONFIRM_DAYS=10; IMPLEMENTATION_LAG=1; MAX_EXPOSURE_CHANGE=.25; TRANSITION_COST=.003


def download_extract():
    if (ROOT/'calendars/day.txt').exists(): return
    ROOT.mkdir(exist_ok=True); arc=Path('qlib_v12_20y.tar.gz')
    url=f'https://github.com/chenditc/investment_data/releases/download/{TAG}/qlib_bin.tar.gz'
    print('DOWNLOAD',url,flush=True); urllib.request.urlretrieve(url,arc)
    with tarfile.open(arc,'r:gz') as tf:
        members=tf.getmembers(); tops={Path(m.name).parts[0] for m in members if Path(m.name).parts}
        if len(tops)==1:
            for m in members:
                parts=Path(m.name).parts
                if len(parts)<=1: continue
                m.name=str(Path(*parts[1:])); tf.extract(m,ROOT)
        else: tf.extractall(ROOT)
    arc.unlink(missing_ok=True)
    if not (ROOT/'calendars/day.txt').exists():
        hits=list(ROOT.rglob('calendars/day.txt'))
        if len(hits)!=1: raise RuntimeError(f'calendar hits={hits}')
        b=hits[0].parent.parent
        for child in b.iterdir():
            tgt=ROOT/child.name
            if not tgt.exists(): child.rename(tgt)


def read_field(folder, field, cal, s, e):
    out=np.full(e-s+1,np.nan,dtype=np.float32); p=folder/f'{field}.day.bin'
    if not p.exists(): return out
    a=np.fromfile(p,dtype='<f4')
    if len(a)<2:return out
    st=int(a[0]); v=a[1:]; lo=max(s,st); hi=min(e,st+len(v)-1)
    if hi>=lo: out[lo-s:hi-s+1]=v[lo-st:hi-st+1]
    return out


def pct_rank(v, mask, ascending=True):
    out=np.full(len(v),np.nan); ids=np.where(mask & np.isfinite(v))[0]
    if not len(ids): return out
    order=np.lexsort((ids,v[ids])); ranks=np.empty(len(ids),float); ranks[order]=np.arange(1,len(ids)+1)
    p=ranks/len(ids)
    if not ascending:p=1-p+1/len(ids)
    out[ids]=p; return out


def trend_t(logp):
    w=90; n=logp.shape[1]; out=np.full(n,np.nan)
    if logp.shape[0]<w:return out
    y=logp[-w:].astype(float); good=np.all(np.isfinite(y),axis=0)
    if not np.any(good):return out
    x=np.arange(w,dtype=float); xc=x-x.mean(); sxx=np.sum(xc*xc)
    yg=y[:,good]; ym=yg.mean(axis=0); slope=np.sum(xc[:,None]*(yg-ym),axis=0)/sxx
    resid=yg-(ym[None,:]+slope[None,:]*xc[:,None]); se=np.sqrt((np.sum(resid*resid,axis=0)/(w-2))/sxx)
    tv=np.divide(slope,se,out=np.full_like(slope,np.nan),where=se>1e-12); out[good]=tv
    return out


def residual_stats(R, M, minobs=50):
    n=R.shape[1]; res=np.full_like(R,np.nan,dtype=float)
    for j in range(n):
        y=R[:,j]; g=np.isfinite(y)&np.isfinite(M)
        if g.sum()<minobs:continue
        yy=y[g]; mm=M[g]; mc=mm-mm.mean(); vm=np.sum(mc*mc)
        b=np.sum(mc*(yy-yy.mean()))/vm if vm>1e-12 else 0.0; a=yy.mean()-b*mm.mean()
        res[g,j]=yy-(a+b*mm)
    return res


def feature_snapshot(close, ret, mret, i, member, first_valid):
    N=close.shape[1]; hist_ok=(first_valid>=0)&((i-first_valid+1)>=126)
    eligible=member & hist_ok & np.isfinite(close[i]) & (close[i]>0)
    mom=np.full(N,np.nan)
    if i>=126:
        a=close[i-21].astype(float); b=close[i-126].astype(float); g=np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)
        mom[g]=np.log(a[g]/b[g])
    y=close[max(0,i-89):i+1]
    tm=trend_t(np.log(np.where(y>0,y,np.nan)))
    R=ret[max(0,i-59):i+1].astype(float); M=mret[max(0,i-59):i+1].astype(float)
    if R.shape[0]<60:
        nan=np.full(N,np.nan); return eligible,nan,nan
    res=residual_stats(R,M,50); cnt=np.sum(np.isfinite(res),axis=0)
    resv=np.nanstd(res,axis=0,ddof=0); resv[cnt<50]=np.nan
    down=np.sqrt(np.nanmean(np.minimum(R,0.0)**2,axis=0)); down[np.sum(np.isfinite(R),axis=0)<50]=np.nan
    ss=np.nansum(res,axis=0); aa=np.nansum(np.abs(res),axis=0)
    eff=np.divide(np.abs(ss),aa,out=np.full(N,np.nan),where=(aa>1e-12)&(cnt>=50))
    basegood=eligible&np.isfinite(mom)&np.isfinite(tm)&np.isfinite(resv)&np.isfinite(down)&np.isfinite(eff)
    if basegood.sum()<20:return basegood,np.full(N,np.nan),np.full(N,np.nan)
    q=np.nanquantile(eff[basegood],GATE_Q); gate=basegood&(eff<=q)
    rm=pct_rank(mom,basegood,True); rt=pct_rank(tm,basegood,True); rv=pct_rank(resv,basegood,False); rd=pct_rank(down,basegood,False)
    score=np.full(N,np.nan); ok=gate&np.isfinite(rm)&np.isfinite(rt)&np.isfinite(rv)&np.isfinite(rd)
    score[ok]=.35*rm[ok]+.25*rt[ok]+.25*rv[ok]+.15*rd[ok]
    sr=pct_rank(score,basegood,True)
    return ok,score,sr


def membership_mask(intervals, codes, d):
    out=np.zeros(len(codes),dtype=bool)
    for j,c in enumerate(codes):
        arr=intervals.get(c)
        if arr: out[j]=any(a<=d<=b for a,b in arr)
    return out


def select_target(ok,score,sr,codes,prev):
    code_to_i={c:i for i,c in enumerate(codes)}; incumb=[]
    for c in prev:
        j=code_to_i.get(c)
        if j is not None and ok[j] and np.isfinite(sr[j]) and sr[j]>=KEEP_Q: incumb.append(j)
    incumb=sorted(set(incumb),key=lambda j:(score[j],codes[j]),reverse=True)[:NMAX]; chosen=list(incumb)
    ent=np.where(ok&np.isfinite(sr)&(sr>=ENTRY_Q))[0]; ent=sorted(ent,key=lambda j:(score[j],codes[j]),reverse=True)
    for j in ent:
        if j not in chosen:chosen.append(j)
        if len(chosen)>=NMAX:break
    return [codes[j] for j in chosen]


def build_signal_targets(close,ret,mret,cal,codes,intervals,first_valid):
    anchor=int(cal.get_loc(ANCHOR)); idx=[]; k=0
    while anchor-k*REBALANCE>=126: idx.append(anchor-k*REBALANCE); k+=1
    k=1
    while anchor+k*REBALANCE<len(cal): idx.append(anchor+k*REBALANCE); k+=1
    idx=sorted(idx); prev=[]; rows=[]; targets={}
    for z,i in enumerate(idx):
        d=cal[i]; member=membership_mask(intervals,codes,d); ok,score,sr=feature_snapshot(close,ret,mret,i,member,first_valid)
        target=select_target(ok,score,sr,codes,prev); targets[i]=target
        rows.append({'signal_date':d.date(),'signal_index':i,'member_n':int(member.sum()),'eligible_n':int(np.sum(ok)),'target_n':len(target),'target_codes':';'.join(target)})
        prev=target
        if z%10==0:print('SIGNAL',d.date(),'members',member.sum(),'eligible',np.sum(ok),'target',len(target),flush=True)
    return targets,pd.DataFrame(rows)


def core_stream(ret,cal,codes,targets,effective_lag=2,oneway_cost=0.0):
    code_to_i={c:i for i,c in enumerate(codes)}; sig=sorted(targets); raw=np.zeros(len(cal),float); turn=np.zeros(len(cal),float); prev=[]
    for k,s in enumerate(sig):
        tgt=targets[s]; start=s+effective_lag; end=(sig[k+1]+effective_lag) if k+1<len(sig) else len(cal)
        ids=[code_to_i[c] for c in tgt if c in code_to_i]
        for t in range(start,min(end,len(cal))):
            if ids:
                rr=ret[t,ids].astype(float); raw[t]=np.nansum(np.where(np.isfinite(rr),rr,0.0))/NMAX
        old={c:1/NMAX for c in prev}; new={c:1/NMAX for c in tgt}; tv=sum(abs(new.get(c,0)-old.get(c,0)) for c in set(old)|set(new))
        if start<len(cal):turn[start]=tv
        prev=tgt
    cost=turn*oneway_cost
    return raw-cost,turn,cost


def realized_vol_features(x,dates):
    s=pd.Series(x,index=dates); cols=[]
    for L in (20,60,120):
        v=s.rolling(L,min_periods=L).std().shift(1)*np.sqrt(252); cols.append(np.log(np.maximum(v.to_numpy(float),1e-6)))
    return np.column_stack(cols)


def future_vol_target(x,h=20):
    y=np.full(len(x),np.nan)
    for t in range(len(x)-h+1):y[t]=np.std(x[t:t+h],ddof=1)*np.sqrt(252)
    return y


def causal_har(x,dates):
    F=realized_vol_features(x,dates);y=future_vol_target(x);n=len(x);p=np.full(n,np.nan);valid=np.all(np.isfinite(F),axis=1)&np.isfinite(y)
    for u in range(MODEL_START,n,REFIT_EVERY):
        idx=np.where(valid&(np.arange(n)<=u-FORECAST_HORIZON))[0]
        if len(idx)<MIN_TRAIN:continue
        X=F[idx];Y=np.log(np.maximum(y[idx],1e-6));mu=X.mean(axis=0);sd=X.std(axis=0);sd[sd<1e-8]=1.;Z=(X-mu)/sd;ym=Y.mean()
        beta=np.linalg.solve(Z.T@Z+RIDGE_ALPHA*np.eye(3),Z.T@(Y-ym));end=min(u+REFIT_EVERY,n);Q=F[u:end];ok=np.all(np.isfinite(Q),axis=1);z=np.full(end-u,np.nan);z[ok]=np.exp(ym+((Q[ok]-mu)/sd)@beta);p[u:end]=z
    return p


def prior_trend(x,dates):
    s=pd.Series(x,index=dates);return ((1+s).rolling(60,min_periods=60).apply(np.prod,raw=True).sub(1).shift(1).to_numpy(float))


def overlay(raw,dates):
    fac=(1-ANNUAL_DRAG)**(1/252);core=(1+raw)*fac-1;pred=causal_har(core,dates);tr=prior_trend(core,dates);n=len(core)
    desired=np.empty(n);state=1.;candidate=1.;count=0
    for t in range(n):
        if not np.isfinite(pred[t]):q=state
        elif pred[t]<=HAR_THRESHOLD:q=1.
        elif np.isfinite(tr[t]) and tr[t]<=TREND_STRESS:q=.25
        else:q=.5
        if q==state:candidate,count=state,0
        elif q==candidate:count+=1
        else:candidate,count=q,1
        if q!=state and count>=CONFIRM_DAYS:state,count=candidate,0
        desired[t]=state
    live=np.empty(n);net=np.empty(n);live[0]=1.;net[0]=core[0]
    for t in range(1,n):
        target=desired[t-1];delta=np.clip(target-live[t-1],-MAX_EXPOSURE_CHANGE,MAX_EXPOSURE_CHANGE);live[t]=live[t-1]+delta;net[t]=live[t]*core[t]-abs(delta)*TRANSITION_COST
    return core,pred,tr,desired,live,net


def metrics(r,dates):
    r=np.asarray(r,float);dates=pd.DatetimeIndex(dates);nav=np.cumprod(1+r);years=max((dates[-1]-dates[0]).days/365.2425,1/365.2425);sd=np.std(r,ddof=1);dn=np.sqrt(np.mean(np.minimum(r,0.)**2))
    return {'total_return':float(nav[-1]-1),'cagr':float(nav[-1]**(1/years)-1),'max_drawdown':float(np.min(nav/np.maximum.accumulate(nav)-1)),'sharpe':float(np.sqrt(252)*np.mean(r)/sd) if sd>0 else None,'sortino':float(np.sqrt(252)*np.mean(r)/dn) if dn>0 else None,'ending_nav':float(nav[-1])}


def annual(r,dates):
    dates=pd.DatetimeIndex(dates);return {int(y):float(np.prod(1+np.asarray(r)[dates.year==y])-1) for y in sorted(np.unique(dates.year))}


def evaluate_variant(raw,turn,cost,dates,name):
    core,pred,tr,desired,live,net=overlay(raw,dates);m=metrics(net,dates);m.update({'name':name,'avg_exposure':float(np.mean(live)),'exposure_adjustment_days':int(np.sum(np.diff(live)!=0)),'turnover':float(np.sum(turn)),'explicit_turnover_cost':float(np.sum(cost))})
    return m,pd.DataFrame({'trade_date':dates,'raw_core_return':raw,'net_core_return':core,'har_pred_future20_vol':pred,'trend60_prior_close':tr,'desired_state':desired,'live_exposure':live,'net_return':net,'nav':np.cumprod(1+net),'turnover':turn,'turnover_cost':cost})


def main():
    download_extract();fullcal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ROOT/'calendars/day.txt',header=None)[0]));s=0;e=int(fullcal.searchsorted(REPORT_END,side='right')-1);cal=fullcal[s:e+1]
    dirs=sorted([p for p in (ROOT/'features').iterdir() if p.is_dir() and p.name.startswith(('sh6','sz0','sz3','bj'))]);codes=np.array([p.name.upper() for p in dirs]);T=len(cal);N=len(codes)
    close=np.full((T,N),np.nan,dtype=np.float32)
    for j,p in enumerate(dirs):
        close[:,j]=read_field(p,'close',fullcal,s,e)
        if j%1000==0:print('LOAD',j,N,flush=True)
    ret=np.full_like(close,np.nan,dtype=np.float32);ret[1:]=close[1:]/close[:-1]-1
    mkt=read_field(ROOT/'features/sh000300','close',fullcal,s,e).astype(float);mret=np.full(T,np.nan);mret[1:]=mkt[1:]/mkt[:-1]-1
    first_valid=np.full(N,-1,int)
    for j in range(N):
        g=np.where(np.isfinite(close[:,j])&(close[:,j]>0))[0]
        if len(g):first_valid[j]=g[0]
    inst=pd.read_csv(ROOT/'instruments/all.txt',sep='\t',header=None,names=['code','start','end'],usecols=[0,1,2]);inst.code=inst.code.astype(str).str.upper();inst.start=pd.to_datetime(inst.start);inst.end=pd.to_datetime(inst.end)
    intervals={}
    for r in inst.itertuples():intervals.setdefault(r.code,[]).append((r.start,r.end))
    targets,holdings=build_signal_targets(close,ret,mret,cal,codes,intervals,first_valid);holdings.to_csv(OUT/'rebalance_holdings.csv',index=False)
    mask=(cal>=REPORT_START)&(cal<=REPORT_END);dates=cal[mask];summaries=[];all_daily={}
    for lag in (2,3,4):
      for cmult in (0.,1.,2.):
        raw0,turn0,cost0=core_stream(ret,cal,codes,targets,effective_lag=lag,oneway_cost=ONEWAY_COST*cmult);raw=raw0[mask];turn=turn0[mask];cost=cost0[mask];name=f'lag{lag}_cost{cmult:g}x'
        met,df=evaluate_variant(raw,turn,cost,dates,name);summaries.append(met)
        if name in ('lag2_cost0x','lag2_cost1x','lag2_cost2x','lag3_cost1x','lag4_cost1x'):df.to_csv(OUT/f'daily_{name}.csv',index=False);all_daily[name]=df
    pd.DataFrame(summaries).to_csv(OUT/'variant_metrics.csv',index=False)
    primary=next(x for x in summaries if x['name']=='lag2_cost1x');pdf=all_daily['lag2_cost1x'];ann=annual(pdf.net_return.to_numpy(),pd.DatetimeIndex(pdf.trade_date));pd.DataFrame([{'year':y,'return':v} for y,v in ann.items()]).to_csv(OUT/'annual_primary.csv',index=False)
    seg=[]
    for nm,a,b in [('pre2016_extension','2006-01-04','2016-07-28'),('old_research_overlap','2016-08-02','2026-07-29'),('full','2006-01-04','2026-07-29')]:
        z=pdf[(pdf.trade_date>=a)&(pdf.trade_date<=b)]
        if len(z)>2:
            q=metrics(z.net_return.to_numpy(),pd.DatetimeIndex(z.trade_date));q.update({'segment':nm,'start':a,'end':b});seg.append(q)
    pd.DataFrame(seg).to_csv(OUT/'segments_primary.csv',index=False)
    r0=all_daily['lag2_cost0x'];ov=r0[(r0.trade_date>='2016-08-02')&(r0.trade_date<='2026-07-29')];rm=metrics(ov.raw_core_return.to_numpy(),pd.DatetimeIndex(ov.trade_date));ra=annual(ov.raw_core_return.to_numpy(),pd.DatetimeIndex(ov.trade_date));yrs=sorted(set(ra)&set(int(k) for k in REFERENCE_RAW['annual']));x=np.array([ra[y] for y in yrs]);y=np.array([REFERENCE_RAW['annual'][str(y)] for y in yrs]);corr=float(np.corrcoef(x,y)[0,1]) if len(yrs)>1 else None
    comp={'reconstruction_raw_metrics':rm,'reference_raw_metrics':{k:v for k,v in REFERENCE_RAW.items() if k!='annual'},'annual_return_correlation':corr,'reconstructed_annual':ra,'reference_annual':REFERENCE_RAW['annual'],'selection_note':'audit only; 2006-2015 and compatibility outputs never alter frozen reconstruction spec'}
    (OUT/'compatibility_audit.json').write_text(json.dumps(comp,indent=2,ensure_ascii=False),encoding='utf-8')
    integ={'tag':TAG,'calendar_start':str(cal.min().date()),'calendar_end':str(cal.max().date()),'symbols':int(N),'anchor_present':bool(ANCHOR in cal),'report_sessions':int(mask.sum()),'recon_spec':RECON_SPEC,'limitations':['no historical dynamic ST table','no exact historical stk_limit table','suspension inferred only through missing price returns','no exact corporate-action share/lot ledger','Qlib adjusted close return proxy rather than exact broker fills'],'classification':'Tier-B reconstructed diagnostic, not exact V12 extension and not clean OOS'}
    (OUT/'data_integrity.json').write_text(json.dumps(integ,indent=2,ensure_ascii=False),encoding='utf-8')
    report=f'''# V12-R 20Y frozen-rule reconstruction\n\nStatus: **Tier-B reconstructed diagnostic; NOT exact V12 extension; NOT clean OOS.**\n\nThe implementation fingerprint was frozen from 2016-2026 archived holdings before any 2006-2015 performance was read. The older period is a reverse-time extrapolation stress, not conventional OOS.\n\nPrimary ex-ante diagnostic convention: lag2, 15.5bp one-way turnover proxy, plus the exact V12 30bp/year drag and exact recovered HAR overlay.\n\nPrimary full-period metrics: {json.dumps(primary,ensure_ascii=False)}\n\n2016-2026 raw-core compatibility audit: reconstructed CAGR={rm['cagr']:.4%}, old exact raw core CAGR={REFERENCE_RAW['cagr']:.4%}, annual-return correlation={corr if corr is not None else float('nan'):.4f}.\n'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8');print('PRIMARY',json.dumps(primary,indent=2),flush=True);print('COMPAT',json.dumps(comp,indent=2,default=str)[:5000],flush=True)

if __name__=='__main__':main()
