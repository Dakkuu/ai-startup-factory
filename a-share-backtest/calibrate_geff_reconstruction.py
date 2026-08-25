from __future__ import annotations
import os, tarfile, urllib.request, math, json
from pathlib import Path
import numpy as np
import pandas as pd

TAG=os.getenv('QLIB_RELEASE_TAG','2026-07-29')
ROOT=Path('qlib_data_cal')
OUT=Path('results_geff_calibration'); OUT.mkdir(exist_ok=True)
TARGETS=pd.read_csv('geff_phase0_actual_holdings.csv',parse_dates=['signal_date'])
TARGETS['aset']=TARGETS.actual_holdings.fillna('').map(lambda s:set(str(s).split(';')) if s else set())


def download_extract():
    if (ROOT/'calendars/day.txt').exists(): return
    ROOT.mkdir(exist_ok=True); arc=Path('qlib_cal.tar.gz')
    url=f'https://github.com/chenditc/investment_data/releases/download/{TAG}/qlib_bin.tar.gz'
    urllib.request.urlretrieve(url,arc)
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
        hit=list(ROOT.rglob('calendars/day.txt'))
        if len(hit)!=1: raise RuntimeError(hit)
        b=hit[0].parent.parent
        for c in b.iterdir():
            t=ROOT/c.name
            if not t.exists(): c.rename(t)


def read_field(folder:Path,field:str,cal:pd.DatetimeIndex,start_ix:int,end_ix:int):
    p=folder/f'{field}.day.bin'; out=np.full(end_ix-start_ix+1,np.nan,dtype=np.float32)
    if not p.exists(): return out
    a=np.fromfile(p,dtype='<f4')
    if len(a)<2:return out
    st=int(a[0]); v=a[1:]; lo=max(start_ix,st); hi=min(end_ix,st+len(v)-1)
    if hi>=lo: out[lo-start_ix:hi-start_ix+1]=v[lo-st:hi-st+1]
    return out


def pct_rank(v,mask,ascending=True):
    out=np.full(len(v),np.nan); ids=np.where(mask & np.isfinite(v))[0]
    if len(ids)==0:return out
    order=np.lexsort((ids,v[ids])); r=np.empty(len(ids),float); r[order]=np.arange(1,len(ids)+1)
    p=r/len(ids)
    if not ascending:p=1-p+1/len(ids)
    out[ids]=p; return out


def trend_t(logp,w=90):
    out=np.full(logp.shape[1],np.nan); x=np.arange(w,dtype=float); xc=x-x.mean(); sxx=np.sum(xc*xc)
    y=logp[-w:,:].astype(float); good=np.all(np.isfinite(y),axis=0)
    if not np.any(good):return out
    yg=y[:,good]; ym=yg.mean(axis=0); slope=np.sum(xc[:,None]*(yg-ym),axis=0)/sxx
    resid=yg-(ym[None,:]+slope[None,:]*xc[:,None]); se=np.sqrt((np.sum(resid*resid,axis=0)/(w-2))/sxx)
    t=np.divide(slope,se,out=np.full_like(slope,np.nan),where=se>1e-12); out[good]=t; return out


def beta_residual(R,M,minobs):
    n=R.shape[1]; out=np.full_like(R,np.nan,dtype=float); bet=np.full(n,np.nan)
    for j in range(n):
        y=R[:,j]; g=np.isfinite(y)&np.isfinite(M)
        if g.sum()<minobs:continue
        yy=y[g]; mm=M[g]; mc=mm-mm.mean(); vm=np.sum(mc*mc)
        b=np.sum(mc*(yy-yy.mean()))/vm if vm>1e-12 else 0.0; a=yy.mean()-b*mm.mean()
        out[g,j]=yy-(a+b*mm); bet[j]=b
    return out,bet


def features(close,rawclose,amount,mkt_close,member_mask,signal_i):
    c=close[:signal_i+1]; rc=rawclose[:signal_i+1]; am=amount[:signal_i+1]; mc=mkt_close[:signal_i+1]
    valid=np.isfinite(c)&(c>0); hist=np.sum(valid,axis=0); logp=np.log(np.where(c>0,c,np.nan))
    ret=np.full(c.shape,np.nan,dtype=float); ret[1:]=c[1:]/c[:-1]-1
    mret=np.full(len(mc),np.nan,dtype=float); mret[1:]=mc[1:]/mc[:-1]-1
    mom=np.full(c.shape[1],np.nan)
    if signal_i>=126:
        a=c[-22]; b=c[-127]; g=np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0); mom[g]=np.log(a[g]/b[g])
    tm=trend_t(logp,90)
    R60=ret[-60:].astype(float); M60=mret[-60:].astype(float); resid60,_=beta_residual(R60,M60,50)
    resv=np.nanstd(resid60,axis=0,ddof=0)
    down=np.sqrt(np.nanmean(np.minimum(R60,0.0)**2,axis=0))
    Rform=ret[-126:-21].astype(float); Mform=mret[-126:-21].astype(float); resid_form,beta_form=beta_residual(Rform,Mform,80)
    reslog=np.where(resid_form>-0.999,np.log1p(resid_form),np.nan)
    resmom=np.nansum(reslog,axis=0); resmom[np.sum(np.isfinite(reslog),axis=0)<80]=np.nan
    L=min(252,len(ret)-1); Rb=ret[-L:].astype(float); Mb=mret[-L:].astype(float); _,beta252=beta_residual(Rb,Mb,min(126,L))
    res252=Rform-beta252[None,:]*Mform[:,None]
    res252log=np.where(res252>-0.999,np.log1p(res252),np.nan); resmom252=np.nansum(res252log,axis=0);resmom252[np.sum(np.isfinite(res252log),axis=0)<80]=np.nan
    mform_log=np.nansum(np.log1p(np.where(Mform>-0.999,Mform,np.nan)))
    resmom1=mom-mform_log
    eff={}
    for L in (20,60,90,126):
        y=logp[-(L+1):].astype(float); dif=np.diff(y,axis=0); num=np.abs(y[-1]-y[0]); den=np.nansum(np.abs(dif),axis=0);cnt=np.sum(np.isfinite(dif),axis=0)
        eff[f'er{L}']=np.divide(num,den,out=np.full(c.shape[1],np.nan),where=(den>1e-12)&(cnt>=L))
    y=logp[-127:-21].astype(float);dif=np.diff(y,axis=0);num=np.abs(y[-1]-y[0]);den=np.nansum(np.abs(dif),axis=0);cnt=np.sum(np.isfinite(dif),axis=0)
    eff['er6_1']=np.divide(num,den,out=np.full(c.shape[1],np.nan),where=(den>1e-12)&(cnt>=104))
    for nm,RR in [('res60',resid60),('res6_1',resid_form)]:
        ss=np.nansum(RR,axis=0); aa=np.nansum(np.abs(RR),axis=0); cc=np.sum(np.isfinite(RR),axis=0); need=50 if nm=='res60' else 80
        eff[nm]=np.divide(np.abs(ss),aa,out=np.full(c.shape[1],np.nan),where=(aa>1e-12)&(cc>=need))
    y=logp[-90:].astype(float);x=np.arange(90,dtype=float);xc=x-x.mean();sxx=np.sum(xc*xc);r2=np.full(c.shape[1],np.nan)
    for j in range(c.shape[1]):
        yy=y[:,j]
        if not np.all(np.isfinite(yy)):continue
        yc=yy-yy.mean();sst=np.sum(yc*yc)
        if sst<=1e-12:continue
        sl=np.sum(xc*yc)/sxx;r2[j]=np.clip(np.sum((sl*xc)**2)/sst,0,1)
    eff['r2_90']=r2
    am20=np.nanmean(am[-20:].astype(float),axis=0)
    return {'hist':hist,'rawclose':rc[-1].astype(float),'amount20':am20,'member':member_mask,'mom':mom,'resmom':resmom,'resmom252':resmom252,'resmom1':resmom1,'trend':tm,'resv':resv,'down':down,'eff':eff}


def select_one(F,codes,actual_prev,universe_name,eff_name,mom_name,rank_scope):
    hist=F['hist'];rp=F['rawclose'];am=F['amount20'];u=F['member'].copy()
    u&=np.isfinite(rp)&(rp>=2.0)&(hist>=126)
    if universe_name=='h252':u&=hist>=252
    elif universe_name=='liq10':u&=np.isfinite(am)&(am>=1e7)
    elif universe_name=='liq20':u&=np.isfinite(am)&(am>=2e7)
    elif universe_name=='liq50':u&=np.isfinite(am)&(am>=5e7)
    elif universe_name=='noprice':u=F['member']&(hist>=126)
    mv=F[mom_name]; basegood=u&np.isfinite(mv)&np.isfinite(F['trend'])&np.isfinite(F['resv'])&np.isfinite(F['down'])&np.isfinite(F['eff'][eff_name])
    if basegood.sum()<50:return set(),0,0
    q=np.nanquantile(F['eff'][eff_name][basegood],.55);gate=basegood&(F['eff'][eff_name]<=q);scope=gate if rank_scope=='gate' else basegood
    rm=pct_rank(mv,scope,True);rt=pct_rank(F['trend'],scope,True);rv=pct_rank(F['resv'],scope,False);rd=pct_rank(F['down'],scope,False)
    good=gate&np.isfinite(rm)&np.isfinite(rt)&np.isfinite(rv)&np.isfinite(rd)
    score=np.full(len(codes),np.nan);score[good]=.30*rv[good]+.20*rd[good]+.30*rm[good]+.20*rt[good]
    sr=pct_rank(score,good,True);code_to_i={c:i for i,c in enumerate(codes)}
    prev_ids=[code_to_i[c] for c in actual_prev if c in code_to_i and good[code_to_i[c]] and sr[code_to_i[c]]>=.70]
    prev_ids=sorted(prev_ids,key=lambda i:score[i],reverse=True)[:15];chosen=list(prev_ids)
    entrants=sorted(np.where(good&np.isfinite(sr)&(sr>=.90))[0],key=lambda i:score[i],reverse=True)
    for i in entrants:
        if i not in chosen:chosen.append(i)
        if len(chosen)>=15:break
    return set(codes[i] for i in chosen),int(basegood.sum()),int(gate.sum())


def main():
    download_extract();cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ROOT/'calendars/day.txt',header=None)[0]))
    first=TARGETS.signal_date.min()-pd.Timedelta(days=700);last=TARGETS.signal_date.max();s=int(cal.searchsorted(first));e=int(cal.searchsorted(last,side='right')-1);subcal=cal[s:e+1]
    inst=pd.read_csv(ROOT/'instruments/all.txt',sep='\t',header=None,names=['code','start','end'],usecols=[0,1,2]);inst.code=inst.code.astype(str).str.upper();inst.start=pd.to_datetime(inst.start);inst.end=pd.to_datetime(inst.end)
    dirs=sorted([p for p in (ROOT/'features').iterdir() if p.is_dir() and p.name.startswith(('sh6','sz0','sz3','bj'))]);codes=np.array([p.name.upper() for p in dirs]);T=len(subcal);N=len(codes)
    close=np.full((T,N),np.nan,dtype=np.float32);factor=np.full((T,N),np.nan,dtype=np.float32);amount=np.full((T,N),np.nan,dtype=np.float32)
    for j,p in enumerate(dirs):
        close[:,j]=read_field(p,'close',cal,s,e);factor[:,j]=read_field(p,'factor',cal,s,e);amount[:,j]=read_field(p,'amount',cal,s,e)
        if j%1000==0:print('LOAD',j,N,flush=True)
    raw=np.divide(close,factor,out=np.full_like(close,np.nan),where=np.isfinite(factor)&(factor>0));mkt=read_field(ROOT/'features/sh000300','close',cal,s,e)
    date_to_i={d:i for i,d in enumerate(subcal)};Fs={};aud=[];code_to_inst={r.code:(r.start,r.end) for r in inst.itertuples()}
    for d in TARGETS.signal_date:
        dd=pd.Timestamp(d);member=np.array([(c in code_to_inst and code_to_inst[c][0]<=dd<=code_to_inst[c][1]) for c in codes],dtype=bool);i=date_to_i[dd]
        Fs[dd]=features(close,raw,amount,mkt,member,i);aud.append({'date':dd.date(),'members':int(member.sum())});print('FEATURE',dd.date(),'members',member.sum(),flush=True)
    pd.DataFrame(aud).to_csv(OUT/'membership_check.csv',index=False)
    configs=[]
    for u in ['base','h252','liq10','liq20','liq50','noprice']:
      for ef in ['er20','er60','er90','er126','er6_1','res60','res6_1','r2_90']:
       for mom in ['mom','resmom','resmom252','resmom1']:
        for scope in ['gate','universe']:
         prev=set();js=[];rec=[];prec=[];exact=0;bg=[];gg=[]
         for _,r in TARGETS.iterrows():
            d=pd.Timestamp(r.signal_date);pred,b,g=select_one(Fs[d],codes,prev,u,ef,mom,scope);act=r.aset;inter=len(pred&act);union=len(pred|act)
            js.append(inter/union if union else 1);rec.append(inter/len(act) if act else 1);prec.append(inter/len(pred) if pred else 0);exact+=int(pred==act);bg.append(b);gg.append(g);prev=act
         configs.append({'universe':u,'eff':ef,'momentum':mom,'rank_scope':scope,'mean_jaccard':np.mean(js),'mean_recall':np.mean(rec),'mean_precision':np.mean(prec),'exact_sets':exact,'basegood_med':np.median(bg),'gate_med':np.median(gg)})
    res=pd.DataFrame(configs).sort_values(['mean_jaccard','mean_recall'],ascending=False);res.to_csv(OUT/'calibration_grid.csv',index=False);print('\nTOP CONFIGS\n',res.head(40).to_string(index=False),flush=True)
    best=res.iloc[0];prev=set();det=[]
    for _,r in TARGETS.iterrows():
        d=pd.Timestamp(r.signal_date);pred,b,g=select_one(Fs[d],codes,prev,best.universe,best.eff,best.momentum,best.rank_scope);act=r.aset;inter=len(act&pred);union=len(act|pred)
        det.append({'signal_date':d.date(),'actual_n':len(act),'pred_n':len(pred),'intersection':inter,'jaccard':inter/union if union else 1,'actual_only':';'.join(sorted(act-pred)),'pred_only':';'.join(sorted(pred-act))});prev=act
    pd.DataFrame(det).to_csv(OUT/'best_detail.csv',index=False);(OUT/'best.json').write_text(json.dumps(best.to_dict(),ensure_ascii=False,indent=2,default=float),encoding='utf-8')

if __name__=='__main__':main()
