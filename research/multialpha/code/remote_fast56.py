from pathlib import Path
import argparse,json,gc
import numpy as np
import pandas as pd

NF=56
FEAT=[f'F{i:02d}' for i in range(1,NF+1)]
REG={'S01':(3,1),'S02':(2,1),'S03':(20,5),'S04':(5,1),'S05':(5,1),'S06':(5,1),'S07':(20,5),'S08':(5,1),'S09':(3,1),'S10':(1,1),'S11':(10,5),'S12':(20,5),'S13':(5,5)}
SPECS={'S01':[('F10',.30),('F11',.20),('F36',.15),('F50',-.20),('F19',-.15)],'S02':[('F09',.25),('F10',.25),('F32',.20),('F29',.15),('F36',.15)],'S03':[('F51',.30),('F52',.25),('F55',.20),('F20',-.15),('F22',-.10)],'S04':[('F39',.30),('F25',.20),('F27',.15),('F33',.15),('F37',.20)],'S05':[('F06',.25),('F03',-.25),('F53',.20),('F37',.15),('F21',-.15)],'S06':[('F17',-.20),('F19',-.20),('F39',.25),('F25',.15),('F37',.20)],'S07':[('F20',-.25),('F22',-.20),('F06',.20),('F53',.20),('F55',.15)],'S08':[('F42',.25),('F43',.30),('F46',-.15),('F48',.15),('F51',.15)],'S09':[('F47',.30),('F48',.25),('F42',.20),('F45',-.15),('F10',.10)],'S10':[('F25',.25),('F37',.25),('F32',.20),('F33',.15),('F51',.15)],'S11':[('F39',.20),('F05',.15),('F06',.15),('F53',.20),('F55',.15),('F37',.15)],'S12':[('F14',-.25),('F20',-.20),('F22',-.20),('F32',-.10),('F06',.15),('F55',.10)]}

def load_data(root):
    parts=[]
    for p in sorted(Path(root).glob('kline_*.parquet')):
        q=pd.read_parquet(p,columns=['code','date','open','high','low','close','volume','amount','turn','pctChg']);parts.append(q);print('LOAD',p.name,len(q),flush=True)
    b=pd.concat(parts,ignore_index=True);del parts;gc.collect();b=b.rename(columns={'code':'ts_code','date':'trade_date','turn':'turnover_rate'});b['ts_code']=b.ts_code.astype(str).str.zfill(6);b['trade_date']=pd.to_datetime(b.trade_date)
    for c in ['open','high','low','close','volume','amount','turnover_rate']:b[c]=pd.to_numeric(b[c],errors='coerce').astype('float32')
    return b[b.close>0].sort_values(['ts_code','trade_date']).reset_index(drop=True)

def cs_z(X,dates):
    starts=np.r_[0,np.flatnonzero(dates[1:]!=dates[:-1])+1];ends=np.r_[starts[1:],len(dates)]
    for a,z in zip(starts,ends):
        q=X[a:z];med=np.nanmedian(q,axis=0);mad=np.nanmedian(np.abs(q-med),axis=0);X[a:z]=np.clip((q-med)/np.maximum(1.4826*mad,1e-6),-5,5)
    return starts,ends

def factors(b):
    n=len(b);X=np.full((n,NF),np.nan,np.float32);g=b.groupby('ts_code',sort=False,group_keys=False);close=b.close.astype(float);op=b.open.astype(float);high=b.high.astype(float);low=b.low.astype(float);vol=b.volume.astype(float);amt=b.amount.astype(float);turn=b.turnover_rate.astype(float);prev=g.close.shift(1).astype(float);ret=close/prev-1
    def put(i,s):X[:,i-1]=np.asarray(s,dtype=np.float32)
    rs={}
    for i,w in enumerate([1,3,5,10,20,60],1):rs[w]=g.close.pct_change(w,fill_method=None).astype(float);put(i,rs[w])
    put(7,g.close.shift(5).astype(float)/g.close.shift(20).astype(float)-1);put(8,g.close.shift(10).astype(float)/g.close.shift(60).astype(float)-1);put(9,-rs[1]);put(10,-rs[3]);put(11,-rs[5])
    def rollret(w,kind):
        arr=[]
        for _,x in b.groupby('ts_code',sort=False):
            s=x.close.astype(float)/x.close.astype(float).shift(1)-1;arr.append(getattr(s.rolling(w,min_periods=w),kind)())
        return pd.concat(arr).sort_index()
    put(12,rollret(5,'max'));put(13,rollret(5,'min'));put(14,rollret(20,'max'));put(15,rollret(20,'min'))
    streak=[]
    for _,x in b.groupby('ts_code',sort=False):streak.append(np.sign(x.close.astype(float)/x.close.astype(float).shift(1)-1).rolling(5,min_periods=5).sum())
    put(16,pd.concat(streak).sort_index())
    for i,w in zip([17,18,19,20],[5,10,20,60]):put(i,ret.groupby(b.ts_code).transform(lambda s:s.rolling(w,min_periods=w).std())*np.sqrt(252))
    for i,w in [(21,20),(22,60)]:put(i,ret.clip(upper=0).pow(2).groupby(b.ts_code).transform(lambda s:s.rolling(w,min_periods=w).mean().pow(.5))*np.sqrt(252))
    tr=np.maximum.reduce([(high-low).abs().to_numpy(),(high-prev).abs().to_numpy(),(low-prev).abs().to_numpy()]);trS=pd.Series(tr,index=b.index);put(23,trS.groupby(b.ts_code).transform(lambda s:s.rolling(14,min_periods=14).mean())/(close.abs()+1e-12));rng=(high-low).abs()/(close.abs()+1e-12);put(24,rng.groupby(b.ts_code).transform(lambda s:s.rolling(20,min_periods=20).mean()))
    for i,src,sw,lw in [(25,vol,5,20),(26,vol,20,60),(27,amt,5,20),(28,amt,20,60)]:
        gg=src.groupby(b.ts_code);put(i,gg.transform(lambda s:s.rolling(sw,min_periods=sw).mean())/(gg.transform(lambda s:s.rolling(lw,min_periods=lw).mean())+1e-12))
    ill=ret.abs()/(amt.abs()+1);put(29,ill.groupby(b.ts_code).transform(lambda s:s.rolling(20,min_periods=20).mean()));put(30,ill.groupby(b.ts_code).transform(lambda s:s.rolling(60,min_periods=60).mean()));put(31,vol.groupby(b.ts_code).transform(lambda s:s.rolling(20,min_periods=20).std())/(vol.groupby(b.ts_code).transform(lambda s:s.rolling(20,min_periods=20).mean())+1e-12));tm=turn.groupby(b.ts_code).transform(lambda s:s.rolling(20,min_periods=20).mean());ts=turn.groupby(b.ts_code).transform(lambda s:s.rolling(20,min_periods=20).std());put(32,(turn-tm)/(ts+1e-12))
    rr=(high-low).abs()+1e-12;put(33,(close-op)/rr);put(34,(close-op).abs()/rr);put(35,(high-pd.concat([op,close],axis=1).max(axis=1))/rr);put(36,(pd.concat([op,close],axis=1).min(axis=1)-low)/rr);put(37,(close-low)/rr);gap=op/prev-1;put(38,gap);ph=high.groupby(b.ts_code).transform(lambda s:s.shift(1).rolling(20,min_periods=20).max());pl=low.groupby(b.ts_code).transform(lambda s:s.shift(1).rolling(20,min_periods=20).min());put(39,close/(ph+1e-12)-1);put(40,close/(pl+1e-12)-1)
    on=gap;intr=close/op-1;put(41,on)
    def comp(s,w):return np.expm1(np.log1p(s.clip(lower=-.999999)).groupby(b.ts_code).transform(lambda q:q.rolling(w,min_periods=w).sum()))
    on5=comp(on,5);on20=comp(on,20);id5=comp(intr,5);id20=comp(intr,20);put(42,on5);put(43,on20);put(44,intr);put(45,id5);put(46,id20);put(47,on5-id5);put(48,on20-id20)
    for i,w in [(49,1),(50,5),(51,20),(52,60)]:r=rs[w];put(i,r-r.groupby(b.trade_date).transform('mean'))
    ma20=close.groupby(b.ts_code).transform(lambda s:s.rolling(20,min_periods=20).mean());ma60=close.groupby(b.ts_code).transform(lambda s:s.rolling(60,min_periods=60).mean());put(53,ma20/(ma60+1e-12)-1);put(54,close/(ma20+1e-12)-1);path=ret.abs().groupby(b.ts_code).transform(lambda s:s.rolling(20,min_periods=20).sum());put(55,rs[20].abs()/(path+1e-12));put(56,rs[20]-rs[20].groupby(b.trade_date).transform('mean'))
    rate=np.full(n,.10,np.float32);code=b.ts_code.astype(str);dt=b.trade_date.to_numpy();rate[code.str.startswith('688').to_numpy()]=.20;chi=code.str.startswith('300').to_numpy();rate[chi&(dt>=np.datetime64('2020-08-24'))]=.20;upl=np.round(prev.to_numpy()*(1+rate),2);limit_hit=(high.to_numpy()>=upl*.999)&np.isfinite(upl)
    order=np.lexsort((b.ts_code.to_numpy(),b.trade_date.to_numpy()));b2=b.iloc[order].reset_index(drop=True);X=X[order];limit_hit=limit_hit[order];starts,ends=cs_z(X,b2.trade_date.to_numpy());print('FACTORS',X.shape,float(np.isfinite(X).mean()),flush=True);return b2,X,limit_hit,starts,ends

def scores(X,limit_hit):
    ci={f:i for i,f in enumerate(FEAT)};out={}
    for sid,terms in SPECS.items():
        num=np.zeros(len(X),np.float32);den=np.zeros(len(X),np.float32)
        for c,w in terms:v=X[:,ci[c]];ok=np.isfinite(v);num+=np.where(ok,v,0)*np.float32(w);den+=ok.astype(np.float32)*abs(w)
        sc=np.divide(num,den,out=np.full(len(X),np.nan,np.float32),where=den>0)
        if sid=='S01':sc[~(X[:,ci['F02']]<0)]=np.nan
        elif sid=='S02':sc[~((X[:,ci['F01']]<0)&(X[:,ci['F32']]>0.5))]=np.nan
        elif sid=='S03':sc[~(X[:,ci['F51']]>0)]=np.nan
        elif sid=='S04':sc[~((X[:,ci['F39']]>0)&(X[:,ci['F25']]>0))]=np.nan
        elif sid=='S05':sc[~((X[:,ci['F06']]>0)&(X[:,ci['F03']]<0)&(X[:,ci['F53']]>0))]=np.nan
        elif sid=='S06':sc[~((X[:,ci['F39']]>-0.2)&(X[:,ci['F17']]<X[:,ci['F19']]))]=np.nan
        elif sid=='S07':sc[~(X[:,ci['F06']]>0)]=np.nan
        elif sid=='S08':sc[~(X[:,ci['F43']]>0)]=np.nan
        elif sid=='S10':sc[~limit_hit]=np.nan
        elif sid=='S11':sc[~((X[:,ci['F53']]>0)&(X[:,ci['F55']]>0))]=np.nan
        out[sid]=sc
    return out

def allfactor_score(b,X,starts,ends,h=5,lookback=126):
    q=b[['ts_code','trade_date','open']].copy().sort_values(['ts_code','trade_date']);gg=q.groupby('ts_code',sort=False).open;q['y']=gg.shift(-(h+1))/gg.shift(-1)-1;q=q.sort_values(['trade_date','ts_code']).reset_index(drop=True);y=q.y.to_numpy(float);D=len(starts);ic=np.full((D,NF),np.nan,np.float32)
    for j,(a,z) in enumerate(zip(starts,ends)):
        yy=y[a:z];xx=X[a:z].astype(float,copy=False);my=np.isfinite(yy)
        if my.sum()<100:continue
        for k in range(NF):
            xv=xx[:,k];m=my&np.isfinite(xv)
            if m.sum()<100:continue
            xa=xv[m];ya=yy[m];sx=xa.std();sy=ya.std()
            if sx>1e-12 and sy>1e-12:ic[j,k]=np.mean((xa-xa.mean())*(ya-ya.mean()))/(sx*sy)
    sc=np.full(len(b),np.nan,np.float32)
    for j,(a,z) in enumerate(zip(starts,ends)):
        e=j-h-1
        if e<40:continue
        w=np.nanmean(ic[max(0,e-lookback+1):e+1],axis=0);w=np.nan_to_num(w);s=np.abs(w).sum()
        if s>1e-8:sc[a:z]=np.nan_to_num(X[a:z])@(w/s).astype(np.float32)
    return sc

def market_arrays(b):
    dates=pd.DatetimeIndex(sorted(b.trade_date.unique()));codes=np.array(sorted(b.ts_code.unique()));ci={c:i for i,c in enumerate(codes)};op=b.pivot(index='trade_date',columns='ts_code',values='open').reindex(index=dates,columns=codes).to_numpy(dtype=np.float32);return dates,codes,ci,op

def groups(b,sc):
    d=b.trade_date.to_numpy();c=b.ts_code.to_numpy();st=np.r_[0,np.flatnonzero(d[1:]!=d[:-1])+1];en=np.r_[st[1:],len(d)];o={}
    for a,z in zip(st,en):
        s=sc[a:z];m=np.isfinite(s)
        if m.any():o[pd.Timestamp(d[a])]=(c[a:z][m],s[m])
    return o

def sim(gr,mkt,h,rbe,buy=15,sell=20,delay=1,top=30):
    dates,codes,ci,op=mkt;sigset=set(pd.Timestamp(x) for x in dates[::max(1,rbe)]);pending={};lots=[];cash=1.;prevnav=1.;rec=[];cf=min(1.,rbe/max(h,1))
    for i,d in enumerate(dates):
        for L in lots:
            p=op[i,L['j']]
            if np.isfinite(p) and L['p']>0:L['v']*=p/L['p'];L['p']=p
        keep=[]
        for L in lots:
            if i>=L['x'] and np.isfinite(op[i,L['j']]):cash+=L['v']*(1-sell/10000);continue
            keep.append(L)
        lots=keep;ords=pending.pop(i,None)
        if ords is not None:
            nav=cash+sum(L['v'] for L in lots);budget=min(cash/(1+buy/10000),cf*nav);sv={}
            for L in lots:sv[L['j']]=sv.get(L['j'],0)+L['v']
            base=budget/max(len(ords),1)
            for j in ords:
                p=op[i,j]
                if not np.isfinite(p):continue
                val=min(base,max(0,.05*nav-sv.get(j,0)),cash/(1+buy/10000))
                if val<=1e-9:continue
                cash-=val*(1+buy/10000);sv[j]=sv.get(j,0)+val;lots.append({'j':j,'v':val,'p':p,'x':i+h})
        nav=cash+sum(L['v'] for L in lots);ret=nav/prevnav-1;gross=sum(L['v'] for L in lots)/nav if nav else 0;mx=0
        if lots and nav:
            sv={}
            for L in lots:sv[L['j']]=sv.get(L['j'],0)+L['v']
            mx=max(sv.values())/nav
        rec.append((pd.Timestamp(d),ret,nav,gross,mx));prevnav=nav;pd_d=pd.Timestamp(d)
        if pd_d in sigset and pd_d in gr and i+delay<len(dates):
            sy,sc=gr[pd_d];k=min(top,len(sc));idx=np.argpartition(sc,-k)[-k:];idx=idx[np.argsort(sc[idx])[::-1]];pending[i+delay]=[ci[x] for x in sy[idx] if x in ci]
    return pd.DataFrame(rec,columns=['trade_date','net_return','nav','gross_exposure','max_single_exposure'])

def met(d):
    if len(d)<2:return {}
    x=d.net_return.fillna(0).to_numpy(float);nav=(1+x).cumprod();yrs=max((d.trade_date.iloc[-1]-d.trade_date.iloc[0]).days/365.2425,1/365);dd=nav/np.maximum.accumulate(nav)-1;sd=x.std(ddof=1);return {'cagr':float(nav[-1]**(1/yrs)-1),'mdd':float(dd.min()),'sharpe':float(np.sqrt(252)*x.mean()/sd if sd>0 else np.nan),'ending_nav':float(nav[-1]),'avg_gross':float(d.gross_exposure.mean()),'max_single':float(d.max_single_exposure.max())}
def sub(d,a,z):return met(d[(d.trade_date>=a)&(d.trade_date<=z)])
def passgate(v,c):return bool(v and v.get('cagr',-1)>0 and v.get('sharpe',-9)>=.5 and v.get('mdd',-1)>-.35 and c and c.get('cagr',-1)>0)

def main(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);b=load_data(root);print('DATA',len(b),b.ts_code.nunique(),b.trade_date.min(),b.trade_date.max(),flush=True);b,X,lim,starts,ends=factors(b);ss=scores(X,lim);print('RULE SCORES DONE',flush=True);ss['S13']=allfactor_score(b,X,starts,ends);print('S13 DONE',flush=True);mkt=market_arrays(b);dates=mkt[0];n=len(dates);dev_end=pd.Timestamp(dates[n//2-1]);val_start=pd.Timestamp(dates[n//2]);val_end=pd.Timestamp(dates[3*n//4-1]);test_start=pd.Timestamp(dates[3*n//4]);end=pd.Timestamp(dates[-1]);start=pd.Timestamp(dates[0]);rows=[];daily={}
    for sid in REG:
        h,r=REG[sid];gr=groups(b,ss[sid]);p=sim(gr,mkt,h,r,15,20,1);c2=sim(gr,mkt,h,r,30,40,1);dl=sim(gr,mkt,h,r,15,20,2);daily[sid]=p;row={'strategy':sid,'dev':sub(p,start,dev_end),'validation':sub(p,val_start,val_end),'test':sub(p,test_start,end),'validation_cost2x':sub(c2,val_start,val_end),'test_delay2':sub(dl,test_start,end),'score_rows':int(np.isfinite(ss[sid]).sum())};row['pass_validation']=passgate(row['validation'],row['validation_cost2x']);rows.append(row);print('DONE',sid,row['pass_validation'],row['validation'],row['test'],flush=True)
    flat=[]
    for r in rows:
        q={'strategy':r['strategy'],'pass_validation':r['pass_validation'],'score_rows':r['score_rows']}
        for p in ['dev','validation','test','validation_cost2x','test_delay2']:
            for k,v in r[p].items():q[p+'_'+k]=v
        flat.append(q)
    summary=pd.DataFrame(flat);summary.to_csv(out/'strategy_summary.csv',index=False);passed=summary.loc[summary.pass_validation,'strategy'].tolist();meta={}
    if passed:
        R=pd.concat([daily[s].set_index('trade_date').net_return.rename(s) for s in passed],axis=1).fillna(0);mr=R.mean(axis=1);md=pd.DataFrame({'trade_date':R.index,'net_return':mr.values,'gross_exposure':1.0,'max_single_exposure':0.0});meta={'all':met(md),'test':sub(md,test_start,end)};md.to_csv(out/'meta_daily.csv',index=False)
    status={'data_source':'newbiestring-lang/astock','rows':len(b),'stocks':int(b.ts_code.nunique()),'sessions':n,'start':str(start.date()),'end':str(end.date()),'alpha_factors':NF,'strategies_tested':len(REG),'passed':passed,'n_passed':len(passed),'requirement_at_least_10':len(passed)>=10,'split':{'dev_end':str(dev_end.date()),'validation_start':str(val_start.date()),'validation_end':str(val_end.date()),'test_start':str(test_start.date())},'meta':meta,'limitations':['forward-adjusted BaoStock prices','no PIT sector labels','no dynamic ST flag/exact 5% ST limit','no exact daily limit-price table; strategy S10 event gate approximated from prior close and board/date rule','source notes early history incomplete for about 685 old stocks']};(out/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2));(out/'RESULTS.md').write_text('# MultiAlpha 56F/13S real-data backtest\n\n```json\n'+json.dumps(status,ensure_ascii=False,indent=2)+'\n```\n\n'+summary.to_markdown(index=False));print('FINAL_STATUS',json.dumps(status,ensure_ascii=False),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();main(a.data_dir,a.out)
