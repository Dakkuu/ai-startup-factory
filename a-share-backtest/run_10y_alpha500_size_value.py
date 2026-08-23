from __future__ import annotations
from pathlib import Path
import re, time, math
import numpy as np
import pandas as pd
import requests

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_grand_opt as grand
import run_10y_balanced_exact as be
import run_10y_max_audit as ma

OUT=Path('results_alpha500_size_value'); OUT.mkdir(exist_ok=True)
ROOT='https://www.dolthub.com/api/v1alpha1/chenditc/investment_data/master'
TARGET_RETURN=5.0  # +500%, final wealth 6x
SNAPSHOT_STEP=4   # existing signal grid is ~weekly => one PIT cross-section about every 20 sessions
NS=(10,15,20,30)
HOLDS=(20,60,120)
AGGR_NS=(5,10,15,20)

SESSION=requests.Session()

def sql(q,tries=5):
    err=None
    for k in range(tries):
        try:
            r=SESSION.get(ROOT,params={'q':q},timeout=180)
            r.raise_for_status(); z=r.json()
            if z.get('query_execution_status')!='Success': raise RuntimeError(z)
            return z.get('rows',[])
        except Exception as e:
            err=e; time.sleep(min(10,1.5**k))
    raise RuntimeError(f'DoltHub query failed after retries: {err}; SQL={q[:500]}')

def norm(x): return re.sub(r'[^a-z0-9]','',str(x).lower())

def pick_col(cols,*aliases):
    mp={norm(c):c for c in cols}
    for a in aliases:
        if norm(a) in mp: return mp[norm(a)]
    return None

def qcode(x):
    s=str(x).strip().upper()
    m=re.fullmatch(r'(\d{6})\.(SH|SZ|BJ)',s)
    if m: return m.group(2)+m.group(1)
    m=re.fullmatch(r'(SH|SZ|BJ)\.(\d{6})',s)
    if m: return m.group(1)+m.group(2)
    m=re.fullmatch(r'(SH|SZ|BJ)(\d{6})',s)
    if m: return m.group(1)+m.group(2)
    m=re.fullmatch(r'(\d{6})',s)
    if m:
        d=m.group(1)
        if d.startswith(('6','9')): return 'SH'+d
        if d.startswith(('4','8')): return 'BJ'+d
        return 'SZ'+d
    return s.replace('.','')

def schema_frame():
    rows=sql("SELECT table_name,column_name,data_type FROM information_schema.columns WHERE table_schema=DATABASE() ORDER BY table_name,ordinal_position")
    z=pd.DataFrame(rows); z.to_csv(OUT/'dolthub_schema.csv',index=False); return z

def source_candidates(schema):
    out=[]
    for t,g in schema.groupby('table_name'):
        cols=list(g.column_name.astype(str)); typ=dict(zip(g.column_name.astype(str),g.data_type.astype(str)))
        code=pick_col(cols,'ts_code','code','symbol','stock_code','stockcode','wind_code')
        date=pick_col(cols,'trade_date','tradedate','date')
        circ=pick_col(cols,'circ_mv','circmv','float_mv','floatmarketcap')
        total=pick_col(cols,'total_mv','totalmv','market_cap','marketcap','totalmarketcap')
        turn=pick_col(cols,'turn','turnover_rate','turnoverrate')
        close=pick_col(cols,'close','close_price','closeprice')
        vol=pick_col(cols,'volume','vol')
        amount=pick_col(cols,'amount','turnover_amount','turnoveramount')
        pb=pick_col(cols,'pb','pb_mrq','pbmrq')
        pe=pick_col(cols,'pe_ttm','pettm','pe')
        ps=pick_col(cols,'ps_ttm','psttm','ps')
        pcf=pick_col(cols,'pcf_ncf_ttm','pcfncfttm','pcf')
        isst=pick_col(cols,'is_st','isst')
        trade=pick_col(cols,'trade_status','tradestatus')
        exact=bool(circ or total); derived=bool(turn and (amount or (close and vol)))
        if not (code and date and (exact or derived)): continue
        score=(100 if exact else 60)+10*sum(x is not None for x in (pb,pe,ps,pcf))+5*(isst is not None)+3*(trade is not None)
        out.append(dict(table=str(t),columns=cols,types=typ,code=code,date=date,circ=circ,total=total,turn=turn,close=close,vol=vol,amount=amount,pb=pb,pe=pe,ps=ps,pcf=pcf,isst=isst,trade=trade,exact=exact,derived=derived,score=score))
    return sorted(out,key=lambda x:x['score'],reverse=True)

def date_where(src,d):
    ds=pd.Timestamp(d).strftime('%Y-%m-%d'); compact=pd.Timestamp(d).strftime('%Y%m%d'); c=src['date']
    return f"(`{c}`='{ds}' OR `{c}`='{compact}')"

def source_count(src,d,head_hash):
    t=src['table']; wh=date_where(src,d)
    try:
        r=sql(f"SELECT COUNT(*) AS n FROM `{t}` AS OF '{head_hash}' WHERE {wh}")
        return int(float(r[0]['n'])) if r else 0
    except Exception:
        r=sql(f"SELECT COUNT(*) AS n FROM `{t}` WHERE {wh}")
        return int(float(r[0]['n'])) if r else 0

def choose_source(schema,test_dates,head_hash):
    cand=source_candidates(schema)
    audit=[]
    for src in cand[:20]:
        counts=[source_count(src,d,head_hash) for d in test_dates]
        row={k:v for k,v in src.items() if k not in ('columns','types')}; row.update({f'count_{i}':n for i,n in enumerate(counts)}); row['min_count']=min(counts); row['median_count']=float(np.median(counts)); audit.append(row)
        print('SOURCE TEST',src['table'],counts,'score',src['score'],flush=True)
    a=pd.DataFrame(audit); a.to_csv(OUT/'source_candidates.csv',index=False)
    if not audit: raise RuntimeError('No PIT market-cap or turnover-derived size source exists in DoltHub schema')
    viable=[x for x in audit if x['min_count']>=500]
    if not viable: raise RuntimeError('No size source has >=500 rows at all test dates')
    best=sorted(viable,key=lambda x:(x['min_count'],x['score'],x['median_count']),reverse=True)[0]
    src=next(c for c in cand if c['table']==best['table'])
    return src

def status_candidates(schema):
    out=[]
    for t,g in schema.groupby('table_name'):
        cols=list(g.column_name.astype(str))
        code=pick_col(cols,'ts_code','code','symbol','stock_code','stockcode')
        date=pick_col(cols,'trade_date','tradedate','date')
        isst=pick_col(cols,'is_st','isst'); trade=pick_col(cols,'trade_status','tradestatus')
        if code and date and (isst or trade): out.append(dict(table=str(t),code=code,date=date,isst=isst,trade=trade,score=10*(isst is not None)+3*(trade is not None)))
    return sorted(out,key=lambda x:x['score'],reverse=True)

def choose_status_source(schema,test_dates,head_hash):
    audit=[]
    for src in status_candidates(schema)[:15]:
        counts=[]
        for d in test_dates:
            try: counts.append(source_count(src,d,head_hash))
            except Exception: counts.append(0)
        audit.append({**src,'min_count':min(counts),'median_count':float(np.median(counts))})
    pd.DataFrame(audit).to_csv(OUT/'status_candidates.csv',index=False)
    v=[x for x in audit if x['min_count']>=500]
    return sorted(v,key=lambda x:(x['min_count'],x['score']),reverse=True)[0] if v else None

def expr(src,key,alias):
    c=src.get(key)
    return f'`{c}` AS `{alias}`' if c else f'NULL AS `{alias}`'

def fetch_cross_section(src,d,head_hash):
    t=src['table']; cols=[f"`{src['code']}` AS code_raw",f"`{src['date']}` AS pit_date"]
    for k,a in [('circ','circ_mv'),('total','total_mv'),('turn','turn'),('close','raw_close'),('vol','volume'),('amount','amount'),('pb','pb'),('pe','pe'),('ps','ps'),('pcf','pcf'),('isst','is_st'),('trade','trade_status')]: cols.append(expr(src,k,a))
    wh=date_where(src,d); select=', '.join(cols)
    queries=[f"SELECT {select} FROM `{t}` AS OF '{head_hash}' WHERE {wh} LIMIT 10000",f"SELECT {select} FROM `{t}` WHERE {wh} LIMIT 10000"]
    last=None
    for qq in queries:
        try:
            rows=sql(qq); break
        except Exception as e: last=e; rows=None
    if rows is None: raise last
    z=pd.DataFrame(rows)
    if z.empty: return z
    z['signal_date']=pd.Timestamp(d); z['code']=z.code_raw.map(qcode)
    for c in ['circ_mv','total_mv','turn','raw_close','volume','amount','pb','pe','ps','pcf','is_st','trade_status']:
        z[c]=pd.to_numeric(z[c],errors='coerce')
    size=z.circ_mv.where(z.circ_mv>0)
    size=size.fillna(z.total_mv.where(z.total_mv>0))
    if size.isna().any() and src.get('turn'):
        turn=z.turn.where(z.turn>0)
        der=(z.amount*100.0/turn) if src.get('amount') else (z.raw_close*z.volume*100.0/turn)
        size=size.fillna(der.where(der>0))
    z['pit_size']=size
    return z[['signal_date','code','pit_size','pb','pe','ps','pcf','is_st','trade_status']].drop_duplicates(['signal_date','code'],keep='last')

def fetch_status_cross(src,d,head_hash):
    if src is None: return pd.DataFrame(columns=['signal_date','code','status_is_st','status_trade'])
    cols=[f"`{src['code']}` AS code_raw",expr(src,'isst','status_is_st'),expr(src,'trade','status_trade')]; wh=date_where(src,d); t=src['table']
    rows=None
    for qq in [f"SELECT {', '.join(cols)} FROM `{t}` AS OF '{head_hash}' WHERE {wh} LIMIT 10000",f"SELECT {', '.join(cols)} FROM `{t}` WHERE {wh} LIMIT 10000"]:
        try: rows=sql(qq); break
        except Exception: pass
    z=pd.DataFrame(rows or [])
    if z.empty: return pd.DataFrame(columns=['signal_date','code','status_is_st','status_trade'])
    z['signal_date']=pd.Timestamp(d); z['code']=z.code_raw.map(qcode)
    z['status_is_st']=pd.to_numeric(z.status_is_st,errors='coerce'); z['status_trade']=pd.to_numeric(z.status_trade,errors='coerce')
    return z[['signal_date','code','status_is_st','status_trade']].drop_duplicates(['signal_date','code'],keep='last')

def fetch_pit(src,status_src,dates,head_hash):
    parts=[]; sts=[]
    for i,d in enumerate(dates,1):
        z=fetch_cross_section(src,d,head_hash); parts.append(z)
        if status_src and status_src['table']!=src['table']: sts.append(fetch_status_cross(status_src,d,head_hash))
        if i%10==0 or i==len(dates): print('PIT',i,'/',len(dates),'rows',len(z),flush=True)
    p=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    if sts:
        s=pd.concat(sts,ignore_index=True); p=p.merge(s,on=['signal_date','code'],how='left')
        if 'is_st' not in p or p.is_st.isna().all(): p['is_st']=p.status_is_st
        else: p['is_st']=p.is_st.fillna(p.status_is_st)
        if 'trade_status' not in p or p.trade_status.isna().all(): p['trade_status']=p.status_trade
        else: p['trade_status']=p.trade_status.fillna(p.status_trade)
        p=p.drop(columns=['status_is_st','status_trade'],errors='ignore')
    p.to_csv(OUT/'pit_factor_snapshot.csv.gz',index=False,compression='gzip')
    return p

def pct(q,m,col,asc): return q.loc[m].groupby('signal_date')[col].rank(pct=True,method='average',ascending=asc)

POOL={
 'core70':dict(liq=.70,age=365,exclude_boards={'STAR','BJ'},skew=.80),
 'core80':dict(liq=.80,age=365,exclude_boards={'STAR','BJ'},skew=.80),
 'core90':dict(liq=.90,age=365,exclude_boards={'STAR','BJ'},skew=.80),
 'core80_noskew':dict(liq=.80,age=365,exclude_boards={'STAR','BJ'},skew=None),
 'broad80':dict(liq=.80,age=180,exclude_boards={'BJ'},skew=.90),
}

def pool_mask(q,name):
    s=POOL[name]; m=np.isfinite(q.pit_size)&np.isfinite(q.ivol60)&np.isfinite(q.eff120)&np.isfinite(q.liq_rank_pct)&(q.liq_rank_pct<=s['liq'])&(q.age_days>=s['age'])&(~q.board.isin(s['exclude_boards']))
    if 'is_st' in q and q.is_st.notna().any(): m=m&(~(q.is_st.fillna(0)>0))
    if 'trade_status' in q and q.trade_status.notna().any(): m=m&(q.trade_status.fillna(1)==1)
    if s['skew'] is not None:
        m=m&np.isfinite(q.skew40); sr=pct(q,m,'skew40',True); ok=pd.Series(False,index=q.index); ok.loc[sr.index]=sr<=s['skew']; m=m&ok
    return m

FAMILIES=('size','size_ivol','size_eff','size_bal','size_pb','size_pb_ivol','size_pb_eff','size_value_bal','small30_ivol_eff','small20_ivol_eff','small30_pb_eff','small20_pb_eff','size_momgate','size_bal_momgate','small30_bal_momgate')

def rerank(q0,family,pool):
    q=q0.copy(); q['rank_test']=np.nan; m=pool_mask(q,pool)
    if family.endswith('momgate') or family in ('size_momgate','size_bal_momgate'):
        m=m&np.isfinite(q.mom120raw)&(q.mom120raw>0)
    if family in ('size_pb','size_pb_ivol','size_pb_eff','size_value_bal','small30_pb_eff','small20_pb_eff'):
        m=m&np.isfinite(q.pb)&(q.pb>0)
    if not m.any(): return q
    rs=pct(q,m,'pit_size',True); ri=pct(q,m,'ivol60',True); re=pct(q,m,'eff120',False)
    rp=pct(q,m,'pb',True) if (np.isfinite(q.loc[m,'pb'])&(q.loc[m,'pb']>0)).any() else pd.Series(np.nan,index=q.loc[m].index)
    if family=='size': raw=rs
    elif family=='size_ivol': raw=.70*rs+.30*ri
    elif family=='size_eff': raw=.70*rs+.30*re
    elif family=='size_bal': raw=.55*rs+.25*ri+.20*re
    elif family=='size_pb': raw=.65*rs+.35*rp
    elif family=='size_pb_ivol': raw=.55*rs+.25*rp+.20*ri
    elif family=='size_pb_eff': raw=.55*rs+.25*rp+.20*re
    elif family=='size_value_bal': raw=.45*rs+.25*rp+.15*ri+.15*re
    elif family in ('small30_ivol_eff','small20_ivol_eff','small30_pb_eff','small20_pb_eff','small30_bal_momgate'):
        cut=.20 if '20' in family else .30; sm=rs<=cut; m2=pd.Series(False,index=q.index); m2.loc[sm.index]=sm; m=m&m2
        if 'pb' in family: m=m&np.isfinite(q.pb)&(q.pb>0)
        if not m.any(): return q
        ri=pct(q,m,'ivol60',True); re=pct(q,m,'eff120',False); rp=pct(q,m,'pb',True); rs2=pct(q,m,'pit_size',True)
        if 'pb_eff' in family: raw=.60*rp+.40*re
        elif family=='small30_bal_momgate': raw=.45*rs2+.30*ri+.25*re
        else: raw=.55*ri+.45*re
    elif family=='size_momgate': raw=rs
    elif family=='size_bal_momgate': raw=.55*rs+.25*ri+.20*re
    else: raise ValueError(family)
    q.loc[raw.index,'rank_test']=raw.groupby(q.loc[raw.index,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q

def run_grid(p,cal,members,bm):
    phase1=[]; qcache={}
    for fam in FAMILIES:
        for pool in POOL:
            q=rerank(p,fam,pool); qcache[(fam,pool)]=q
            try:
                st,_,_,_=grand.run(q,60,20,.10,.30,cal,members,bm,fast=True); st.update({'family':fam,'pool':pool,'wave':'phase1'}); phase1.append(st)
            except Exception as e: phase1.append({'family':fam,'pool':pool,'wave':'phase1','error':repr(e)})
            print('PHASE1',fam,pool,phase1[-1].get('total_return'),flush=True)
    p1=pd.DataFrame(phase1); p1.to_csv(OUT/'phase1.csv',index=False)
    ok1=p1[p1.total_return.notna()&p1.train_cagr.notna()&(p1.train_return>0)&(p1.positions_max<=p1.n_hold)].copy(); ok1['train_score']=ok1.train_cagr-.15*ok1.train_mdd.abs(); top_pairs=[(r.family,r.pool) for _,r in ok1.sort_values('train_score',ascending=False).drop_duplicates(['family','pool']).head(10).iterrows()]
    print('TOP PAIRS TRAIN',top_pairs,flush=True)
    phase2=[]
    for fam,pool in top_pairs:
        q=qcache[(fam,pool)]
        for n in NS:
            for h in HOLDS:
                st,_,_,_=grand.run(q,h,n,.10,.30,cal,members,bm,fast=True); st.update({'family':fam,'pool':pool,'wave':'phase2'}); phase2.append(st)
                print('PHASE2',fam,pool,n,h,'ret',st['total_return'],'train',st['train_cagr'],'val',st['validation_return'],flush=True)
    p2=pd.DataFrame(phase2); p2.to_csv(OUT/'phase2.csv',index=False)
    ok=p2[(p2.train_return>0)&(p2.positions_max<=p2.n_hold)].copy(); ok['train_score']=ok.train_cagr-.15*ok.train_mdd.abs()
    train_top=ok.sort_values('train_score',ascending=False).head(15)
    targetish=ok[ok.total_return>=3.5].sort_values('total_return',ascending=False).head(20)
    picks=pd.concat([train_top,targetish],ignore_index=True).drop_duplicates(['family','pool','n_hold','hold_days']).head(30)
    exact=[]; cache={}
    for _,r in picks.iterrows():
        key=(str(r.family),str(r.pool),int(r.n_hold),int(r.hold_days)); q=qcache[(key[0],key[1])]
        st,eq,tr,tm=grand.run(q,key[3],key[2],.10,.30,cal,members,bm,fast=False); st.update({'family':key[0],'pool':key[1],'wave':'exact'}); exact.append(st); cache[key]=(eq,tr,tm,q)
        print('EXACT',key,'ret',st['total_return'],'cagr',st['cagr'],'mdd',st['max_drawdown'],'train',st['train_return'],'val',st['validation_return'],flush=True)
    ex=pd.DataFrame(exact); ex.to_csv(OUT/'exact_candidates.csv',index=False)
    return p1,p2,ex,cache,qcache

def aggressive_wave(p,cal,members,bm,qcache):
    rows=[]; exact=[]; cache={}
    combos=[]
    for fam in ('size','size_ivol','size_bal','small20_ivol_eff','small30_ivol_eff','size_momgate','size_bal_momgate','small30_bal_momgate'):
        for pool in ('core80_noskew','core90','broad80'):
            q=qcache.get((fam,pool))
            if q is None: q=rerank(p,fam,pool); qcache[(fam,pool)]=q
            for n in AGGR_NS:
                for h in HOLDS:
                    st,_,_,_=grand.run(q,h,n,.05,.20,cal,members,bm,fast=True); st.update({'family':fam,'pool':pool,'wave':'aggressive_fast'}); rows.append(st)
                    if st['total_return']>=3.5: combos.append((st['total_return'],fam,pool,n,h))
    f=pd.DataFrame(rows); f.to_csv(OUT/'aggressive_fast.csv',index=False)
    combos=sorted(combos,reverse=True)[:25]
    for _,fam,pool,n,h in combos:
        q=qcache[(fam,pool)]; st,eq,tr,tm=grand.run(q,h,n,.05,.20,cal,members,bm,fast=False); st.update({'family':fam,'pool':pool,'wave':'aggressive_exact'}); exact.append(st); cache[(fam,pool,n,h)]=(eq,tr,tm,q)
        print('AGGR EXACT',fam,pool,n,h,st['total_return'],st['cagr'],st['max_drawdown'],flush=True)
    e=pd.DataFrame(exact); e.to_csv(OUT/'aggressive_exact.csv',index=False); return f,e,cache

def deep_audit(best,cache,cal,members,bm):
    fam=str(best.family); pool=str(best.pool); n=int(best.n_hold); h=int(best.hold_days); entry=float(best.entry_pct); keep=float(best.keep_pct); key=(fam,pool,n,h)
    if key not in cache: return
    eq,tr,tm,q=cache[key]
    costs=[]
    for cm in (1.,2.,4.,8.):
        st,_,_,_=grand.run(q,h,n,entry,keep,cal,members,bm,cost=cm,fast=False); st['cost_mult_test']=cm; costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'winner_costs.csv',index=False)
    sim.annual_returns(eq).to_csv(OUT/'winner_annual.csv',index=False); pd.DataFrame([sim.robustness(eq,tr)]).to_csv(OUT/'winner_tail.csv',index=False)
    neigh=[]
    for nn in sorted(set([max(5,n-5),n,min(30,n+5)])):
        for hh in sorted(set([20,h,60,120])):
            st,_,_,_=grand.run(q,hh,nn,entry,keep,cal,members,bm,fast=False); st.update({'neighbor_n':nn,'neighbor_h':hh}); neigh.append(st)
    pd.DataFrame(neigh).to_csv(OUT/'winner_neighborhood.csv',index=False)
    # Capacity: same frozen signal with 1% and 5% participation at 1m/5m/10m.
    z=grand.subset(q,h); cap=[]
    for cash in (1e6,5e6,1e7):
        for vp in (.01,.05):
            st,_,_,_=ma.run_panel(z,cal,members,bm,n=n,entry=entry,keep=keep,initial_cash=cash,vol_part=vp); st.update({'cash_test':cash,'vp_test':vp}); cap.append(st)
    pd.DataFrame(cap).to_csv(OUT/'winner_capacity.csv',index=False)

def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members)
    bm=market_close.loc[sim.START:sim.END].dropna(); all_dates=pd.DatetimeIndex(sorted(pd.to_datetime(p.signal_date.unique()))); snap_dates=all_dates[::SNAPSHOT_STEP]
    try: head=sql("SELECT DOLT_HASHOF('HEAD') AS h")[0]['h']
    except Exception: head='master'
    schema=schema_frame(); tests=[snap_dates[0],snap_dates[len(snap_dates)//2],snap_dates[-1]]; src=choose_source(schema,tests,head); status_src=choose_status_source(schema,tests,head)
    meta={k:v for k,v in src.items() if k not in ('columns','types')}; meta.update({'dolt_head':head,'status_table':status_src['table'] if status_src else None,'snapshot_dates':len(snap_dates),'target_total_return':TARGET_RETURN}); pd.DataFrame([meta]).to_csv(OUT/'source_selected.csv',index=False); print('SOURCE SELECTED',meta,flush=True)
    pit=fetch_pit(src,status_src,snap_dates,head)
    p=p.merge(pit,on=['signal_date','code'],how='left'); cov=p[p.signal_date.isin(snap_dates)].groupby('signal_date').agg(panel_n=('code','size'),size_n=('pit_size',lambda x:int(np.isfinite(x).sum())),pb_n=('pb',lambda x:int(np.isfinite(x).sum())),st_known=('is_st',lambda x:int(x.notna().sum()))).reset_index(); cov['size_coverage']=cov.size_n/cov.panel_n; cov['pb_coverage']=cov.pb_n/cov.panel_n; cov['st_coverage']=cov.st_known/cov.panel_n; cov.to_csv(OUT/'pit_coverage.csv',index=False)
    if cov.size_coverage.median()<.50 or cov.size_coverage.min()<.20: raise RuntimeError(f'PIT size coverage too weak: median {cov.size_coverage.median()} min {cov.size_coverage.min()}')
    p1,p2,ex,cache,qcache=run_grid(p,cal,members,bm)
    all_exact=ex.copy()
    if ex.empty or ex.total_return.max()<TARGET_RETURN:
        af,ae,ac=aggressive_wave(p,cal,members,bm,qcache); all_exact=pd.concat([all_exact,ae],ignore_index=True); cache.update(ac)
    all_exact.to_csv(OUT/'all_exact.csv',index=False)
    hits=all_exact[all_exact.total_return>=TARGET_RETURN].sort_values('total_return',ascending=False) if len(all_exact) else pd.DataFrame(); hits.to_csv(OUT/'target_hits.csv',index=False)
    if len(all_exact):
        train=all_exact.copy(); train['train_score']=train.train_cagr-.15*train.train_mdd.abs(); train_best=train.sort_values('train_score',ascending=False).iloc[0]; full_best=all_exact.sort_values('total_return',ascending=False).iloc[0]
        pd.DataFrame([train_best]).to_csv(OUT/'train_selected_best.csv',index=False); pd.DataFrame([full_best]).to_csv(OUT/'fullsample_best_exploratory.csv',index=False)
        # Audit target hit if any; otherwise audit the strongest exact candidate.
        best=hits.iloc[0] if len(hits) else full_best; key=(str(best.family),str(best.pool),int(best.n_hold),int(best.hold_days)); deep_audit(best,cache,cal,members,bm)
        verdict={**ua,'market_factor':market_code,'dolt_head':head,'pit_table':src['table'],'status_table':status_src['table'] if status_src else None,'size_coverage_median':float(cov.size_coverage.median()),'st_coverage_median':float(cov.st_coverage.median()),'exact_candidates':len(all_exact),'target_hits':len(hits),'best_total_return':float(full_best.total_return),'best_cagr':float(full_best.cagr),'best_mdd':float(full_best.max_drawdown),'best_family':str(full_best.family),'best_pool':str(full_best.pool),'best_n':int(full_best.n_hold),'best_hold':int(full_best.hold_days),'target_500_reached':int(float(full_best.total_return)>=TARGET_RETURN),'research_status':'EXPLORATORY; 2016-2026 repeatedly inspected; any hit must be frozen and independently validated before promotion'}
    else:
        verdict={**ua,'market_factor':market_code,'dolt_head':head,'pit_table':src['table'],'target_500_reached':0,'research_status':'NO EXACT CANDIDATES'}
    pd.DataFrame([verdict]).to_csv(OUT/'verdict.csv',index=False); print('=== VERDICT ==='); print(pd.DataFrame([verdict]).to_string(index=False),flush=True)
    if len(all_exact):
        print('=== TOP EXACT ==='); print(all_exact.sort_values('total_return',ascending=False).head(25).to_string(index=False),flush=True)
        print('=== TARGET HITS ==='); print(hits.head(25).to_string(index=False) if len(hits) else 'NONE',flush=True)

if __name__=='__main__': main()
