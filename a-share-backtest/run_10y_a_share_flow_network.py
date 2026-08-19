from __future__ import annotations
import math, os, re, time, warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import akshare as ak
import run_10y_china_behavior_daily as base

warnings.filterwarnings("ignore")
OUT = Path("results_10y_flow_network")
OUT.mkdir(exist_ok=True)

START = pd.Timestamp("2016-07-29")
END = pd.Timestamp("2026-07-29")
INITIAL = 1_000_000.0
MAX_NAMES = 5
COMMISSION = 0.00025
MIN_COMMISSION = 5.0
SLIPPAGE = 0.0020
PARTICIPATION = 0.02

VARIANTS = [
    "01_leader_only",
    "02_network_leader",
    "03_network_auction",
    "04_network_auction_lhb",
    "05_full_regime_network_auction_lhb_event",
]

def pct_rank(x):
    x=np.asarray(x,float); out=np.full(x.shape,np.nan); ok=np.isfinite(x)
    if ok.sum()==0: return out
    vals=x[ok]; order=np.argsort(np.argsort(vals, kind="mergesort"), kind="mergesort")
    out[ok]=(order+0.5)/len(vals); return out

def rolling_mean_mat(a,w):
    a=np.asarray(a,float); out=np.full_like(a,np.nan,dtype=float)
    s=np.nancumsum(np.where(np.isfinite(a),a,0.0),axis=0); c=np.cumsum(np.isfinite(a),axis=0)
    for t in range(w-1,len(a)):
        s0=s[t]-(s[t-w] if t>=w else 0); c0=c[t]-(c[t-w] if t>=w else 0)
        out[t]=np.divide(s0,c0,out=np.full(a.shape[1],np.nan),where=c0>=max(3,w//2))
    return out

def code6(c): return c[-6:]
def map_code(c6):
    s=str(c6).zfill(6)
    if s.startswith(("600","601","603","605","688")): return "SH"+s
    if s.startswith(("000","001","002","003","300","301")): return "SZ"+s
    if s.startswith(("4","8","9")): return "BJ"+s
    return None

def get_limit_pct(code, d):
    s=code6(code)
    if code.startswith("BJ"): return 0.30
    if code.startswith("SH") and s.startswith("688"): return 0.20
    if code.startswith("SZ") and s.startswith(("300","301")) and pd.Timestamp(d)>=pd.Timestamp("2020-08-24"): return 0.20
    return 0.10

def fee(gross, side, d):
    if gross<=0: return 0.0
    f=max(MIN_COMMISSION,gross*COMMISSION)+gross*0.00001
    if side=="sell": f += gross*(0.0005 if pd.Timestamp(d)>=pd.Timestamp("2023-08-28") else 0.001)
    return f

def fetch_lhb(start,end):
    rows=[]; audit=[]
    for y in range(start.year,end.year+1):
        a=max(start,pd.Timestamp(f"{y}-01-01")); b=min(end,pd.Timestamp(f"{y}-12-31"))
        try:
            df=ak.stock_lhb_detail_em(start_date=a.strftime("%Y%m%d"),end_date=b.strftime("%Y%m%d"))
            if df is None or df.empty: audit.append((y,0,"empty")); continue
            keep=["代码","上榜日","龙虎榜净买额","龙虎榜买入额","龙虎榜卖出额","龙虎榜成交额","市场总成交额","净买额占总成交比","成交额占总成交比","换手率","流通市值","上榜原因"]
            df=df[[c for c in keep if c in df.columns]].copy(); df["code"]=df["代码"].map(map_code); df["date"]=pd.to_datetime(df["上榜日"])
            agg=df.groupby(["date","code"],as_index=False).agg(lhb_net=("龙虎榜净买额","first"),lhb_buy=("龙虎榜买入额","first"),lhb_sell=("龙虎榜卖出额","first"),lhb_deal=("龙虎榜成交额","first"),lhb_market=("市场总成交额","first"),lhb_net_ratio=("净买额占总成交比","first"),lhb_deal_ratio=("成交额占总成交比","first"),lhb_turnover=("换手率","first"),lhb_floatcap=("流通市值","first"),reason_count=("上榜原因","nunique"))
            rows.append(agg); audit.append((y,len(agg),"ok")); print("LHB",y,len(agg),flush=True)
        except Exception as e:
            print("LHB ERROR",y,type(e).__name__,str(e)[:200],flush=True); audit.append((y,0,"error:"+type(e).__name__))
    pd.DataFrame(audit,columns=["year","rows","status"]).to_csv(OUT/"lhb_coverage.csv",index=False)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

EVENT_QUERIES={"业绩预增":1.0,"扭亏":0.9,"中标":0.7,"重大合同":0.8,"增持计划":0.7,"立案调查":-1.2,"终止重组":-0.9}
def fetch_sparse_events(start,end):
    rows=[]; audit=[]
    for y in range(max(2017,start.year),end.year+1):
        a=max(start,pd.Timestamp(f"{y}-01-01")); b=min(end,pd.Timestamp(f"{y}-12-31"))
        for kw,score in EVENT_QUERIES.items():
            try:
                df=ak.stock_zh_a_disclosure_report_cninfo(symbol="",market="沪深京",keyword=kw,category="",start_date=a.strftime("%Y%m%d"),end_date=b.strftime("%Y%m%d"))
                n=0 if df is None else len(df); audit.append((y,kw,n,"ok"))
                if n:
                    z=df[["代码","简称","公告标题","公告时间"]].copy(); z["code"]=z["代码"].map(map_code); z["announce_time"]=pd.to_datetime(z["公告时间"],errors="coerce"); z["date"]=z["announce_time"].dt.normalize(); z["event_raw"]=score; z["keyword"]=kw
                    rows.append(z[["date","code","event_raw","keyword","公告标题","announce_time"]])
                print("EVENT",y,kw,n,flush=True)
            except Exception as e:
                audit.append((y,kw,0,"error:"+type(e).__name__)); print("EVENT ERROR",y,kw,type(e).__name__,str(e)[:200],flush=True)
    pd.DataFrame(audit,columns=["year","keyword","rows","status"]).to_csv(OUT/"event_coverage.csv",index=False)
    if not rows: return pd.DataFrame()
    ev=pd.concat(rows,ignore_index=True).dropna(subset=["date","code"]); ev2=ev.groupby(["date","code"],as_index=False).agg(event_raw=("event_raw","sum"),event_hits=("keyword","nunique")); ev2["event_raw"]=ev2["event_raw"].clip(-2,2); return ev2

def build_regime(dates, member, ret1, limitup, limitdn, close):
    n=len(dates); rec=[]; ma20=rolling_mean_mat(close,20)
    for t in range(n):
        m=member[t] & np.isfinite(ret1[t]); up=limitup[t,m].mean() if m.any() else np.nan; dn=limitdn[t,m].mean() if m.any() else np.nan; med=np.nanmedian(ret1[t,m]) if m.any() else np.nan; br=np.nanmean(close[t,m]>ma20[t,m]) if t>=19 and m.any() else np.nan
        if t>0:
            p=member[t-1] & limitup[t-1] & np.isfinite(ret1[t]); prem=np.nanmean(ret1[t,p]) if p.any() else np.nan; reseal=np.nanmean(limitup[t,p]) if p.any() else np.nan
        else: prem=reseal=np.nan
        rec.append([dates[t],up,dn,med,br,prem,reseal])
    df=pd.DataFrame(rec,columns=["date","limitup_share","limitdn_share","median_ret","breadth20","board_premium","reseal_rate"]); df["up_med60"]=df.limitup_share.rolling(60,min_periods=20).median(); df["up_q85_60"]=df.limitup_share.rolling(60,min_periods=20).quantile(.85); df["dn_q80_60"]=df.limitdn_share.rolling(60,min_periods=20).quantile(.80)
    states=[]
    for _,r in df.iterrows():
        up,dn,med,br,prem,rs=r.limitup_share,r.limitdn_share,r.median_ret,r.breadth20,r.board_premium,r.reseal_rate; um=r.up_med60 if pd.notna(r.up_med60) else .01; uq=r.up_q85_60 if pd.notna(r.up_q85_60) else .02; dq=r.dn_q80_60 if pd.notna(r.dn_q80_60) else .005
        if (pd.notna(prem) and prem<-.025) or (dn>max(.008,dq) and med<-.01): st="retreat"
        elif br<.28 and up<max(.004,um*.6): st="ice"
        elif up>max(uq,.02) and pd.notna(prem) and prem>.025 and br>.6: st="climax"
        elif pd.notna(prem) and prem>.015 and pd.notna(rs) and rs>.25 and br>.45: st="main"
        elif med>0 and (pd.isna(prem) or prem>-.005) and br>.33: st="repair"
        else: st="divergence"
        states.append(st)
    df["state"]=states; return df

EXPOSURE={"ice":.20,"repair":.60,"main":.90,"climax":.55,"divergence":.35,"retreat":.10}
@dataclass
class Position:
    units:int; entry_px:float; entry_date:pd.Timestamp; signal_date:pd.Timestamp; peak:float; entry_value:float; entry_fee:float

def compute_daily_candidates(t,dates,codes,member,close,volume,ret1,mom5,volratio,amount_rank,limitup,streak,lhb_map,event_map):
    valid=member[t]&np.isfinite(close[t])&np.isfinite(ret1[t])&np.isfinite(mom5[t])&np.isfinite(volratio[t])
    if valid.sum()<30: return pd.DataFrame()
    ids=np.where(valid)[0]; att=.30*pct_rank(ret1[t,ids])+.25*pct_rank(mom5[t,ids])+.20*pct_rank(np.log1p(np.maximum(volratio[t,ids],0)))+.15*pct_rank(amount_rank[t,ids])+.10*np.minimum(streak[t,ids],3)/3; take=ids[np.argsort(np.nan_to_num(att,nan=-9))[-80:]]; k=len(take)
    lo=max(1,t-14); H=ret1[lo:t+1,take].copy()
    for r in range(H.shape[0]):
        mm=member[lo+r]&np.isfinite(ret1[lo+r]); H[r]-=np.nanmedian(ret1[lo+r,mm]) if mm.any() else 0
    H=np.nan_to_num(H,nan=0,posinf=0,neginf=0); H-=H.mean(axis=0,keepdims=True); sd=H.std(axis=0,keepdims=True); sd[sd<1e-6]=1; Z=H/sd; corr=(Z.T@Z)/max(1,H.shape[0]-1); np.fill_diagonal(corr,0); adj=corr>.55; pc=adj.sum(axis=1)
    tr=np.nan_to_num(ret1[t,take],nan=0); tl=limitup[t,take].astype(float); tv=np.nan_to_num(np.log1p(np.maximum(volratio[t,take],0)),nan=0); peer_strength=np.divide(adj@tr,pc,out=np.zeros(k),where=pc>0); peer_limit=np.divide(adj@tl,pc,out=np.zeros(k),where=pc>0); peer_vol=np.divide(adj@tv,pc,out=np.zeros(k),where=pc>0); theme=pct_rank(1.8*peer_strength+1.2*peer_limit+.15*peer_vol+.015*np.minimum(pc,20)); lead_raw=1.2*np.nan_to_num(mom5[t,take])+.7*np.nan_to_num(ret1[t,take])+.12*np.minimum(streak[t,take],4)+.18*np.log1p(np.maximum(volratio[t,take],0))+.35*np.nan_to_num(theme); leader=pct_rank(lead_raw); capacity=pct_rank(.55*pct_rank(amount_rank[t,take])+.25*leader+.20*theme); crowd=(np.maximum(mom5[t,take]-.28,0)/.28+(streak[t,take]>=3)*.7+np.maximum(volratio[t,take]-5,0)/5+(close[t,take]<5)*.25); crowd=np.clip(crowd,0,3)/3; base_score=.35*pct_rank(mom5[t,take])+.20*pct_rank(ret1[t,take])+.20*pct_rank(volratio[t,take])+.15*pct_rank(amount_rank[t,take])+.10*np.minimum(streak[t,take],3)/3
    d=pd.Timestamp(dates[t]).normalize(); rows=[]
    for j,idx in enumerate(take):
        l=lhb_map.get((d,codes[idx])); lhb=0 if l is None else (0 if not np.isfinite(l.get("net_ratio",np.nan)) else np.tanh(l["net_ratio"]/10.0))+(0.25 if np.isfinite(l.get("net",np.nan)) and l["net"]>0 else -0.15); ev=event_map.get((d,codes[idx]),0.0); ev=ev*max(.20,1-max(mom5[t,idx],0)/.30) if ev>0 and np.isfinite(mom5[t,idx]) else ev
        rows.append({"idx":idx,"code":codes[idx],"base":float(base_score[j]),"theme":float(theme[j]),"leader":float(leader[j]),"capacity":float(capacity[j]),"crowd":float(crowd[j]),"lhb":float(lhb),"event":float(ev),"peer_count":int(pc[j]),"streak":int(streak[t,idx]),"ret1":float(ret1[t,idx]),"mom5":float(mom5[t,idx]),"volratio":float(volratio[t,idx])})
    return pd.DataFrame(rows)

def score_variant(df,variant):
    if df.empty: return df
    x=df.copy()
    if variant=="01_leader_only": x["score"]=.55*x.leader+.25*x.base+.20*x.capacity-.18*x.crowd
    elif variant=="02_network_leader": x["score"]=.35*x.theme+.35*x.leader+.20*x.capacity+.10*x.base-.18*x.crowd
    elif variant=="03_network_auction": x["score"]=.35*x.theme+.35*x.leader+.20*x.capacity+.10*x.base-.20*x.crowd
    elif variant=="04_network_auction_lhb": x["score"]=.31*x.theme+.31*x.leader+.18*x.capacity+.08*x.base+.12*x.lhb-.20*x.crowd
    else: x["score"]=.27*x.theme+.27*x.leader+.18*x.capacity+.06*x.base+.10*x.lhb+.12*x.event-.22*x.crowd
    return x.sort_values("score",ascending=False)

def stats(eq,trades):
    s=pd.Series(eq.equity.values,index=pd.to_datetime(eq.date)); r=s.pct_change().fillna(0); total=s.iloc[-1]/s.iloc[0]-1; yrs=(s.index[-1]-s.index[0]).days/365.25; cagr=(1+total)**(1/yrs)-1 if total>-1 and yrs>0 else np.nan; dd=s/s.cummax()-1; sharpe=np.sqrt(252)*r.mean()/r.std() if r.std()>0 else np.nan; neg=r[r<0]; sortino=np.sqrt(252)*r.mean()/neg.std() if len(neg)>2 and neg.std()>0 else np.nan; t=pd.DataFrame(trades)
    if len(t):
        wins=t.net_pnl[t.net_pnl>0]; losses=t.net_pnl[t.net_pnl<0]; pf=wins.sum()/(-losses.sum()) if losses.sum()<0 else np.nan; wr=(t.net_pnl>0).mean()
    else: pf=wr=np.nan
    return {"final_asset":s.iloc[-1],"total_return":total,"cagr":cagr,"max_drawdown":dd.min(),"sharpe":sharpe,"sortino":sortino,"trades":len(t),"win_rate":wr,"profit_factor":pf}

def main():
    dates,codes,close,open_,high,volume,factor,member,load_audit=base.load_data(); dates=pd.DatetimeIndex(dates); codes=list(codes); member=member.astype(bool); n,p=close.shape; print("DATA",n,p,"member union",int(member.any(axis=0).sum()),flush=True)
    ret1=np.full_like(close,np.nan,dtype=float); ret1[1:]=close[1:]/close[:-1]-1; mom5=np.full_like(close,np.nan,dtype=float); mom5[5:]=close[5:]/close[:-5]-1; vma20=rolling_mean_mat(volume,20); volratio=np.divide(volume,vma20,out=np.full_like(volume,np.nan,dtype=float),where=vma20>0); liq=np.log1p(np.maximum(volume*close,0)); amount_rank=np.full_like(close,np.nan,dtype=float)
    for t in range(n):
        ids=np.where(member[t]&np.isfinite(liq[t]))[0]; amount_rank[t,ids]=pct_rank(liq[t,ids])
    limitup=np.zeros((n,p),bool); limitdn=np.zeros((n,p),bool)
    for t in range(1,n):
        for idx in np.where(member[t]&np.isfinite(ret1[t]))[0]:
            lim=get_limit_pct(codes[idx],dates[t]); limitup[t,idx]=ret1[t,idx]>=lim*.985; limitdn[t,idx]=ret1[t,idx]<=-lim*.985
    streak=np.zeros((n,p),np.int16)
    for t in range(1,n): streak[t]=np.where(limitup[t],streak[t-1]+1,0)
    market=build_regime(dates,member,ret1,limitup,limitdn,close); market.to_csv(OUT/"market_regime.csv",index=False)
    lhb=fetch_lhb(START,END); lhb_map={}
    if len(lhb):
        for r in lhb.itertuples(index=False):
            if isinstance(r.code,str): lhb_map[(pd.Timestamp(r.date).normalize(),r.code)]={"net":float(r.lhb_net) if pd.notna(r.lhb_net) else np.nan,"net_ratio":float(r.lhb_net_ratio) if pd.notna(r.lhb_net_ratio) else np.nan}
        lhb.to_csv(OUT/"lhb_raw_safe_fields.csv",index=False)
    events=fetch_sparse_events(START,END); event_map={}
    if len(events):
        for r in events.itertuples(index=False): event_map[(pd.Timestamp(r.date).normalize(),r.code)]=float(r.event_raw)
        events.to_csv(OUT/"events_sparse.csv",index=False)
    start_i=int(np.searchsorted(dates.values,START.to_datetime64())); end_i=int(np.searchsorted(dates.values,END.to_datetime64(),side="right")-1); cache={}
    for t in range(max(start_i,25),end_i+1):
        cache[t]=compute_daily_candidates(t,dates,codes,member,close,volume,ret1,mom5,volratio,amount_rank,limitup,streak,lhb_map,event_map)
        if t%250==0: print("candidate day",t,"n",len(cache[t]),flush=True)
    summaries=[]; all_trades=[]; all_eq=[]; failed=[]; timing=[]
    for variant in VARIANTS:
        print("SIM",variant,flush=True); cash=INITIAL; pos={}; pending_buys=[]; pending_exits=set(); trades=[]; eq=[]
        for t in range(start_i,end_i+1):
            d=pd.Timestamp(dates[t])
            for idx in list(pending_exits):
                if idx not in pos: continue
                op=open_[t,idx]
                if not np.isfinite(op) or base.open_locked(codes[idx],d,op,close[t-1,idx],"sell"): failed.append([variant,d,codes[idx],"sell","unfilled_open_limit_or_missing"]); continue
                px=op*(1-SLIPPAGE); P=pos.pop(idx); gross=P.units*px; f=fee(gross,"sell",d); cash+=gross-f; pnl=(gross-f)-(P.entry_value+P.entry_fee); trades.append({"strategy":variant,"code":codes[idx],"signal_date":P.signal_date,"entry_date":P.entry_date,"exit_date":d,"entry_px":P.entry_px,"exit_px":px,"entry_value":P.entry_value,"net_pnl":pnl,"net_return":pnl/(P.entry_value+P.entry_fee)})
            pending_exits=set()
            if pending_buys and len(pos)<MAX_NAMES:
                slots=MAX_NAMES-len(pos); state=market.iloc[t-1].state if t>0 else "divergence"; exposure=EXPOSURE[state] if variant.startswith("05_") else .80; marked=sum(P.units*(close[t-1,i] if t>0 and np.isfinite(close[t-1,i]) else P.entry_px) for i,P in pos.items()); target_total=max(0,cash+marked)*exposure; target_each=target_total/MAX_NAMES; accepted=0
                for rec in pending_buys:
                    if accepted>=slots: break
                    idx=int(rec["idx"])
                    if idx in pos or not member[t,idx]: continue
                    op=open_[t,idx]
                    if not np.isfinite(op) or not np.isfinite(close[t-1,idx]): failed.append([variant,d,codes[idx],"buy","missing_or_not_member"]); continue
                    if base.open_locked(codes[idx],d,op,close[t-1,idx],"buy"): failed.append([variant,d,codes[idx],"buy","open_limit_locked"]); continue
                    if variant in ("03_network_auction","04_network_auction_lhb","05_full_regime_network_auction_lhb_event"):
                        gap=op/close[t-1,idx]-1; maxgap=.08 if rec.get("streak",0)>=1 else .055
                        if gap<-.035 or gap>maxgap: failed.append([variant,d,codes[idx],"buy",f"auction_gap_{gap:.4f}"]); continue
                    px=op*(1+SLIPPAGE); cap=PARTICIPATION*max(volume[t-1,idx],0)*max(close[t-1,idx],0); budget=min(target_each,cap,cash*.98); units=int(budget/px//100*100)
                    if units<100: continue
                    gross=units*px; f=fee(gross,"buy",d)
                    if gross+f>cash: continue
                    cash-=gross+f; pos[idx]=Position(units,px,d,pd.Timestamp(rec["signal_date"]),px,gross,f); timing.append([variant,pd.Timestamp(rec["signal_date"]),d,codes[idx],(d-pd.Timestamp(rec["signal_date"])).days]); accepted+=1
                pending_buys=[]
            holdings=0
            for idx,P in pos.items():
                cp=close[t,idx]
                if np.isfinite(cp): P.peak=max(P.peak,cp); holdings+=P.units*cp
                else: holdings+=P.units*P.entry_px
            eq.append({"strategy":variant,"date":d,"equity":cash+holdings,"cash":cash,"positions":len(pos)})
            state=market.iloc[t].state
            for idx,P in list(pos.items()):
                cp=close[t,idx]
                if not np.isfinite(cp): continue
                held=max(0,t-int(np.searchsorted(dates.values,P.entry_date.to_datetime64()))); ctab=cache.get(t,pd.DataFrame()); row=ctab[ctab.idx==idx] if not ctab.empty else pd.DataFrame(); theme_now=float(row.theme.iloc[0]) if len(row) else 0; exit_flag=(held>=5 or cp/P.entry_px-1<=-.08 or cp/max(P.peak,1e-9)-1<=-.07)
                if variant.startswith("05_") and (state=="retreat" or theme_now<.20) and held>=1: exit_flag=True
                if exit_flag: pending_exits.add(idx)
            ranked=score_variant(cache.get(t,pd.DataFrame()),variant)
            if not ranked.empty:
                if variant!="01_leader_only": ranked=ranked[(ranked.peer_count>=3)&(ranked.theme>=.45)]
                if variant.startswith("05_"):
                    ranked=ranked.iloc[0:0] if state=="retreat" else ranked[ranked.score>=.42]
                else: ranked=ranked[ranked.score>=.40]
                pending_buys=[]
                for r in ranked.head(12).to_dict("records"):
                    if int(r["idx"]) not in pos: r["signal_date"]=d; pending_buys.append(r)
        eqdf=pd.DataFrame(eq); st=stats(eqdf,trades); st["strategy"]=variant; summaries.append(st); all_trades.extend(trades); all_eq.append(eqdf); print("RESULT",variant,st,flush=True)
    summary=pd.DataFrame(summaries).sort_values("total_return",ascending=False); bench=np.nan
    if "SH000985" in codes:
        j=codes.index("SH000985"); a=close[start_i,j]; b=close[end_i,j]; bench=b/a-1 if np.isfinite(a) and np.isfinite(b) and a>0 else np.nan
    summary["benchmark_return"]=bench; summary["excess_return"]=summary.total_return-bench if np.isfinite(bench) else np.nan; summary.to_csv(OUT/"summary.csv",index=False); pd.concat(all_eq,ignore_index=True).to_csv(OUT/"equity.csv",index=False); tdf=pd.DataFrame(all_trades); tdf.to_csv(OUT/"trades.csv",index=False); pd.DataFrame(failed,columns=["strategy","date","code","side","reason"]).to_csv(OUT/"failed_fills.csv",index=False); tim=pd.DataFrame(timing,columns=["strategy","signal_date","trade_date","code","lag_days"]); tim.to_csv(OUT/"timing_audit.csv",index=False)
    ae=[]; edf=pd.concat(all_eq,ignore_index=True)
    for s,g in edf.groupby("strategy"):
        g=g.sort_values("date")
        for y,yg in g.groupby(pd.to_datetime(g.date).dt.year): ae.append([s,y,yg.equity.iloc[-1]/yg.equity.iloc[0]-1])
    pd.DataFrame(ae,columns=["strategy","year","return"]).to_csv(OUT/"annual_returns.csv",index=False); rb=[]
    if len(tdf):
        for s,g in tdf.groupby("strategy"):
            g=g.sort_values("net_pnl",ascending=False); pnl=g.net_pnl.sum(); rb.append([s,pnl,g.head(5).net_pnl.sum(),pnl-g.head(5).net_pnl.sum(),g.head(10).net_pnl.sum(),pnl-g.head(10).net_pnl.sum()])
    pd.DataFrame(rb,columns=["strategy","completed_pnl","best5_pnl","pnl_without_best5","best10_pnl","pnl_without_best10"]).to_csv(OUT/"robustness.csv",index=False)
    audit={"release_tag":base.RELEASE_TAG,"start":str(START.date()),"end":str(END.date()),"union_members":int(member[start_i:end_i+1].any(axis=0).sum()),"min_daily_members":int(member[start_i:end_i+1].sum(axis=1).min()),"max_daily_members":int(member[start_i:end_i+1].sum(axis=1).max()),"trade_timing_violations":int((pd.to_datetime(tim.trade_date)<=pd.to_datetime(tim.signal_date)).sum()) if len(tim) else 0,"min_trade_lag_days":float(tim.lag_days.min()) if len(tim) else np.nan,"next_day_volume_used_for_capacity":0,"lhb_future_return_fields_used":0,"lhb_vendor_success_rate_used":0,"historical_limit_pool_endpoint_used":0,"theme_network":"rolling 15d residual-return co-movement among T-close attention set","auction_proxy":"T+1 official open used only after T-close candidate list frozen; no later intraday data","news_layer":"sparse CNINFO title queries, actual announcement dates, no future-return labels"}; pd.DataFrame([audit]).to_csv(OUT/"audit.csv",index=False); pd.DataFrame([{"variant":v} for v in VARIANTS]).to_csv(OUT/"variants.csv",index=False); print("=== AUDIT ==="); print(pd.DataFrame([audit]).to_string(index=False)); print("=== SUMMARY ==="); print(summary.to_string(index=False)); print("=== ROBUSTNESS ==="); print(pd.read_csv(OUT/"robustness.csv").to_string(index=False))

if __name__=="__main__": main()
