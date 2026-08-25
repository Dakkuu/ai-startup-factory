from __future__ import annotations
import math, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import akshare as ak
import run_10y_china_behavior_daily as base
import run_10y_a_share_flow_network as m

warnings.filterwarnings("ignore")
OUT=Path("results_10y_flow_network_strict"); OUT.mkdir(exist_ok=True); m.OUT=OUT
START=pd.Timestamp("2016-07-29"); END=pd.Timestamp("2026-07-29"); INITIAL=1_000_000.0
MAX_NAMES=5; SLIPPAGE=.0020; PARTICIPATION=.02
VARIANTS=[
 "01_leader_only_strict",
 "02_network_leader_strict",
 "03_network_lhb_event_strict",
 "04_full_regime_network_lhb_event_strict",
 "05_full_regime_network_lhb_event_strict_auction_T2",
]

def _lhb_one(y):
    a=max(START,pd.Timestamp(f"{y}-01-01")); b=min(END,pd.Timestamp(f"{y}-12-31"))
    last=None
    for k in range(3):
        try:
            df=ak.stock_lhb_detail_em(start_date=a.strftime("%Y%m%d"),end_date=b.strftime("%Y%m%d"))
            if df is None or df.empty:return y,pd.DataFrame(),"empty"
            keep=["代码","上榜日","龙虎榜净买额","龙虎榜买入额","龙虎榜卖出额","龙虎榜成交额","市场总成交额","净买额占总成交比","成交额占总成交比","换手率","流通市值","上榜原因"]
            df=df[[c for c in keep if c in df.columns]].copy(); df["code"]=df["代码"].map(m.map_code); df["date"]=pd.to_datetime(df["上榜日"])
            df=df.dropna(subset=["code","date"])
            agg=df.groupby(["date","code"],as_index=False).agg(lhb_net=("龙虎榜净买额","first"),lhb_buy=("龙虎榜买入额","first"),lhb_sell=("龙虎榜卖出额","first"),lhb_deal=("龙虎榜成交额","first"),lhb_market=("市场总成交额","first"),lhb_net_ratio=("净买额占总成交比","first"),lhb_deal_ratio=("成交额占总成交比","first"),lhb_turnover=("换手率","first"),lhb_floatcap=("流通市值","first"),reason_count=("上榜原因","nunique"))
            return y,agg,"ok"
        except Exception as e:
            last=type(e).__name__; time.sleep(1.5*(k+1))
    return y,pd.DataFrame(),"error:"+str(last)

def fetch_lhb_parallel():
    rows=[]; audit=[]
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs=[ex.submit(_lhb_one,y) for y in range(START.year,END.year+1)]
        for f in as_completed(futs):
            y,df,st=f.result(); audit.append((y,len(df),st)); print("LHB",y,len(df),st,flush=True)
            if len(df): rows.append(df)
    pd.DataFrame(audit,columns=["year","rows","status"]).sort_values("year").to_csv(OUT/"lhb_coverage.csv",index=False)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

EVENT_QUERIES={"业绩预增":1.0,"扭亏":.9,"中标":.7,"重大合同":.8,"增持计划":.7,"立案调查":-1.2,"终止重组":-.9}
def _event_one(y,kw,score):
    a=max(START,pd.Timestamp(f"{y}-01-01")); b=min(END,pd.Timestamp(f"{y}-12-31")); last=None
    for k in range(3):
        try:
            df=ak.stock_zh_a_disclosure_report_cninfo(symbol="",market="沪深京",keyword=kw,category="",start_date=a.strftime("%Y%m%d"),end_date=b.strftime("%Y%m%d"))
            n=0 if df is None else len(df)
            if not n:return y,kw,pd.DataFrame(),"ok"
            z=df[["代码","简称","公告标题","公告时间"]].copy(); z["code"]=z["代码"].map(m.map_code); z["announce_time"]=pd.to_datetime(z["公告时间"],errors="coerce"); z["date"]=z["announce_time"].dt.normalize(); z["event_raw"]=score; z["keyword"]=kw; z=z.dropna(subset=["date","code"])
            return y,kw,z[["date","code","event_raw","keyword","公告标题","announce_time"]],"ok"
        except Exception as e:
            last=type(e).__name__; time.sleep(2*(k+1))
    return y,kw,pd.DataFrame(),"error:"+str(last)

def fetch_events_parallel():
    rows=[]; audit=[]; jobs=[]
    for y in range(max(2017,START.year),END.year+1):
        for kw,sc in EVENT_QUERIES.items():jobs.append((y,kw,sc))
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs=[ex.submit(_event_one,*j) for j in jobs]
        for f in as_completed(futs):
            y,kw,z,st=f.result(); audit.append((y,kw,len(z),st)); print("EVENT",y,kw,len(z),st,flush=True)
            if len(z):rows.append(z)
    pd.DataFrame(audit,columns=["year","keyword","rows","status"]).sort_values(["year","keyword"]).to_csv(OUT/"event_coverage.csv",index=False)
    if not rows:return pd.DataFrame()
    ev=pd.concat(rows,ignore_index=True)
    ev2=ev.groupby(["date","code"],as_index=False).agg(event_raw=("event_raw","sum"),event_hits=("keyword","nunique")); ev2["event_raw"]=ev2.event_raw.clip(-2,2)
    return ev2

def score_variant(df,v):
    if df.empty:return df
    x=df.copy()
    if v.startswith("01_"):x["score"]=.55*x.leader+.25*x.base+.20*x.capacity-.18*x.crowd
    elif v.startswith("02_"):x["score"]=.35*x.theme+.35*x.leader+.20*x.capacity+.10*x.base-.18*x.crowd
    elif v.startswith("03_"):x["score"]=.29*x.theme+.29*x.leader+.18*x.capacity+.08*x.base+.10*x.lhb+.06*x.event-.20*x.crowd
    else:x["score"]=.27*x.theme+.27*x.leader+.18*x.capacity+.06*x.base+.10*x.lhb+.12*x.event-.22*x.crowd
    return x.sort_values("score",ascending=False)

def main():
    dates,codes,close,open_,high,volume,factor,member,load_audit=base.load_data(); dates=pd.DatetimeIndex(dates); codes=list(codes); member=member.astype(bool)
    stock_mask=np.array([bool(base.STOCK_RE.match(c)) for c in codes]); member[:,~stock_mask]=False
    n,p=close.shape; print("DATA",n,p,"stock_union",int(member.any(axis=0).sum()),flush=True)
    ret1=np.full_like(close,np.nan,float); ret1[1:]=close[1:]/close[:-1]-1
    mom5=np.full_like(close,np.nan,float); mom5[5:]=close[5:]/close[:-5]-1
    vma20=m.rolling_mean_mat(volume,20); volratio=np.divide(volume,vma20,out=np.full_like(volume,np.nan,float),where=vma20>0)
    liq=np.log1p(np.maximum(volume*close,0)); amount_rank=np.full_like(close,np.nan,float)
    for t in range(n):
        ids=np.where(member[t]&np.isfinite(liq[t]))[0]; amount_rank[t,ids]=m.pct_rank(liq[t,ids])
    limitup=np.zeros((n,p),bool); limitdn=np.zeros((n,p),bool)
    for t in range(1,n):
        for idx in np.where(member[t]&np.isfinite(ret1[t]))[0]:
            lim=m.get_limit_pct(codes[idx],dates[t]); limitup[t,idx]=ret1[t,idx]>=lim*.985; limitdn[t,idx]=ret1[t,idx]<=-lim*.985
    streak=np.zeros((n,p),np.int16)
    for t in range(1,n):streak[t]=np.where(limitup[t],streak[t-1]+1,0)
    market=m.build_regime(dates,member,ret1,limitup,limitdn,close); market.to_csv(OUT/"market_regime.csv",index=False)
    lhb=fetch_lhb_parallel(); lhb_map={}
    if len(lhb):
        for r in lhb.itertuples(index=False):
            lhb_map[(pd.Timestamp(r.date).normalize(),r.code)]={"net":float(r.lhb_net) if pd.notna(r.lhb_net) else np.nan,"net_ratio":float(r.lhb_net_ratio) if pd.notna(r.lhb_net_ratio) else np.nan}
        lhb.to_csv(OUT/"lhb_raw_safe_fields.csv",index=False)
    events=fetch_events_parallel(); event_map={}
    if len(events):
        for r in events.itertuples(index=False):event_map[(pd.Timestamp(r.date).normalize(),r.code)]=float(r.event_raw)
        events.to_csv(OUT/"events_sparse.csv",index=False)
    start_i=int(np.searchsorted(dates.values,START.to_datetime64())); end_i=int(np.searchsorted(dates.values,END.to_datetime64(),side="right")-1)
    cache={}
    for t in range(max(start_i,25),end_i+1):
        cache[t]=m.compute_daily_candidates(t,dates,codes,member,close,volume,ret1,mom5,volratio,amount_rank,limitup,streak,lhb_map,event_map)
        if t%250==0:print("candidate",t,len(cache[t]),flush=True)
    summaries=[]; alltr=[]; alleq=[]; failed=[]; timing=[]; confirms=[]
    for v in VARIANTS:
        print("SIM",v,flush=True); cash=INITIAL; pos={}; pending=[]; pending_auction=[]; confirmed_to_buy=[]; pending_exits=set(); trades=[]; eq=[]
        for t in range(start_i,end_i+1):
            d=pd.Timestamp(dates[t])
            # exits decided at previous close, filled at today's open
            for idx in list(pending_exits):
                if idx not in pos:continue
                op=open_[t,idx]
                if not np.isfinite(op) or base.open_locked(codes[idx],d,op,close[t-1,idx],"sell"):
                    failed.append([v,d,codes[idx],"sell","unfilled_open_limit_or_missing"]); continue
                px=op*(1-SLIPPAGE); P=pos.pop(idx); gross=P.units*px; f=m.fee(gross,"sell",d); cash+=gross-f; pnl=(gross-f)-(P.entry_value+P.entry_fee)
                trades.append({"strategy":v,"code":codes[idx],"signal_date":P.signal_date,"entry_date":P.entry_date,"exit_date":d,"entry_px":P.entry_px,"exit_px":px,"entry_value":P.entry_value,"net_pnl":pnl,"net_return":pnl/(P.entry_value+P.entry_fee)})
            pending_exits=set()
            # strict auction variant: candidates confirmed using yesterday's final open; only now (one full session later) may be bought
            buylist=confirmed_to_buy if v.startswith("05_") else pending
            if buylist and len(pos)<MAX_NAMES:
                slots=MAX_NAMES-len(pos); state=market.iloc[t-1].state if t>0 else "divergence"; exposure=m.EXPOSURE[state] if v.startswith(("04_","05_")) else .80
                marked=sum(P.units*(close[t-1,i] if t>0 and np.isfinite(close[t-1,i]) else P.entry_px) for i,P in pos.items()); target_total=max(0,cash+marked)*exposure; target_each=target_total/MAX_NAMES; accepted=0
                for rec in buylist:
                    if accepted>=slots:break
                    idx=int(rec["idx"])
                    if idx in pos or not member[t,idx]:continue
                    op=open_[t,idx]
                    if not np.isfinite(op) or not np.isfinite(close[t-1,idx]):failed.append([v,d,codes[idx],"buy","missing_or_not_member"]);continue
                    if base.open_locked(codes[idx],d,op,close[t-1,idx],"buy"):failed.append([v,d,codes[idx],"buy","open_limit_locked"]);continue
                    px=op*(1+SLIPPAGE); cap=PARTICIPATION*max(volume[t-1,idx],0)*max(close[t-1,idx],0); budget=min(target_each,cap,cash*.98); units=int(budget/px//100*100)
                    if units<100:continue
                    gross=units*px; f=m.fee(gross,"buy",d)
                    if gross+f>cash:continue
                    cash-=gross+f; pos[idx]=m.Position(units,px,d,pd.Timestamp(rec["signal_date"]),px,gross,f); timing.append([v,pd.Timestamp(rec["signal_date"]),rec.get("confirm_date",pd.NaT),d,codes[idx],t-int(rec["signal_t"])]);accepted+=1
            pending=[]; confirmed_to_buy=[]
            # Observe today's final open only to confirm yesterday's auction candidates; no same-open fill is allowed.
            next_confirmed=[]
            if v.startswith("05_") and pending_auction:
                for rec in pending_auction:
                    idx=int(rec["idx"])
                    if not member[t,idx] or not np.isfinite(open_[t,idx]) or not np.isfinite(close[t-1,idx]):continue
                    gap=open_[t,idx]/close[t-1,idx]-1; maxgap=.08 if rec.get("streak",0)>=1 else .055
                    if gap<-.035 or gap>maxgap:failed.append([v,d,codes[idx],"confirm","auction_gap_reject"]);continue
                    q=dict(rec); q["confirm_date"]=d; next_confirmed.append(q); confirms.append([v,pd.Timestamp(rec["signal_date"]),d,codes[idx],gap])
                pending_auction=[]
            holdings=0
            for idx,P in pos.items():
                cp=close[t,idx]
                if np.isfinite(cp):P.peak=max(P.peak,cp);holdings+=P.units*cp
                else:holdings+=P.units*P.entry_px
            eq.append({"strategy":v,"date":d,"equity":cash+holdings,"cash":cash,"positions":len(pos)})
            state=market.iloc[t].state
            for idx,P in list(pos.items()):
                cp=close[t,idx]
                if not np.isfinite(cp):continue
                held=max(0,t-int(np.searchsorted(dates.values,P.entry_date.to_datetime64()))); ctab=cache.get(t,pd.DataFrame()); row=ctab[ctab.idx==idx] if not ctab.empty else pd.DataFrame(); theme_now=float(row.theme.iloc[0]) if len(row) else 0
                exit_flag=(held>=5 or cp/P.entry_px-1<=-.08 or cp/max(P.peak,1e-9)-1<=-.07)
                if v.startswith(("04_","05_")) and (state=="retreat" or theme_now<.20) and held>=1:exit_flag=True
                if exit_flag:pending_exits.add(idx)
            ranked=score_variant(cache.get(t,pd.DataFrame()),v)
            if not ranked.empty:
                if not v.startswith("01_"):ranked=ranked[(ranked.peer_count>=3)&(ranked.theme>=.45)]
                if v.startswith(("04_","05_")):ranked=ranked.iloc[0:0] if state=="retreat" else ranked[ranked.score>=.42]
                else:ranked=ranked[ranked.score>=.40]
                picks=[]
                for r in ranked.head(12).to_dict("records"):
                    if int(r["idx"]) not in pos:r["signal_date"]=d;r["signal_t"]=t;picks.append(r)
                if v.startswith("05_"):pending_auction=picks
                else:pending=picks
            if v.startswith("05_"):confirmed_to_buy=next_confirmed
        eqdf=pd.DataFrame(eq); st=m.stats(eqdf,trades); st["strategy"]=v; summaries.append(st); alltr.extend(trades); alleq.append(eqdf); print("RESULT",v,st,flush=True)
    summary=pd.DataFrame(summaries).sort_values("total_return",ascending=False)
    bench=np.nan
    if "SH000985" in codes:
        j=codes.index("SH000985"); a=close[start_i,j]; b=close[end_i,j]; bench=b/a-1 if np.isfinite(a) and np.isfinite(b) and a>0 else np.nan
    # dynamic equal-weight daily benchmark, no transaction costs; context only
    ew=[]
    for t in range(start_i+1,end_i+1):
        mm=member[t-1]&member[t]&np.isfinite(ret1[t]); ew.append(np.nanmean(ret1[t,mm]) if mm.any() else 0.0)
    ewret=float(np.prod(1+np.asarray(ew))-1) if ew else np.nan
    summary["csi_allshare_return"]=bench; summary["dynamic_equal_weight_return"]=ewret; summary["excess_vs_csi_allshare"]=summary.total_return-bench if np.isfinite(bench) else np.nan
    summary.to_csv(OUT/"summary.csv",index=False); pd.concat(alleq,ignore_index=True).to_csv(OUT/"equity.csv",index=False); tdf=pd.DataFrame(alltr); tdf.to_csv(OUT/"trades.csv",index=False)
    pd.DataFrame(failed,columns=["strategy","date","code","side","reason"]).to_csv(OUT/"failed_fills.csv",index=False)
    tim=pd.DataFrame(timing,columns=["strategy","signal_date","confirm_date","trade_date","code","trade_session_lag"]); tim.to_csv(OUT/"timing_audit.csv",index=False)
    pd.DataFrame(confirms,columns=["strategy","signal_date","confirm_date","code","observed_open_gap"]).to_csv(OUT/"auction_confirmations.csv",index=False)
    ar=[]; edf=pd.concat(alleq,ignore_index=True)
    for s,g in edf.groupby("strategy"):
        g=g.sort_values("date")
        for y,yg in g.groupby(pd.to_datetime(g.date).dt.year):ar.append([s,y,yg.equity.iloc[-1]/yg.equity.iloc[0]-1])
    pd.DataFrame(ar,columns=["strategy","year","return"]).to_csv(OUT/"annual_returns.csv",index=False)
    rb=[]
    if len(tdf):
        for s,g in tdf.groupby("strategy"):
            g=g.sort_values("net_pnl",ascending=False); pnl=g.net_pnl.sum(); rb.append([s,pnl,g.head(5).net_pnl.sum(),pnl-g.head(5).net_pnl.sum(),g.head(10).net_pnl.sum(),pnl-g.head(10).net_pnl.sum()])
    pd.DataFrame(rb,columns=["strategy","completed_pnl","best5_pnl","pnl_without_best5","best10_pnl","pnl_without_best10"]).to_csv(OUT/"robustness.csv",index=False)
    auction_tim=tim[tim.strategy.str.startswith("05_")] if len(tim) else pd.DataFrame()
    audit={"release_tag":base.RELEASE_TAG,"start":str(START.date()),"end":str(END.date()),"stock_union":int(member[start_i:end_i+1].any(axis=0).sum()),"min_daily_stocks":int(member[start_i:end_i+1].sum(axis=1).min()),"max_daily_stocks":int(member[start_i:end_i+1].sum(axis=1).max()),"trade_timing_violations":int((pd.to_datetime(tim.trade_date)<=pd.to_datetime(tim.signal_date)).sum()) if len(tim) else 0,"min_trade_session_lag":int(tim.trade_session_lag.min()) if len(tim) else -1,"strict_auction_min_session_lag":int(auction_tim.trade_session_lag.min()) if len(auction_tim) else -1,"same_open_used_for_filter_and_fill":0,"next_day_volume_used_for_capacity":0,"lhb_future_return_fields_used":0,"lhb_vendor_success_rate_used":0,"historical_limit_pool_endpoint_used":0,"theme_network":"15-session residual-return co-movement among T-date attention candidates","lhb_rule":"T-date raw leaderboard fields may affect T+1-or-later trades only","event_rule":"CNINFO timestamp date may affect next-session-or-later trades only","auction_rule":"T+1 final open may only confirm; earliest fill is T+2 open"}
    pd.DataFrame([audit]).to_csv(OUT/"audit.csv",index=False); pd.DataFrame([{"variant":v} for v in VARIANTS]).to_csv(OUT/"variants.csv",index=False)
    print("=== AUDIT ==="); print(pd.DataFrame([audit]).to_string(index=False)); print("=== SUMMARY ==="); print(summary.to_string(index=False)); print("=== ROBUSTNESS ==="); print(pd.read_csv(OUT/"robustness.csv").to_string(index=False) if (OUT/"robustness.csv").exists() else "none")

if __name__=="__main__":main()
