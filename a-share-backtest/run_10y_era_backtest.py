from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import run_backtest_qlib as qb

RELEASE_TAG="2026-07-29"
START=pd.Timestamp("2016-07-29")
END=pd.Timestamp("2026-07-29")
WARM=START-pd.Timedelta(days=260)
INITIAL_CASH=1_000_000.0
COMMISSION=.00025
MIN_COMMISSION=5.0
SLIPPAGE=.0010
VOLUME_PARTICIPATION=.05
MIN_LIQ=20_000_000.0
OUT=Path("results_10y_era")
OUT.mkdir(exist_ok=True)
STOCK_RE=re.compile(r'^(?:SH(?:600|601|603|605|688)\d{3}|SZ(?:000|001|002|003|300|301)\d{3}|BJ\d{6})$')
FEATURES=["r_ret1","r_mom5","r_mom20","r_mom60","r_mom120","r_lowvol20","r_lowvol60","r_volratio","r_liq","r_ma20gap","r_ma60gap","r_high20"]

AGENTS=[
 {"strategy":"01_defensive_institution","risk":"low","max_names":20,"invest":.65},
 {"strategy":"02_breakout_swing","risk":"low_mid","max_names":12,"invest":.80},
 {"strategy":"03_smart_money_trend","risk":"mid","max_names":15,"invest":.85},
 {"strategy":"04_era_appropriate_quant","risk":"mid","max_names":30,"invest":.90},
 {"strategy":"05_retail_attention_chase","risk":"mid_high","max_names":8,"invest":.95},
 {"strategy":"06_hot_money_relay","risk":"high","max_names":5,"invest":.95},
 {"strategy":"07_limit_up_relay","risk":"very_high","max_names":4,"invest":.90},
 {"strategy":"08_panic_reversal","risk":"high_contrarian","max_names":8,"invest":.75},
 {"strategy":"09_medium_term_momentum","risk":"mid","max_names":20,"invest":.85},
 {"strategy":"10_regime_adaptive","risk":"adaptive","max_names":20,"invest":.85},
]

def era_name(d):
    y=pd.Timestamp(d).year
    if y<=2018:return "2016-2018_handcrafted"
    if y<=2020:return "2019-2020_rolling_ridge"
    if y<=2022:return "2021-2022_rolling_lightgbm"
    return "2023-2026_rolling_ensemble"

def fee(gross,side,d):
    d=pd.Timestamp(d)
    stamp=.001 if d < pd.Timestamp("2023-08-28") else .0005
    transfer=.00002 if d < pd.Timestamp("2022-04-29") else .00001
    return max(MIN_COMMISSION,gross*COMMISSION)+gross*transfer+(gross*stamp if side=="sell" else 0.0)

def load_base():
    qb.RELEASE_TAG=RELEASE_TAG; qb.ROOT=Path("qlib_data")
    qb.download_and_extract()
    cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(qb.ROOT/"calendars"/"day.txt",header=None)[0]))
    m=pd.read_csv(qb.ROOT/"instruments"/"all.txt",sep="\t",header=None,names=["code","start","end"],usecols=[0,1,2])
    m["code"]=m.code.astype(str).str.upper(); m["start"]=pd.to_datetime(m.start); m["end"]=pd.to_datetime(m.end)
    m=m[m.code.str.match(STOCK_RE)].copy(); m=m[(m.end>=WARM)&(m.start<=END)]
    t=cal[(cal>=START)&(cal<=END)]
    cnt=[int(((m.start<=d)&(m.end>=d)).sum()) for d in t]
    audit={"release_tag":RELEASE_TAG,"start":str(START.date()),"end":str(END.date()),"union_members":int(m[(m.end>=START)&(m.start<=END)].code.nunique()),"entered":int(m[(m.start>START)&(m.start<=END)].code.nunique()),"exited":int(m[(m.end>=START)&(m.end<END)].code.nunique()),"min_daily_members":min(cnt),"max_daily_members":max(cnt)}
    if audit["union_members"]<3000 or audit["exited"]<50 or min(cnt)<2500:raise RuntimeError(f"FAIL-CLOSED universe {audit}")
    pd.DataFrame([audit]).to_csv(OUT/"universe_audit.csv",index=False)
    return cal,m,audit

def active_for(mm, d):
    return bool(((mm.start<=d)&(mm.end>=d)).any())

def build_weekly_panel(cal,m):
    test=cal[(cal>=START)&(cal<=END)]
    signal_dates=pd.DatetimeIndex(test[::5])
    # ensure enough warm data; trading at the first exchange day after signal.
    alltrade=cal[cal<=END]
    exec_dates=[]
    for s in signal_dates:
        k=alltrade.searchsorted(s,side="right")
        exec_dates.append(alltrade[k] if k<len(alltrade) else pd.NaT)
    exec_dates=pd.DatetimeIndex(exec_dates)
    exec_map=dict(zip(signal_dates,exec_dates))
    next_exec=dict(zip(signal_dates[:-1],exec_dates[1:]))
    frames=[]
    codes=sorted(m.code.unique())
    for i,code in enumerate(codes,1):
        cols={}
        for f in ["open","high","low","close","volume","factor"]:
            s=qb.read_bin(code,f,cal)
            if not s.empty:cols[f]=s
        if not all(f in cols for f in ["open","high","low","close","volume"]):continue
        z=pd.concat(cols,axis=1).loc[WARM:END].copy()
        if z.empty:continue
        if "factor" not in z:z["factor"]=1.0
        z["factor"]=z.factor.replace(0,np.nan).fillna(1.0)
        r1=z.close.pct_change()
        z["ret1"]=r1; z["mom5"]=z.close.pct_change(5); z["mom20"]=z.close.pct_change(20); z["mom60"]=z.close.pct_change(60); z["mom120"]=z.close.pct_change(120)
        z["vol20"]=r1.rolling(20).std(); z["vol60"]=r1.rolling(60).std()
        z["vol_ma20"]=z.volume.rolling(20).mean(); z["vol_ratio"]=z.volume/z.vol_ma20
        z["liq_ma20"]=(z.close.abs()*z.volume.abs()).rolling(20).mean()
        z["ma20gap"]=z.close/z.close.rolling(20).mean()-1; z["ma60gap"]=z.close/z.close.rolling(60).mean()-1
        z["high20"]=z.close/z.high.shift(1).rolling(20).max()-1
        z["drawdown120"]=z.close/z.close.rolling(120).max()-1
        mm=m[m.code==code]
        rec=[]
        for s in signal_dates:
            e=exec_map.get(s,pd.NaT)
            if pd.isna(e) or s not in z.index or e not in z.index:continue
            if not active_for(mm,s) or not active_for(mm,e):continue
            rr=z.loc[s]
            needed=["ret1","mom5","mom20","mom60","mom120","vol20","vol60","vol_ratio","liq_ma20","ma20gap","ma60gap","high20"]
            if not all(np.isfinite(rr.get(x,np.nan)) for x in needed):continue
            er=z.loc[e]
            ne=next_exec.get(s,pd.NaT)
            label=np.nan
            if pd.notna(ne) and ne in z.index and active_for(mm,ne):
                op0=float(er.open); op1=float(z.loc[ne].open)
                if np.isfinite(op0) and np.isfinite(op1) and op0>0:label=op1/op0-1
            rec.append({"signal_date":s,"trade_date":e,"code":code,
                **{x:float(rr[x]) for x in needed},"drawdown120":float(rr.drawdown120) if np.isfinite(rr.drawdown120) else np.nan,
                "signal_close":float(rr.close),"exec_open":float(er.open),"exec_high":float(er.high),"exec_low":float(er.low),"exec_close":float(er.close),"exec_volume":float(er.volume),"exec_factor":float(er.factor) if np.isfinite(er.factor) and er.factor>0 else 1.0,
                "label":label,"label_exit_date":ne})
        if rec:frames.append(pd.DataFrame(rec))
        if i%500==0:print("weekly features",i,"/",len(codes),flush=True)
    if not frames:raise RuntimeError("no weekly features")
    p=pd.concat(frames,ignore_index=True)
    p=p[(p.trade_date<=END)&(p.signal_date>=START)].copy()
    # Cross-sectional point-in-time ranks.
    def ranks(g):
        g=g.copy()
        def rp(s):return s.rank(pct=True,method="average")
        g["r_ret1"]=rp(g.ret1);g["r_mom5"]=rp(g.mom5);g["r_mom20"]=rp(g.mom20);g["r_mom60"]=rp(g.mom60);g["r_mom120"]=rp(g.mom120)
        g["r_lowvol20"]=rp(-g.vol20);g["r_lowvol60"]=rp(-g.vol60);g["r_volratio"]=rp(g.vol_ratio);g["r_liq"]=rp(g.liq_ma20);g["r_ma20gap"]=rp(g.ma20gap);g["r_ma60gap"]=rp(g.ma60gap);g["r_high20"]=rp(g.high20)
        return g
    p=p.groupby("signal_date",group_keys=False).apply(ranks,include_groups=False).reset_index(drop=True)
    p.to_pickle(OUT/"weekly_panel.pkl")
    return p,signal_dates

def benchmark_state(cal, panel):
    b=qb.read_bin("SH000985","close",cal).loc[WARM:END].dropna()
    out=[]
    for d,g in panel.groupby("signal_date"):
        if d not in b.index:continue
        pos=b.index.get_loc(d)
        mom60=(float(b.iloc[pos]/b.iloc[pos-60]-1) if pos>=60 else np.nan)
        mom120=(float(b.iloc[pos]/b.iloc[pos-120]-1) if pos>=120 else np.nan)
        breadth=float((g.ma60gap>0).mean())
        if np.isfinite(mom60) and mom60>.05 and breadth>.55:reg="hot"
        elif np.isfinite(mom60) and mom60<0 and breadth<.45:reg="cold"
        else:reg="neutral"
        out.append({"signal_date":d,"benchmark_close":float(b.loc[d]),"benchmark_mom60":mom60,"benchmark_mom120":mom120,"breadth_ma60":breadth,"regime":reg,"era":era_name(d)})
    s=pd.DataFrame(out)
    s.to_csv(OUT/"market_states.csv",index=False)
    return s

class EraQuant:
    def __init__(self,panel):
        self.p=panel.copy();self.cache={};self.audit=[]
    def score(self,d,current):
        d=pd.Timestamp(d); era=era_name(d)
        if era=="2016-2018_handcrafted":
            return (.18*current.r_mom20+.22*current.r_mom60+.18*current.r_mom120+.16*current.r_lowvol20+.10*current.r_lowvol60+.10*current.r_liq+.06*current.r_volratio).to_numpy()
        # retrain at most once per calendar quarter; predictions still use current-day features only.
        key=(era,d.year,(d.month-1)//3)
        models=self.cache.get(key)
        if models is None:
            years=2 if era=="2019-2020_rolling_ridge" else (3 if era=="2021-2022_rolling_lightgbm" else 2)
            cutoff=d
            train=self.p[(self.p.signal_date>=d-pd.DateOffset(years=years))&(self.p.label_exit_date<cutoff)&self.p.label.notna()].copy()
            train=train.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+["label"])
            if len(train)>120000:train=train.sample(120000,random_state=d.year*100+d.month)
            if len(train)<5000:raise RuntimeError(f"insufficient matured training data {d.date()} {len(train)}")
            X=train[FEATURES].to_numpy();y=train.label.clip(-.30,.30).to_numpy()
            ridge=Ridge(alpha=8.0).fit(X,y)
            lgb=None
            if era!="2019-2020_rolling_ridge":
                lgb=LGBMRegressor(n_estimators=140,num_leaves=31,learning_rate=.035,max_depth=-1,subsample=.8,colsample_bytree=.8,reg_lambda=2.0,verbosity=-1,random_state=7,n_jobs=2)
                lgb.fit(X,y)
            models=(ridge,lgb)
            self.cache[key]=models
            max_exit=pd.to_datetime(train.label_exit_date).max()
            violation=not(max_exit<d)
            self.audit.append({"train_asof":d,"era":era,"train_rows":len(train),"max_label_exit_date":max_exit,"timing_violation":int(violation)})
            if violation:raise RuntimeError("LOOKAHEAD VIOLATION in model training")
        Xc=current[FEATURES].to_numpy();ridge,lgb=models
        pr=ridge.predict(Xc)
        if lgb is None:return pr
        pl=lgb.predict(Xc)
        return (.40*pr+.60*pl) if era=="2023-2026_rolling_ensemble" else pl

def target_for(g,state,agent,qscore):
    x=g.copy();x["qscore"]=qscore
    x=x[(x.liq_ma20>=MIN_LIQ)&(x.exec_open>0)&np.isfinite(x.exec_open)].copy()
    s=agent["strategy"];reg=state["regime"]
    if s=="01_defensive_institution":
        x=x[(x.mom120>-.10)&(x.ret1<.07)&(x.vol_ratio<3)]
        x["score"]=.34*x.r_lowvol20+.22*x.r_lowvol60+.24*x.r_liq+.10*x.r_mom120+.10*x.r_mom60
    elif s=="02_breakout_swing":
        x=x[(x.mom20>0)&(x.high20>-.025)&(x.vol_ratio>1.15)&(x.ret1<.095)]
        x["score"]=.30*x.r_high20+.25*x.r_volratio+.20*x.r_mom20+.15*x.r_mom60+.10*x.r_liq
    elif s=="03_smart_money_trend":
        x=x[(x.mom60>0)&(x.mom120>0)&(x.ma60gap>0)&(x.vol_ratio.between(.7,2.8))]
        x["score"]=.28*x.r_liq+.22*x.r_mom120+.20*x.r_mom60+.18*x.r_lowvol20+.12*x.r_mom20
    elif s=="04_era_appropriate_quant":
        x["score"]=x.qscore
    elif s=="05_retail_attention_chase":
        x=x[(x.ret1>.01)&(x.mom5>0)&(x.vol_ratio>1.5)&(x.high20>-.04)]
        x["score"]=.30*x.r_volratio+.25*x.r_ret1+.20*x.r_high20+.15*x.r_mom5+.10*x.r_mom20
    elif s=="06_hot_money_relay":
        x=x[(x.mom5>.06)&(x.ret1>.015)&(x.ret1<.19)&(x.vol_ratio>1.35)&(x.high20>-.05)]
        x["score"]=.30*x.r_mom5+.25*x.r_ret1+.25*x.r_volratio+.10*x.r_high20+.10*x.r_liq
    elif s=="07_limit_up_relay":
        if reg!="hot":return x.iloc[0:0]
        x=x[(x.ret1>=.095)&(x.mom20>0)&(x.vol_ratio>.8)]
        x["score"]=.35*x.r_ret1+.25*x.r_volratio+.20*x.r_mom20+.20*x.r_liq
    elif s=="08_panic_reversal":
        panic=(x.ret1<=-.07)|(x.mom5<=-.12)
        x=x[panic&(x.vol_ratio>1.15)&(x.mom120>-.35)]
        x["score"]=.40*(1-x.r_ret1)+.20*x.r_volratio+.20*x.r_liq+.20*x.r_mom120
    elif s=="09_medium_term_momentum":
        x=x[(x.mom60>0)&(x.mom120>0)&(x.ret1<.095)]
        x["score"]=.35*x.r_mom120+.30*x.r_mom60+.15*x.r_mom20+.10*x.r_liq+.10*x.r_lowvol20
    elif s=="10_regime_adaptive":
        if reg=="cold":
            x=x[(x.mom120>-.08)&(x.ret1<.06)]
            x["score"]=.35*x.r_lowvol20+.25*x.r_liq+.20*x.r_mom120+.10*x.r_mom60+.10*x.qscore.rank(pct=True)
        elif reg=="hot":
            x=x[(x.mom20>0)&(x.vol_ratio>1.0)]
            x["score"]=.25*x.r_mom20+.20*x.r_mom60+.20*x.r_volratio+.15*x.r_high20+.20*x.qscore.rank(pct=True)
        else:
            x=x[(x.mom60>-.03)]
            x["score"]=.25*x.r_lowvol20+.20*x.r_liq+.20*x.r_mom60+.15*x.r_mom120+.20*x.qscore.rank(pct=True)
    else:raise ValueError(s)
    return x.sort_values("score",ascending=False).head(agent["max_names"])

def exposure(agent,reg):
    base=agent["invest"]
    if agent["strategy"] in {"05_retail_attention_chase","06_hot_money_relay","07_limit_up_relay"}:
        return base if reg=="hot" else (base*.65 if reg=="neutral" else base*.30)
    if agent["strategy"]=="01_defensive_institution":return min(base, .75 if reg!="cold" else .55)
    return base if reg!="cold" else base*.60

@dataclass
class Pos:
    units:float
    entry_cost:float
    entry_date:pd.Timestamp
    last_price:float

def locked(r,side):
    if r is None:return True
    vals=[r.exec_open,r.exec_high,r.exec_low,r.exec_close,r.signal_close]
    if not all(np.isfinite(v) and v>0 for v in vals):return True
    one=abs(r.exec_high-r.exec_low)<1e-8 and abs(r.exec_open-r.exec_high)<1e-8
    if not one:return False
    pct=r.exec_open/r.signal_close-1
    return (side=="buy" and pct>=.045) or (side=="sell" and pct<=-.045)

def simulate(panel,states,agent,quant):
    by={pd.Timestamp(d):g.set_index("code",drop=False) for d,g in panel.groupby("signal_date")}
    state_by=states.set_index("signal_date").to_dict("index")
    cash=INITIAL_CASH;pos={};eq=[];trades=[];timing=[]
    for d in sorted(by):
        g=by[d]; st=state_by.get(d,{"regime":"neutral"})
        qs=quant.score(d,g.reset_index(drop=True))
        sel=target_for(g.reset_index(drop=True),st,agent,qs)
        tgt=set(sel.code)
        # mark positions at this executable next-open price where available.
        for c,p in pos.items():
            if c in g.index and np.isfinite(g.loc[c].exec_open):p.last_price=float(g.loc[c].exec_open)
        nav_open=cash+sum(p.units*p.last_price for p in pos.values())
        # sell names no longer targeted; suspension/one-price lock means no fill.
        for c in list(pos):
            if c in tgt:continue
            r=g.loc[c] if c in g.index else None
            if r is None or locked(r,"sell"):continue
            px=float(r.exec_open)*(1-SLIPPAGE); gross=pos[c].units*px; cost=fee(gross,"sell",r.trade_date)
            p=pos.pop(c);cash+=gross-cost
            trades.append({"strategy":agent["strategy"],"code":c,"entry_date":p.entry_date,"exit_date":r.trade_date,"net_pnl":gross-cost-p.entry_cost,"net_return":((gross-cost)/p.entry_cost-1)})
            timing.append({"strategy":agent["strategy"],"signal_date":d,"trade_date":r.trade_date,"side":"sell"})
        invest=exposure(agent,st["regime"]); desired=max(1,len(tgt)); per=nav_open*invest/desired
        for c in sel.code:
            if c in pos or c not in g.index:continue
            r=g.loc[c]
            if locked(r,"buy"):continue
            factor=float(r.exec_factor) if np.isfinite(r.exec_factor) and r.exec_factor>0 else 1.0
            adjpx=float(r.exec_open)*(1+SLIPPAGE);rawpx=adjpx/factor
            if rawpx<=0:continue
            # RQAlpha-inspired liquidity cap: <=5% of daily volume and board lot 100 shares.
            max_raw_by_volume=max(0,int(abs(float(r.exec_volume))*factor*VOLUME_PARTICIPATION//100)*100)
            raw_shares=int(min(per,cash*.98)//(rawpx*100))*100
            if max_raw_by_volume>0:raw_shares=min(raw_shares,max_raw_by_volume)
            if raw_shares<=0:continue
            units=raw_shares/factor;gross=units*adjpx;cost=fee(gross,"buy",r.trade_date);total=gross+cost
            if total>cash:continue
            cash-=total;pos[c]=Pos(units,total,pd.Timestamp(r.trade_date),float(r.exec_open))
            timing.append({"strategy":agent["strategy"],"signal_date":d,"trade_date":r.trade_date,"side":"buy"})
        nav=cash+sum(p.units*p.last_price for p in pos.values())
        eq.append({"strategy":agent["strategy"],"signal_date":d,"trade_date":g.trade_date.iloc[0],"equity":nav,"cash":cash,"positions":len(pos),"regime":st["regime"],"era":era_name(d)})
    e=pd.DataFrame(eq);t=pd.DataFrame(trades);tm=pd.DataFrame(timing)
    if len(tm):
        tm["signal_date"]=pd.to_datetime(tm.signal_date);tm["trade_date"]=pd.to_datetime(tm.trade_date)
        bad=tm[tm.signal_date>=tm.trade_date]
        if len(bad):raise RuntimeError(f"trade timing violations {len(bad)}")
    return e,t,tm

def stats(eq,tr):
    e=eq.set_index("trade_date").equity.astype(float);r=e.pct_change().dropna()
    total=float(e.iloc[-1]/e.iloc[0]-1); years=max((e.index[-1]-e.index[0]).days/365.25,1e-9)
    cagr=float((e.iloc[-1]/e.iloc[0])**(1/years)-1);dd=e/e.cummax()-1
    sharpe=float(r.mean()/r.std()*np.sqrt(52)) if r.std()>0 else np.nan
    return {"final_asset":float(e.iloc[-1]),"total_return":total,"cagr":cagr,"max_drawdown":float(dd.min()),"sharpe":sharpe,"trades":int(len(tr)),"win_rate":float((tr.net_pnl>0).mean()) if len(tr) else np.nan,"profit_factor":float(tr.loc[tr.net_pnl>0,"net_pnl"].sum()/abs(tr.loc[tr.net_pnl<0,"net_pnl"].sum())) if len(tr) and tr.loc[tr.net_pnl<0,"net_pnl"].sum()!=0 else np.nan}

def main():
    cal,m,ua=load_base();panel,sigs=build_weekly_panel(cal,m);states=benchmark_state(cal,panel)
    quant=EraQuant(panel)
    sums=[];alleq=[];alltr=[];alltm=[]
    for a in AGENTS:
        print("running",a["strategy"],flush=True)
        eq,tr,tm=simulate(panel,states,a,quant);st=stats(eq,tr);st.update({"strategy":a["strategy"],"risk":a["risk"]});sums.append(st);alleq.append(eq);alltr.append(tr);alltm.append(tm)
    sm=pd.DataFrame(sums).sort_values("total_return",ascending=False)
    eq=pd.concat(alleq,ignore_index=True);tr=pd.concat(alltr,ignore_index=True) if alltr else pd.DataFrame();tm=pd.concat(alltm,ignore_index=True) if alltm else pd.DataFrame()
    # Benchmark for the exact executable period.
    b=qb.read_bin("SH000985","close",cal).loc[START:END].dropna();br=float(b.iloc[-1]/b.iloc[0]-1)
    sm["benchmark_return"]=br;sm["excess_return"]=sm.total_return-br
    annual=[]
    for s,g in eq.groupby("strategy"):
        for y,z in g.groupby(pd.to_datetime(g.trade_date).dt.year):
            annual.append({"strategy":s,"year":int(y),"return":float(z.equity.iloc[-1]/z.equity.iloc[0]-1),"era":era_name(pd.Timestamp(f"{int(y)}-12-31"))})
    # robustness: remove top 10 completed trades from PnL, not a recomputed path.
    rob=[]
    for s,z in tr.groupby("strategy") if len(tr) else []:
        pnl=float(z.net_pnl.sum());best=float(z.nlargest(min(10,len(z)),"net_pnl").net_pnl.sum());rob.append({"strategy":s,"completed_trade_pnl":pnl,"best10_pnl":best,"pnl_without_best10":pnl-best})
    audit={**ua,"weekly_panel_rows":len(panel),"signal_dates":int(panel.signal_date.nunique()),"trade_timing_violations":0,"model_timing_violations":int(sum(x["timing_violation"] for x in quant.audit)),"min_trade_lag_days":float((pd.to_datetime(tm.trade_date)-pd.to_datetime(tm.signal_date)).dt.total_seconds().min()/86400) if len(tm) else np.nan,"model_rule":"2016-18 handcrafted; 2019-20 rolling Ridge; 2021-22 rolling LightGBM; 2023-26 rolling Ridge+LightGBM ensemble; all labels fully matured before training","execution_rule":"weekly signal at close, first later exchange open; price-limit/inactive fail-closed; 5% volume participation; historical stamp/transfer fees; 10bp slippage"}
    if audit["model_timing_violations"]!=0:raise RuntimeError("model future leakage")
    pd.DataFrame([audit]).to_csv(OUT/"audit.csv",index=False);sm.to_csv(OUT/"summary.csv",index=False);eq.to_csv(OUT/"equity.csv",index=False);tr.to_csv(OUT/"trades.csv",index=False);tm.to_csv(OUT/"timing_events.csv",index=False);pd.DataFrame(quant.audit).to_csv(OUT/"model_training_audit.csv",index=False);pd.DataFrame(annual).to_csv(OUT/"annual_returns.csv",index=False);pd.DataFrame(rob).to_csv(OUT/"robustness.csv",index=False)
    pd.DataFrame(AGENTS).to_csv(OUT/"agent_definitions.csv",index=False)
    print("=== AUDIT ===");print(pd.DataFrame([audit]).to_string(index=False),flush=True)
    print("=== LEADERBOARD ===");print(sm.to_string(index=False),flush=True)
    print("=== ANNUAL ===");print(pd.DataFrame(annual).pivot(index="year",columns="strategy",values="return").to_string(),flush=True)
    print("=== ROBUSTNESS ===");print(pd.DataFrame(rob).to_string(index=False),flush=True)

if __name__=="__main__":main()
