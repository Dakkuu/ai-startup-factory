from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import run_backtest_qlib as qb

# Frozen market-data release. News test window is dictated only by public-news coverage,
# not by return outcomes.
RELEASE_TAG = "2026-07-29"
START = pd.Timestamp("2021-01-04")
END = pd.Timestamp("2021-11-15")  # includes next open after final 2021-11-12 news
WARM = START - pd.Timedelta(days=150)
INITIAL_CASH = 1_000_000.0
COMMISSION = 0.00025
MIN_COMMISSION = 5.0
STAMP_DUTY_SELL = 0.001       # 2021 rate: 0.1%, sell side
TRANSFER_FEE = 0.00002         # pre-2022-04-29 A-share transfer fee, both sides
SLIPPAGE = 0.0008              # 8bp each side; deliberately above frictionless fill
MIN_LIQ = 20_000_000.0
OUT = Path("results_behavioral_news")
OUT.mkdir(exist_ok=True)

# Never load fields such as LABEL / CHANGE / FIRST_DAY / future open-close columns / READ.
NEWS_URL = "https://raw.githubusercontent.com/JinanZou/Astock/main/data/df_all_year_srl.csv"
NEWS_FILE = Path("astock_news_behavioral.tsv")
NEWS_USECOLS = ["CODE", "NAME", "CREATED_DATE", "text_a", "DESCRIPTION"]

POS_STRONG = ["预增", "扭亏", "中标", "获批", "回购", "增持", "涨价", "签订大单", "重大合同", "超预期", "创新高"]
POS_WEAK = ["增长", "同比增长", "订单", "签署", "合作", "投产", "扩产", "分红", "研发", "突破", "核准", "注册证", "收购", "投资"]
NEG_STRONG = ["立案", "退市", "处罚", "预亏", "巨亏", "减持", "终止", "违约", "停产", "风险提示", "问询函", "关注函", "诉讼", "违规", "爆雷", "下修"]
NEG_WEAK = ["下降", "下滑", "亏损", "减值", "延期", "被查", "质押", "冻结", "解禁", "监管"]
THEMES = ["新能源", "锂电", "锂电池", "光伏", "半导体", "芯片", "军工", "白酒", "医药", "医疗", "稀土", "煤炭", "钢铁", "化工", "有色", "汽车", "新能源汽车", "储能", "风电", "5G", "数字", "软件", "云计算", "人工智能", "机器人", "元宇宙", "消费", "券商", "证券", "房地产", "农业", "猪肉", "电力", "碳中和", "环保"]

AGENTS = [
    {"strategy":"01_defensive_institution", "risk":"low", "max_names":8, "invest":0.65, "hold":10, "stop":-0.08},
    {"strategy":"02_quality_event", "risk":"low_mid", "max_names":6, "invest":0.85, "hold":6, "stop":-0.10},
    {"strategy":"03_smart_money_trend", "risk":"mid", "max_names":8, "invest":0.90, "hold":8, "stop":-0.12},
    {"strategy":"04_quant_crowding", "risk":"mid", "max_names":10, "invest":0.95, "hold":5, "stop":-0.10},
    {"strategy":"05_retail_attention_chase", "risk":"mid_high", "max_names":5, "invest":0.98, "hold":3, "stop":-0.12},
    {"strategy":"06_hot_money_theme_relay", "risk":"high", "max_names":3, "invest":0.95, "hold":2, "stop":-0.15},
    {"strategy":"07_limit_up_relay", "risk":"very_high", "max_names":3, "invest":0.90, "hold":2, "stop":-0.15},
    {"strategy":"08_panic_reversal", "risk":"high_contrarian", "max_names":5, "invest":0.75, "hold":3, "stop":-0.10},
    {"strategy":"09_bad_news_defense", "risk":"low_adaptive", "max_names":8, "invest":0.70, "hold":7, "stop":-0.08},
    {"strategy":"10_adaptive_barbell", "risk":"adaptive", "max_names":6, "invest":0.80, "hold":5, "stop":-0.12},
]

AGENT_DESCRIPTIONS = {
"01_defensive_institution":"高流动性+低波动+非负新闻，弱市主动留现金；模拟保守机构。",
"02_quality_event":"只做明确正面公司事件，避免消息后已大涨的股票；模拟事件驱动价值资金。",
"03_smart_money_trend":"高流动性、平滑中期趋势、温和放量、新闻不差；模拟大资金顺势。",
"04_quant_crowding":"动量+低波+流动性横截面排名，新闻仅作过滤；模拟量化拥挤。",
"05_retail_attention_chase":"多新闻/新题材+放量+接近新高+正面文本；模拟散户注意力追涨。",
"06_hot_money_theme_relay":"同日多股同题材共振+强放量+短线涨幅；模拟游资题材接力。",
"07_limit_up_relay":"前一日接近涨停+题材/新闻催化+热市场；模拟打板接力。",
"08_panic_reversal":"坏消息+放量大跌后博弈反弹；模拟高风险逆向资金。",
"09_bad_news_defense":"市场坏消息密集时只持正面新闻、低波高流动性标的并降仓；模拟防守资金。",
"10_adaptive_barbell":"热市切题材接力，冷市切机构防守，中性做质量事件；模拟自适应操盘手。",
}


def fee(gross: float, side: str) -> float:
    return max(MIN_COMMISSION, gross * COMMISSION) + gross * TRANSFER_FEE + (gross * STAMP_DUTY_SELL if side == "sell" else 0.0)


def conservative_locked(row, prev_close, side):
    if row is None:
        return True
    vals = [row.get(c, np.nan) for c in ["open","high","low","close"]]
    if not all(np.isfinite(v) and v > 0 for v in vals):
        return True
    one = abs(float(row.high)-float(row.low)) < 1e-8 and abs(float(row.open)-float(row.high)) < 1e-8
    if not one:
        return False
    if not np.isfinite(prev_close) or prev_close <= 0:
        return True
    pct = float(row.open)/float(prev_close)-1
    # Fail closed for ST and non-ST one-price locks.
    return (side == "buy" and pct >= 0.045) or (side == "sell" and pct <= -0.045)


def read_calendar():
    qb.RELEASE_TAG = RELEASE_TAG
    qb.ROOT = Path("qlib_data")
    qb.download_and_extract()
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(qb.ROOT/"calendars"/"day.txt", header=None)[0]))
    if cal.max() < END:
        raise RuntimeError("Frozen Qlib release does not cover requested test end")
    return cal


def load_all_membership(cal):
    candidates = [qb.ROOT/"instruments"/"all.txt", qb.ROOT/"instruments"/"csi500.txt"]
    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        raise RuntimeError("No dynamic instrument file found")
    df = pd.read_csv(p, sep="\t", header=None, names=["code","start","end"], usecols=[0,1,2])
    df["code"] = df.code.astype(str).str.upper()
    df["start"] = pd.to_datetime(df.start)
    df["end"] = pd.to_datetime(df.end)
    df = df[(df.end >= WARM) & (df.start <= END)].copy()
    dates = cal[(cal>=START)&(cal<=END)]
    counts = [int(((df.start<=d)&(df.end>=d)).sum()) for d in dates]
    union = df[(df.end>=START)&(df.start<=END)].code.nunique()
    exited = df[(df.end>=START)&(df.end<END)].code.nunique()
    entered = df[(df.start>START)&(df.start<=END)].code.nunique()
    if p.name == "all.txt":
        if union < 3000 or min(counts) < 2800:
            raise RuntimeError(f"FAIL-CLOSED all-A universe suspicious: union={union}, daily={min(counts)}..{max(counts)}")
        if exited < 5:
            raise RuntimeError(f"FAIL-CLOSED too few historical exits ({exited}); survivor-bias risk")
    audit = {"instrument_file":p.name,"union_members":union,"entered":entered,"exited":exited,"min_daily_members":min(counts),"max_daily_members":max(counts)}
    return df, audit


def active_codes(mdf, d):
    return set(mdf.loc[(mdf.start<=d)&(mdf.end>=d),"code"])


def build_panel(cal, mdf):
    codes = sorted(set(mdf.loc[(mdf.end>=WARM)&(mdf.start<=END),"code"]))
    frames=[]; missing=0
    for i,code in enumerate(codes,1):
        cols={}
        for fld in ["open","high","low","close","volume","factor"]:
            s=qb.read_bin(code,fld,cal)
            if not s.empty: cols[fld]=s
        if not all(f in cols for f in ["open","high","low","close","volume"]):
            missing += 1; continue
        d=pd.concat(cols,axis=1)
        d=d[(d.index>=WARM)&(d.index<=END)].copy()
        if d.empty: continue
        d["date"]=d.index; d["code"]=code
        if "factor" not in d: d["factor"]=1.0
        d["factor"]=d.factor.replace(0,np.nan)
        d["liquidity"]=(d.close.abs()*d.volume.abs()).replace([np.inf,-np.inf],np.nan)
        frames.append(d.reset_index(drop=True))
        if i%500==0: print("loaded prices",i,"/",len(codes),flush=True)
    if len(frames) < 2500:
        raise RuntimeError(f"FAIL-CLOSED insufficient histories: {len(frames)}, missing={missing}")
    p=pd.concat(frames,ignore_index=True).sort_values(["code","date"])
    g=p.groupby("code",group_keys=False)
    p["ret1"]=g.close.pct_change()
    p["mom3"]=g.close.pct_change(3)
    p["mom5"]=g.close.pct_change(5)
    p["mom20"]=g.close.pct_change(20)
    p["mom60"]=g.close.pct_change(60)
    p["ma20"]=g.close.transform(lambda s:s.rolling(20).mean())
    p["ma60"]=g.close.transform(lambda s:s.rolling(60).mean())
    p["vol20"]=g.ret1.transform(lambda s:s.rolling(20).std())
    p["vol_ma20"]=g.volume.transform(lambda s:s.rolling(20).mean())
    p["liq_ma20"]=g.liquidity.transform(lambda s:s.rolling(20).mean())
    p["prev20_high"]=g.high.transform(lambda s:s.rolling(20).max())
    p["vol_ratio"]=p.volume/p.vol_ma20
    p["dist_high20"]=p.close/p.prev20_high-1
    return p


def text_score(t):
    t=str(t)
    ps=2*sum(w in t for w in POS_STRONG)+sum(w in t for w in POS_WEAK)
    ng=2*sum(w in t for w in NEG_STRONG)+sum(w in t for w in NEG_WEAK)
    return ps-ng, ps, ng


def map_news(cal, mdf):
    if not NEWS_FILE.exists():
        print("downloading Astock raw news",flush=True)
        urllib.request.urlretrieve(NEWS_URL,NEWS_FILE)
    n=pd.read_csv(NEWS_FILE,sep="\t",usecols=NEWS_USECOLS,low_memory=False)
    n["CREATED_DATE"]=pd.to_datetime(n.CREATED_DATE,errors="coerce")
    n=n.dropna(subset=["CREATED_DATE","CODE"]).copy()
    n["digits"]=pd.to_numeric(n.CODE,errors="coerce").astype("Int64").astype(str).str.zfill(6)
    # Map six-digit ticker to Qlib instrument using the dynamic historical file.
    allcodes=sorted(set(mdf.code))
    suffix=defaultdict(list)
    for c in allcodes:
        digits="".join(ch for ch in c if ch.isdigit())[-6:]
        if len(digits)==6: suffix[digits].append(c)
    def resolve(x):
        z=suffix.get(x,[])
        if len(z)==1:return z[0]
        # Prefer standard A-share exchange prefixes if duplicate.
        z2=[c for c in z if c.startswith(("SH","SZ","BJ"))]
        return z2[0] if len(z2)==1 else (z[0] if z else None)
    n["code"]=n.digits.map(resolve)
    matched=float(n.code.notna().mean())
    n=n.dropna(subset=["code"]).copy()
    # Strict execution mapping: news on calendar date t may only trade on first exchange day AFTER t.
    trade_days=cal[(cal>=START)&(cal<=END)]
    arr=trade_days.values.astype("datetime64[ns]")
    dayvals=n.CREATED_DATE.dt.normalize().values.astype("datetime64[ns]")
    idx=np.searchsorted(arr,dayvals,side="right")
    ok=idx < len(arr)
    n=n.loc[ok].copy(); idx=idx[ok]
    n["exec_date"]=pd.to_datetime(arr[idx])
    n=n[(n.exec_date>=START)&(n.exec_date<=END)].copy()
    n["text"]=(n.text_a.fillna("").astype(str)+" "+n.DESCRIPTION.fillna("").astype(str))
    scores=n.text.map(text_score)
    n[["sentiment","pos_words","neg_words"]]=pd.DataFrame(scores.tolist(),index=n.index)
    n["announcement"]=n.text.str.contains("公告",regex=False).astype(int)
    n["theme_list"]=n.text.map(lambda t:[x for x in THEMES if x in t])
    # Daily theme diffusion = number of distinct stocks sharing a theme, known before next open.
    ex=n[["exec_date","code","theme_list"]].explode("theme_list").dropna(subset=["theme_list"])
    theme_counts=ex.groupby(["exec_date","theme_list"]).code.nunique().rename("theme_count")
    def row_heat(r):
        if not r.theme_list:return 0
        return max(int(theme_counts.get((r.exec_date,t),0)) for t in r.theme_list)
    n["theme_heat"]=n.apply(row_heat,axis=1)
    agg=n.groupby(["exec_date","code"]).agg(
        news_count=("text","size"), sentiment=("sentiment","sum"), pos_words=("pos_words","sum"), neg_words=("neg_words","sum"),
        announcement=("announcement","max"), theme_heat=("theme_heat","max"), max_news_time=("CREATED_DATE","max"),
        title_sample=("text_a","first"), name=("NAME","first")
    ).reset_index()
    agg=agg.sort_values(["code","exec_date"])
    agg["days_since_news"]=agg.groupby("code").exec_date.diff().dt.days.fillna(99).clip(upper=99)
    agg["novelty"]=(agg.days_since_news.clip(upper=20)/20.0)+np.log1p(agg.news_count)
    if len(agg)<5000:
        raise RuntimeError(f"FAIL-CLOSED too few mapped news events: {len(agg)}")
    audit={"raw_news_rows":len(pd.read_csv(NEWS_FILE,sep="\t",usecols=["CODE"])),"mapped_row_rate":matched,"mapped_stock_day_events":len(agg),"mapped_codes":agg.code.nunique(),"news_start":n.CREATED_DATE.min(),"news_end":n.CREATED_DATE.max()}
    return agg,n,audit


def pct_rank(s,ascending=True):
    return s.rank(pct=True,ascending=ascending,method="average")


def market_state(prev, news_today):
    x=prev.dropna(subset=["ret1","ma20","ma60","close"])
    breadth20=float((x.close>x.ma20).mean()) if len(x) else 0.5
    breadth60=float((x.close>x.ma60).mean()) if len(x) else 0.5
    mean_ret=float(x.ret1.mean()) if len(x) else 0.0
    limit_up=float((x.ret1>=0.095).mean()) if len(x) else 0.0
    limit_dn=float((x.ret1<=-0.095).mean()) if len(x) else 0.0
    if len(news_today):
        pos=float((news_today.sentiment>0).mean()); neg=float((news_today.sentiment<0).mean()); theme=int(news_today.theme_heat.max())
    else: pos=neg=0.0; theme=0
    hot=(breadth20>=0.55 and (limit_up>=0.006 or theme>=8) and mean_ret>-0.005)
    cold=(breadth20<0.40 or breadth60<0.38 or mean_ret<-0.008 or limit_dn>0.01 or neg>0.50)
    regime="hot" if hot else ("cold" if cold else "neutral")
    return {"breadth20":breadth20,"breadth60":breadth60,"mean_ret1":mean_ret,"limit_up_share":limit_up,"limit_down_share":limit_dn,"news_pos_ratio":pos,"news_neg_ratio":neg,"theme_max":theme,"regime":regime}


def prepare_candidates(prev, news_today):
    if news_today.empty:return pd.DataFrame()
    c=news_today.merge(prev,on="code",how="inner",suffixes=("_news",""))
    needed=["ret1","mom5","mom20","mom60","vol20","vol_ratio","liq_ma20","dist_high20"]
    c=c.dropna(subset=needed)
    c=c[(c.close>0)&(c.open>0)&(c.liq_ma20>=MIN_LIQ)].copy()
    if c.empty:return c
    c["r_mom20"]=pct_rank(c.mom20); c["r_mom60"]=pct_rank(c.mom60)
    c["r_liq"]=pct_rank(c.liq_ma20); c["r_lowvol"]=pct_rank(-c.vol20)
    c["r_volratio"]=pct_rank(c.vol_ratio); c["r_news"]=pct_rank(c.news_count)
    c["r_novelty"]=pct_rank(c.novelty); c["r_theme"]=pct_rank(c.theme_heat)
    c["r_sent"]=pct_rank(c.sentiment); c["r_drop"]=pct_rank(-c.ret1)
    return c


def choose(c,state,agent):
    if c.empty:return c
    s=agent["strategy"]; x=c.copy()
    if s=="01_defensive_institution":
        if state["regime"]=="cold": x=x.iloc[0:0]
        else:
            x=x[(x.sentiment>=0)&(x.mom20>-0.05)&(x.ret1<0.07)&(x.vol_ratio<2.5)]
            x["score"]=.30*x.r_liq+.25*x.r_lowvol+.20*x.r_mom60+.15*x.r_sent+.10*x.r_novelty
    elif s=="02_quality_event":
        x=x[(x.announcement==1)&((x.sentiment>=2)|(x.pos_words>=2))&(x.ret1<0.07)&(x.mom20>-0.10)]
        x["score"]=.30*x.r_sent+.20*x.r_novelty+.20*x.r_liq+.15*x.r_lowvol+.15*x.r_mom20
    elif s=="03_smart_money_trend":
        x=x[(x.sentiment>=0)&(x.mom20>0)&(x.mom60>0)&(x.vol_ratio.between(.7,2.5))]
        x["score"]=.30*x.r_liq+.25*x.r_mom60+.20*x.r_mom20+.15*x.r_lowvol+.10*x.r_sent
    elif s=="04_quant_crowding":
        x=x[(x.sentiment>=0)&(x.mom20>-0.02)]
        x["score"]=.28*x.r_mom20+.22*x.r_mom60+.22*x.r_lowvol+.18*x.r_liq+.10*x.r_news
    elif s=="05_retail_attention_chase":
        x=x[(x.sentiment>0)&((x.news_count>=2)|(x.theme_heat>=3))&(x.ret1>0.005)&(x.vol_ratio>1.25)&(x.dist_high20>-0.05)]
        x["score"]=.25*x.r_news+.20*x.r_novelty+.20*x.r_volratio+.15*x.r_theme+.10*x.r_sent+.10*x.r_mom20
    elif s=="06_hot_money_theme_relay":
        x=x[(x.theme_heat>=3)&(x.sentiment>=0)&(x.ret1.between(.02,.095))&(x.vol_ratio>1.4)&(x.mom5>0)]
        x["score"]=.35*x.r_theme+.25*x.r_volratio+.15*x.r_news+.15*x.r_mom20+.10*x.r_sent
    elif s=="07_limit_up_relay":
        if state["regime"]!="hot": x=x.iloc[0:0]
        else:
            x=x[(x.ret1>=.09)&(x.sentiment>=0)&((x.theme_heat>=2)|(x.news_count>=2))]
            x["score"]=.30*x.r_theme+.25*x.r_news+.20*x.r_volratio+.15*x.r_mom20+.10*x.r_sent
    elif s=="08_panic_reversal":
        x=x[(x.sentiment<=-1)&((x.ret1<=-.03)|(x.mom5<=-.08))&(x.vol_ratio>1.15)]
        x["score"]=.35*x.r_drop+.20*x.r_news+.20*x.r_liq+.15*x.r_volratio+.10*x.r_novelty
    elif s=="09_bad_news_defense":
        x=x[(x.sentiment>=1)&(x.ret1<.06)&(x.mom20>-.08)]
        x["score"]=.30*x.r_lowvol+.25*x.r_liq+.20*x.r_sent+.15*x.r_mom20+.10*x.r_novelty
    elif s=="10_adaptive_barbell":
        if state["regime"]=="hot":
            x=x[(x.theme_heat>=2)&(x.sentiment>=0)&(x.ret1>0)&(x.vol_ratio>1.2)]
            x["score"]=.30*x.r_theme+.25*x.r_volratio+.20*x.r_news+.15*x.r_mom20+.10*x.r_sent
        elif state["regime"]=="cold":
            x=x[(x.sentiment>=0)&(x.ret1<.05)&(x.mom20>-.08)]
            x["score"]=.35*x.r_lowvol+.30*x.r_liq+.15*x.r_sent+.10*x.r_mom20+.10*x.r_novelty
        else:
            x=x[(x.sentiment>=1)&(x.ret1<.07)]
            x["score"]=.25*x.r_sent+.20*x.r_liq+.20*x.r_lowvol+.20*x.r_mom20+.15*x.r_novelty
    else: raise ValueError(s)
    return x.sort_values("score",ascending=False).head(agent["max_names"]) if len(x) else x

@dataclass
class Pos:
    units:float; entry_cost:float; entry_date:pd.Timestamp; entry_raw:float; signal_time:pd.Timestamp


def simulate(panel,mdf,news_agg,agent):
    dates=sorted(pd.Timestamp(d) for d in panel.date.unique() if START<=pd.Timestamp(d)<=END)
    by={pd.Timestamp(d):z.set_index("code",drop=False) for d,z in panel.groupby("date")}
    news_by={pd.Timestamp(d):z.copy() for d,z in news_agg.groupby("exec_date")}
    cash=INITIAL_CASH; pos={}; trades=[]; eq=[]; timing=[]; states=[]
    for k,d in enumerate(dates):
        day=by.get(d)
        if day is None or k==0: continue
        prev_d=dates[k-1]; prev=by.get(prev_d)
        if prev is None: continue
        mem=active_codes(mdf,prev_d)
        prev_active=prev[prev.code.isin(mem)].copy()
        nt=news_by.get(d,pd.DataFrame(columns=news_agg.columns))
        # Keep only news mapped to stocks that were historically active at the signal close.
        if len(nt): nt=nt[nt.code.isin(mem)].copy()
        state=market_state(prev_active,nt); state.update(date=d,strategy=agent["strategy"]); states.append(state)
        c=prepare_candidates(prev_active,nt)
        picks=choose(c,state,agent)
        pickcodes=set(picks.code) if len(picks) else set()
        newsmap=nt.set_index("code",drop=False) if len(nt) else pd.DataFrame()

        # News is from a prior calendar date; price signals are previous trading-day close.
        trade_open_ts=d+pd.Timedelta(hours=9,minutes=30)
        price_info_ts=prev_d+pd.Timedelta(hours=15)
        if not price_info_ts < trade_open_ts: raise RuntimeError("LOOKAHEAD price timing violation")

        # Exits first: stop, max holding age, or severe new negative news.
        for code in list(pos):
            p=pos[code]
            row=day.loc[code] if code in day.index else None
            prow=prev.loc[code] if code in prev.index else None
            if row is None or prow is None: continue
            factor_prev=float(prow.factor) if np.isfinite(prow.factor) and prow.factor>0 else 1.0
            prev_raw=float(prow.close)/factor_prev
            ret_since=prev_raw/p.entry_raw-1 if p.entry_raw>0 else 0
            age=sum(1 for x in dates if p.entry_date < x <= d)
            severe=False; signal_time=price_info_ts; reason="hold_expiry"
            if len(newsmap) and code in newsmap.index:
                nr=newsmap.loc[code]
                if isinstance(nr,pd.DataFrame): nr=nr.iloc[-1]
                if nr.sentiment<=-2:
                    severe=True; signal_time=pd.Timestamp(nr.max_news_time); reason="negative_news"
            if ret_since<=agent["stop"]: reason="stop_loss"; severe=True; signal_time=price_info_ts
            should_sell=severe or age>=agent["hold"]
            # defensive/adaptive agents can de-risk in cold regimes after minimum 2 days
            if agent["strategy"] in {"01_defensive_institution","09_bad_news_defense","10_adaptive_barbell"} and state["regime"]=="cold" and age>=2:
                should_sell=True; reason="risk_off"; signal_time=max(signal_time,price_info_ts)
            if not should_sell: continue
            prev_close=float(prow.close)
            if conservative_locked(row,prev_close,"sell"): continue
            if not pd.Timestamp(signal_time) < trade_open_ts: raise RuntimeError("LOOKAHEAD sell timing violation")
            adj_px=float(row.open)*(1-SLIPPAGE); gross=p.units*adj_px; cost=fee(gross,"sell"); cash+=gross-cost
            q=pos.pop(code); pnl=gross-cost-q.entry_cost
            fac=float(row.factor) if np.isfinite(row.factor) and row.factor>0 else 1.0
            trades.append({"strategy":agent["strategy"],"code":code,"entry_date":q.entry_date,"exit_date":d,"net_pnl":pnl,"net_return":pnl/q.entry_cost,"exit_reason":reason})
            timing.append({"strategy":agent["strategy"],"side":"sell","signal_time":signal_time,"trade_time":trade_open_ts})

        # Dynamic exposure: defensive agents hold more cash in weak regimes.
        invest=agent["invest"]
        if agent["strategy"]=="09_bad_news_defense" and state["regime"]=="cold": invest=min(invest,.45)
        if agent["strategy"]=="10_adaptive_barbell": invest=.95 if state["regime"]=="hot" else (.45 if state["regime"]=="cold" else .80)
        # New buys from today's point-in-time news candidates.
        nav_open=cash
        for code,p in pos.items():
            if code in day.index and np.isfinite(day.loc[code].open): nav_open+=p.units*float(day.loc[code].open)
        slots=max(0,agent["max_names"]-len(pos))
        if slots and len(picks):
            per_target=nav_open*invest/agent["max_names"]
            for _,pick in picks.iterrows():
                code=pick.code
                if code in pos or code not in day.index or slots<=0: continue
                row=day.loc[code]; prow=prev.loc[code] if code in prev.index else None
                if prow is None: continue
                if conservative_locked(row,float(prow.close),"buy"): continue
                signal_time=max(pd.Timestamp(pick.max_news_time),price_info_ts)
                if not signal_time < trade_open_ts: raise RuntimeError(f"LOOKAHEAD buy {code}: {signal_time} vs {trade_open_ts}")
                fac=float(row.factor) if np.isfinite(row.factor) and row.factor>0 else 1.0
                adj_px=float(row.open)*(1+SLIPPAGE); raw_px=adj_px/fac
                if raw_px<=0 or not np.isfinite(raw_px): continue
                raw_shares=int(min(per_target,cash*.98)//(raw_px*100))*100
                if raw_shares<=0: continue
                units=raw_shares/fac; gross=units*adj_px; total=gross+fee(gross,"buy")
                if total>cash: continue
                cash-=total; pos[code]=Pos(units,total,d,raw_px,signal_time); slots-=1
                timing.append({"strategy":agent["strategy"],"side":"buy","signal_time":signal_time,"trade_time":trade_open_ts})

        nav=cash
        for code,p in pos.items():
            if code in day.index and np.isfinite(day.loc[code].close): nav+=p.units*float(day.loc[code].close)
        eq.append({"date":d,"strategy":agent["strategy"],"equity":nav,"cash":cash,"n_positions":len(pos)})
    return pd.DataFrame(eq),pd.DataFrame(trades),pd.DataFrame(timing),pd.DataFrame(states)


def stats(eq,tr):
    e=eq.set_index("date").equity.astype(float); r=e.pct_change().dropna()
    total=e.iloc[-1]/INITIAL_CASH-1; years=max((e.index[-1]-e.index[0]).days/365.25,1/252); cagr=(e.iloc[-1]/INITIAL_CASH)**(1/years)-1
    dd=e/e.cummax()-1; sd=r.std(ddof=0); dn=r[r<0].std(ddof=0)
    out={"final_asset":e.iloc[-1],"total_return":total,"cagr":cagr,"max_drawdown":dd.min(),"sharpe":np.sqrt(252)*r.mean()/sd if sd>0 else np.nan,"sortino":np.sqrt(252)*r.mean()/dn if np.isfinite(dn) and dn>0 else np.nan,"trades":len(tr)}
    if len(tr):
        w=tr[tr.net_pnl>0]; l=tr[tr.net_pnl<0]; out.update(win_rate=len(w)/len(tr),avg_win=w.net_return.mean() if len(w) else np.nan,avg_loss=l.net_return.mean() if len(l) else np.nan,profit_factor=w.net_pnl.sum()/abs(l.net_pnl.sum()) if len(l) and l.net_pnl.sum()!=0 else np.nan)
    else: out.update(win_rate=np.nan,avg_win=np.nan,avg_loss=np.nan,profit_factor=np.nan)
    return out


def benchmark(cal):
    for code in ["sh000985","SH000985","sh000905","SH000905","sh000300","SH000300"]:
        s=qb.read_bin(code,"close",cal)
        if not s.empty:
            s=s[(s.index>=START)&(s.index<=END)].dropna()
            if len(s)>5: return code,s/s.iloc[0]
    return "none",pd.Series(dtype=float)


def main():
    cal=read_calendar(); mdf,ua=load_all_membership(cal); panel=build_panel(cal,mdf); news_agg,news_raw,na=map_news(cal,mdf)
    # Explicitly prove forbidden future-labelled fields were not loaded.
    forbidden_loaded=set(NEWS_USECOLS)&{"READ","CHANGE","FIRST_DAY","SECOND_DAY","open1","close1","day1","label","co_label","cc_label"}
    if forbidden_loaded: raise RuntimeError(f"Forbidden future-contaminated columns loaded: {forbidden_loaded}")
    bench_code,bench=benchmark(cal); bench_ret=float(bench.iloc[-1]-1) if len(bench) else np.nan
    summaries=[]; eqs=[]; trs=[]; tes=[]; sts=[]
    for a in AGENTS:
        print("running",a["strategy"],flush=True)
        eq,tr,te,st=simulate(panel,mdf,news_agg,a)
        if eq.empty: raise RuntimeError(f"No equity for {a['strategy']}")
        m=stats(eq,tr); m.update(strategy=a["strategy"],risk=a["risk"],benchmark=bench_code,benchmark_return=bench_ret,excess_return=m["total_return"]-bench_ret if np.isfinite(bench_ret) else np.nan)
        summaries.append(m); eqs.append(eq); trs.append(tr); tes.append(te); sts.append(st)
    sm=pd.DataFrame(summaries).sort_values("total_return",ascending=False)
    eq=pd.concat(eqs,ignore_index=True); tr=pd.concat(trs,ignore_index=True) if trs else pd.DataFrame(); te=pd.concat(tes,ignore_index=True) if tes else pd.DataFrame(); st=pd.concat(sts,ignore_index=True)
    if len(te):
        te["signal_time"]=pd.to_datetime(te.signal_time); te["trade_time"]=pd.to_datetime(te.trade_time); bad=te[te.signal_time>=te.trade_time]
    else: bad=pd.DataFrame()
    if len(bad): raise RuntimeError(f"FAIL-CLOSED lookahead events={len(bad)}")
    minlag=(te.trade_time-te.signal_time).dt.total_seconds().min()/3600 if len(te) else np.nan
    audit={**ua,**na,"release_tag":RELEASE_TAG,"test_start":START,"test_end":END,"timing_events":len(te),"timing_violations":len(bad),"min_information_lag_hours":minlag,"forbidden_future_columns_loaded":len(forbidden_loaded),"READ_used":0,"signal_rule":"news strictly prior calendar date + previous trading close; trade next open","news_note":"Astock CREATED_DATE/text only; no labels, future prices, CHANGE or cumulative READ"}
    pd.DataFrame([audit]).to_csv(OUT/"audit.csv",index=False)
    sm.to_csv(OUT/"summary.csv",index=False); eq.to_csv(OUT/"equity.csv",index=False); tr.to_csv(OUT/"trades.csv",index=False); te.to_csv(OUT/"timing_events.csv",index=False); st.to_csv(OUT/"market_states.csv",index=False)
    pd.DataFrame([{**a,"description":AGENT_DESCRIPTIONS[a["strategy"]]} for a in AGENTS]).to_csv(OUT/"agent_definitions.csv",index=False)
    # Split-period robustness and best-five dependence.
    mid=START+(END-START)/2; seg=[]
    for a in AGENTS:
        e=eq[eq.strategy==a["strategy"]].set_index("date").equity
        for name,a0,z0 in [("H1",START,mid),("H2",mid,END)]:
            x=e[(e.index>=a0)&(e.index<=z0)]
            if len(x)>1: seg.append({"strategy":a["strategy"],"segment":name,"return":x.iloc[-1]/x.iloc[0]-1})
    pd.DataFrame(seg).to_csv(OUT/"segments.csv",index=False)
    robust=[]
    for a in AGENTS:
        t=tr[tr.strategy==a["strategy"]] if len(tr) else pd.DataFrame()
        total=t.net_pnl.sum() if len(t) else 0.; b5=t.nlargest(min(5,len(t)),"net_pnl").net_pnl.sum() if len(t) else 0.
        robust.append({"strategy":a["strategy"],"completed_trade_pnl":total,"best5_pnl":b5,"pnl_without_best5":total-b5})
    pd.DataFrame(robust).to_csv(OUT/"robustness.csv",index=False)
    # Keep auditable examples of actual historical news mapped to the next trade date.
    news_agg[["exec_date","code","name","max_news_time","sentiment","news_count","theme_heat","title_sample"]].head(300).to_csv(OUT/"news_examples.csv",index=False)
    print("\n=== AUDIT ===\n",pd.DataFrame([audit]).to_string(index=False),flush=True)
    print("\n=== 10 TRADER LEADERBOARD ===\n",sm.to_string(index=False),flush=True)
    print("\n=== SEGMENTS ===\n",pd.DataFrame(seg).to_string(index=False),flush=True)
    print("\n=== ROBUSTNESS ===\n",pd.DataFrame(robust).to_string(index=False),flush=True)

if __name__=="__main__": main()
