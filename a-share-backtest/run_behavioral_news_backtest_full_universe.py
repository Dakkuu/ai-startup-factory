import numpy as np
import pandas as pd
import run_behavioral_news_backtest as b


def prepare_candidates_full(prev, news_today):
    # Structural fix: institutional/quant agents see the entire historically-active
    # A-share cross section. News fields are a point-in-time overlay, not a mandatory gate.
    if 'code' in getattr(prev.index, 'names', []):
        prev = prev.reset_index(drop=True)
    if 'code' in getattr(news_today.index, 'names', []):
        news_today = news_today.reset_index(drop=True)
    base = prev.copy()
    ncols=['code','news_count','sentiment','pos_words','neg_words','announcement','theme_heat','max_news_time','novelty','title_sample','name']
    if news_today.empty:
        n=pd.DataFrame(columns=ncols)
    else:
        n=news_today[[c for c in ncols if c in news_today.columns]].copy()
    c=base.merge(n,on='code',how='left',suffixes=('','_news'))
    for col in ['news_count','sentiment','pos_words','neg_words','announcement','theme_heat','novelty']:
        if col not in c: c[col]=0.0
        c[col]=pd.to_numeric(c[col],errors='coerce').fillna(0.0)
    if 'max_news_time' not in c: c['max_news_time']=pd.NaT
    c['max_news_time']=pd.to_datetime(c.max_news_time,errors='coerce')
    # For no-news names the latest information used is previous trading close.
    fallback=pd.to_datetime(c['date'])+pd.Timedelta(hours=15)
    c['max_news_time']=c['max_news_time'].fillna(fallback)
    c['has_news']=(c.news_count>0).astype(int)
    needed=['ret1','mom5','mom20','mom60','vol20','vol_ratio','liq_ma20','dist_high20']
    c=c.dropna(subset=needed)
    c=c[(c.close>0)&(c.open>0)&(c.liq_ma20>=b.MIN_LIQ)].copy()
    if c.empty:return c
    c['r_mom20']=b.pct_rank(c.mom20); c['r_mom60']=b.pct_rank(c.mom60)
    c['r_liq']=b.pct_rank(c.liq_ma20); c['r_lowvol']=b.pct_rank(-c.vol20)
    c['r_volratio']=b.pct_rank(c.vol_ratio); c['r_news']=b.pct_rank(c.news_count)
    c['r_novelty']=b.pct_rank(c.novelty); c['r_theme']=b.pct_rank(c.theme_heat)
    c['r_sent']=b.pct_rank(c.sentiment); c['r_drop']=b.pct_rank(-c.ret1)
    return c


def choose_full(c,state,agent):
    if c.empty:return c
    s=agent['strategy']; x=c.copy()
    if s=='01_defensive_institution':
        if state['regime']=='cold':
            # Do not disappear completely: hold only the most liquid/low-vol names.
            x=x[(x.sentiment>=0)&(x.mom20>-.08)&(x.ret1<.06)&(x.vol_ratio<2.5)]
            x['score']=.38*x.r_lowvol+.32*x.r_liq+.15*x.r_mom60+.10*x.r_mom20+.05*x.r_sent
        else:
            x=x[(x.sentiment>=0)&(x.mom20>-.05)&(x.ret1<.07)&(x.vol_ratio<3)]
            x['score']=.32*x.r_liq+.28*x.r_lowvol+.20*x.r_mom60+.15*x.r_mom20+.05*x.r_sent
    elif s=='02_quality_event':
        x=x[(x.has_news==1)&(x.announcement==1)&((x.sentiment>=2)|(x.pos_words>=2))&(x.ret1<.07)&(x.mom20>-.10)]
        x['score']=.30*x.r_sent+.20*x.r_novelty+.20*x.r_liq+.15*x.r_lowvol+.15*x.r_mom20
    elif s=='03_smart_money_trend':
        x=x[(x.sentiment>=0)&(x.mom20>0)&(x.mom60>0)&(x.vol_ratio.between(.7,2.8))]
        x['score']=.30*x.r_liq+.25*x.r_mom60+.20*x.r_mom20+.15*x.r_lowvol+.05*x.r_sent+.05*x.r_news
    elif s=='04_quant_crowding':
        x=x[(x.sentiment>=-1)&(x.mom20>-.02)]
        x['score']=.30*x.r_mom20+.25*x.r_mom60+.20*x.r_lowvol+.20*x.r_liq+.05*x.r_news
    elif s=='05_retail_attention_chase':
        x=x[(x.has_news==1)&(x.sentiment>0)&((x.news_count>=2)|(x.theme_heat>=3)|(x.novelty>1.2))&(x.ret1>.005)&(x.vol_ratio>1.25)&(x.dist_high20>-.05)]
        x['score']=.25*x.r_news+.20*x.r_novelty+.20*x.r_volratio+.15*x.r_theme+.10*x.r_sent+.10*x.r_mom20
    elif s=='06_hot_money_theme_relay':
        x=x[(x.has_news==1)&(x.theme_heat>=3)&(x.sentiment>=0)&(x.ret1.between(.02,.095))&(x.vol_ratio>1.4)&(x.mom5>0)]
        x['score']=.35*x.r_theme+.25*x.r_volratio+.15*x.r_news+.15*x.r_mom20+.10*x.r_sent
    elif s=='07_limit_up_relay':
        if state['regime']!='hot': x=x.iloc[0:0]
        else:
            # A limit-up itself is an attention event; explicit news is not mandatory.
            x=x[(x.ret1>=.09)&(x.mom20>0)&(x.vol_ratio>.9)&(x.sentiment>=-1)]
            x['score']=.30*x.r_volratio+.25*x.r_mom20+.15*x.r_theme+.15*x.r_news+.15*x.r_liq
    elif s=='08_panic_reversal':
        panic=((x.has_news==1)&(x.sentiment<=-1)) | (x.ret1<=-.07)
        x=x[panic&((x.ret1<=-.03)|(x.mom5<=-.08))&(x.vol_ratio>1.15)]
        x['score']=.35*x.r_drop+.20*x.r_liq+.20*x.r_volratio+.15*x.r_news+.10*x.r_novelty
    elif s=='09_bad_news_defense':
        x=x[(x.sentiment>=0)&(x.ret1<.06)&(x.mom20>-.08)]
        x['score']=.35*x.r_lowvol+.30*x.r_liq+.15*x.r_mom20+.10*x.r_mom60+.10*x.r_sent
    elif s=='10_adaptive_barbell':
        if state['regime']=='hot':
            hot=((x.theme_heat>=2)&(x.has_news==1)) | (x.ret1>=.08)
            x=x[hot&(x.sentiment>=-1)&(x.mom20>0)&(x.vol_ratio>1.0)]
            x['score']=.25*x.r_theme+.25*x.r_volratio+.20*x.r_mom20+.15*x.r_news+.15*x.r_liq
        elif state['regime']=='cold':
            x=x[(x.sentiment>=0)&(x.ret1<.05)&(x.mom20>-.08)]
            x['score']=.40*x.r_lowvol+.30*x.r_liq+.15*x.r_mom20+.10*x.r_mom60+.05*x.r_sent
        else:
            # Neutral market: blend quality event and smooth trend rather than force news.
            x=x[(x.sentiment>=0)&(x.ret1<.07)&(x.mom20>0)]
            x['score']=.25*x.r_liq+.20*x.r_lowvol+.20*x.r_mom20+.15*x.r_mom60+.10*x.r_sent+.10*x.r_news
    else: raise ValueError(s)
    return x.sort_values('score',ascending=False).head(agent['max_names']) if len(x) else x

b.prepare_candidates=prepare_candidates_full
b.choose=choose_full

if __name__=='__main__':
    b.main()
