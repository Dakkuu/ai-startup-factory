from pathlib import Path

src=Path('run_10y_era_backtest.py').read_text(encoding='utf-8')
old='''    p=p.groupby("signal_date",group_keys=False).apply(ranks,include_groups=False).reset_index(drop=True)'''
new='''    # Preserve signal_date explicitly; avoid groupby.apply dropping the grouping column.\n    rank_map={\n        "r_ret1":("ret1",True),"r_mom5":("mom5",True),"r_mom20":("mom20",True),\n        "r_mom60":("mom60",True),"r_mom120":("mom120",True),\n        "r_lowvol20":("vol20",False),"r_lowvol60":("vol60",False),\n        "r_volratio":("vol_ratio",True),"r_liq":("liq_ma20",True),\n        "r_ma20gap":("ma20gap",True),"r_ma60gap":("ma60gap",True),"r_high20":("high20",True),\n    }\n    for outcol,(incol,ascending) in rank_map.items():\n        p[outcol]=p.groupby("signal_date")[incol].rank(pct=True,method="average",ascending=ascending)\n    p=p.reset_index(drop=True)'''
if old not in src:
    raise RuntimeError('expected rank implementation not found')
src=src.replace(old,new)
exec(compile(src,'run_10y_era_backtest_v2_patched.py','exec'),{'__name__':'__main__'})
