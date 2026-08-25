import run_10y_china_behavior_daily as m
x=m.load_data()
print('load_data type',type(x),'len',len(x) if hasattr(x,'__len__') else None)
if isinstance(x,tuple):
    for i,v in enumerate(x):
        print(i,type(v),getattr(v,'shape',None))
        if hasattr(v,'__len__') and not hasattr(v,'shape'):
            try: print(' sample',list(v)[:3])
            except: pass
# Try known unpack based on simulate signature assumptions
try:
    dates,codes,close,open_,high,volume,factor,member=x
    print('dates',dates[:3],dates[-3:])
    print('codes',codes[:10])
    feats,market=m.build_features(dates,codes,close,open_,high,volume,factor,member)
    print('feats type',type(feats))
    if isinstance(feats,dict):
        for k,v in feats.items(): print('feat',k,getattr(v,'shape',None), 'nan%', float((~(v==v)).mean()) if hasattr(v,'shape') else '')
    print('market type',type(market))
    print(market.head() if hasattr(market,'head') else str(market)[:3000])
except Exception as e:
    import traceback; traceback.print_exc()
