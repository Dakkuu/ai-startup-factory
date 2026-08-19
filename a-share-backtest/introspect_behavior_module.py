import inspect
import run_10y_china_behavior_daily as m
for name in sorted(dir(m)):
    if name.startswith('_'): continue
    obj=getattr(m,name)
    if inspect.isfunction(obj) or inspect.isclass(obj):
        try: sig=str(inspect.signature(obj))
        except Exception: sig='?'
        print(name, type(obj).__name__, sig)
    elif name.isupper():
        try:
            print(name, type(obj).__name__, repr(obj)[:500])
        except Exception: pass
