from __future__ import annotations
import numpy as np
import pandas as pd

import run_execution_units_fixed as exfix
exfix.install()

import run_10y_alpha500_size_value as research

# Compatibility-only schema patch. DoltHub may return information_schema field
# labels with different casing; normalize labels, but do not alter source
# viability thresholds, PIT rules, strategy definitions, or audit gates.
def schema_frame_fixed():
    rows=research.sql("SELECT table_name,column_name,data_type FROM information_schema.columns WHERE table_schema=DATABASE() ORDER BY table_name,ordinal_position")
    z=pd.DataFrame(rows)
    if z.empty:
        raise RuntimeError('DoltHub information_schema query returned no rows')
    mp={research.norm(c):c for c in z.columns}
    need={'tablename':'table_name','columnname':'column_name','datatype':'data_type'}
    ren={}
    for canonical,out in need.items():
        src=mp.get(canonical)
        if src is None:
            raise RuntimeError(f'DoltHub schema response missing {out}; columns={list(z.columns)} sample={z.head(3).to_dict(orient="records")}')
        ren[src]=out
    z=z.rename(columns=ren)[['table_name','column_name','data_type']]
    z.to_csv(research.OUT/'dolthub_schema.csv',index=False)
    return z

research.schema_frame = schema_frame_fixed

# Literature-guided extensions: earnings yield (E/P), turnover, and residual
# momentum. These are fixed formulas; no continuous weight optimizer is used.
_ORIG_RERANK = research.rerank
EXTRA_FAMILIES = (
    'ep', 'size_ep', 'size_ep_ivol', 'size_ep_eff', 'size_ep_bal',
    'exmicro30_ep', 'exmicro30_ep_ivol', 'exmicro30_ep_eff',
    'small30_ep_eff', 'small20_ep_ivol',
    'lowturn', 'size_lowturn', 'size_ep_lowturn', 'lowturn_ivol_eff',
    'rmom', 'size_ep_rmom', 'size_ep_ivol_rmom',
)
research.FAMILIES = tuple(research.FAMILIES) + EXTRA_FAMILIES


def fetch_cross_section_ext(src,d,head_hash):
    t=src['table']; cols=[f"`{src['code']}` AS code_raw",f"`{src['date']}` AS pit_date"]
    for k,a in [('circ','circ_mv'),('total','total_mv'),('turn','turn'),('close','raw_close'),('vol','volume'),('amount','amount'),('pb','pb'),('pe','pe'),('ps','ps'),('pcf','pcf'),('isst','is_st'),('trade','trade_status')]:
        cols.append(research.expr(src,k,a))
    wh=research.date_where(src,d); select=', '.join(cols)
    rows=None; last=None
    for qq in [f"SELECT {select} FROM `{t}` AS OF '{head_hash}' WHERE {wh} LIMIT 10000",f"SELECT {select} FROM `{t}` WHERE {wh} LIMIT 10000"]:
        try:
            rows=research.sql(qq); break
        except Exception as e: last=e
    if rows is None: raise last
    z=pd.DataFrame(rows)
    if z.empty: return z
    z['signal_date']=pd.Timestamp(d); z['code']=z.code_raw.map(research.qcode)
    for c in ['circ_mv','total_mv','turn','raw_close','volume','amount','pb','pe','ps','pcf','is_st','trade_status']:
        z[c]=pd.to_numeric(z[c],errors='coerce')
    size=z.circ_mv.where(z.circ_mv>0).fillna(z.total_mv.where(z.total_mv>0))
    if size.isna().any() and src.get('turn'):
        turn=z.turn.where(z.turn>0)
        der=(z.amount*100.0/turn) if src.get('amount') else (z.raw_close*z.volume*100.0/turn)
        size=size.fillna(der.where(der>0))
    z['pit_size']=size
    return z[['signal_date','code','pit_size','turn','pb','pe','ps','pcf','is_st','trade_status']].drop_duplicates(['signal_date','code'],keep='last')

research.fetch_cross_section = fetch_cross_section_ext


def rerank_ext(q0,family,pool):
    if family not in EXTRA_FAMILIES:
        return _ORIG_RERANK(q0,family,pool)
    q=q0.copy(); q['rank_test']=np.nan; m=research.pool_mask(q,pool)
    need_ep = ('ep' in family)
    need_turn = ('turn' in family)
    need_rmom = ('rmom' in family)
    if need_ep: m=m&np.isfinite(q.pe)&(q.pe>0)
    if need_turn: m=m&np.isfinite(q.turn)&(q.turn>0)
    if need_rmom: m=m&np.isfinite(q.rmom126)
    if not m.any(): return q

    rs=research.pct(q,m,'pit_size',True)
    ri=research.pct(q,m,'ivol60',True)
    re=research.pct(q,m,'eff120',False)
    rep=research.pct(q,m,'pe',True) if need_ep else None  # low positive PE == high E/P
    rt=research.pct(q,m,'turn',True) if need_turn else None
    rr=research.pct(q,m,'rmom126',False) if need_rmom else None

    # Size gates are applied only after PIT ranks are known.
    if family.startswith('exmicro30_'):
        gate=rs>.30; mm=pd.Series(False,index=q.index); mm.loc[gate.index]=gate; m=m&mm
    elif family.startswith('small30_'):
        gate=rs<=.30; mm=pd.Series(False,index=q.index); mm.loc[gate.index]=gate; m=m&mm
    elif family.startswith('small20_'):
        gate=rs<=.20; mm=pd.Series(False,index=q.index); mm.loc[gate.index]=gate; m=m&mm
    if not m.any(): return q

    # Recompute ranks after a size gate so percentile weights refer to the final PIT universe.
    rs=research.pct(q,m,'pit_size',True); ri=research.pct(q,m,'ivol60',True); re=research.pct(q,m,'eff120',False)
    rep=research.pct(q,m,'pe',True) if need_ep else None
    rt=research.pct(q,m,'turn',True) if need_turn else None
    rr=research.pct(q,m,'rmom126',False) if need_rmom else None

    if family=='ep': raw=rep
    elif family=='size_ep': raw=.55*rs+.45*rep
    elif family=='size_ep_ivol': raw=.45*rs+.30*rep+.25*ri
    elif family=='size_ep_eff': raw=.45*rs+.30*rep+.25*re
    elif family=='size_ep_bal': raw=.40*rs+.30*rep+.15*ri+.15*re
    elif family=='exmicro30_ep': raw=rep
    elif family=='exmicro30_ep_ivol': raw=.60*rep+.40*ri
    elif family=='exmicro30_ep_eff': raw=.60*rep+.40*re
    elif family=='small30_ep_eff': raw=.55*rep+.45*re
    elif family=='small20_ep_ivol': raw=.55*rep+.45*ri
    elif family=='lowturn': raw=rt
    elif family=='size_lowturn': raw=.60*rs+.40*rt
    elif family=='size_ep_lowturn': raw=.40*rs+.35*rep+.25*rt
    elif family=='lowturn_ivol_eff': raw=.45*rt+.30*ri+.25*re
    elif family=='rmom': raw=rr
    elif family=='size_ep_rmom': raw=.40*rs+.35*rep+.25*rr
    elif family=='size_ep_ivol_rmom': raw=.35*rs+.30*rep+.20*ri+.15*rr
    else: raise ValueError(family)
    q.loc[raw.index,'rank_test']=raw.groupby(q.loc[raw.index,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q

research.rerank = rerank_ext

if __name__ == '__main__':
    research.main()
