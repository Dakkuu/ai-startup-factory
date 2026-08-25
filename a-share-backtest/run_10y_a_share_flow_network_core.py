from pathlib import Path
import pandas as pd
import run_10y_a_share_flow_network as m

m.OUT = Path('results_10y_flow_network_core')
m.OUT.mkdir(exist_ok=True)
m.VARIANTS = [
    '01_leader_only',
    '02_network_leader',
    '03_network_auction',
    '04_network_auction_lhb',
    '05_full_regime_network_auction_lhb',
]

def _no_events(start,end):
    pd.DataFrame([{'status':'disabled_in_10y_core','reason':'semantic news tested separately to isolate incremental value'}]).to_csv(m.OUT/'event_coverage.csv',index=False)
    return pd.DataFrame()

m.fetch_sparse_events = _no_events
m.main()
a=m.pd.read_csv(m.OUT/'audit.csv')
a['news_layer']='disabled in 10y core; separate point-in-time news validation'
a.to_csv(m.OUT/'audit.csv',index=False)
