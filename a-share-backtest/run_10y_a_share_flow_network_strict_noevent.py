from pathlib import Path
import pandas as pd
import run_10y_a_share_flow_network_strict as s

s.OUT=Path('results_10y_flow_network_strict_noevent')
s.OUT.mkdir(exist_ok=True)
s.m.OUT=s.OUT
s.fetch_events_parallel=lambda: pd.DataFrame()

if __name__=='__main__':
    s.main()
