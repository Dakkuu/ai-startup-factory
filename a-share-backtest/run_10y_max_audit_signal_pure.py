from __future__ import annotations
import run_10y_hard_executor_v2 as hv2
import run_10y_signal_pure_panel as sp
hv2.patch(); sp.patch()
import run_10y_max_audit_fixed as compat
import run_10y_max_audit as audit
sp.OUT=audit.OUT

if __name__=='__main__':
    audit.main()
