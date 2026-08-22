from __future__ import annotations
import run_10y_hard_executor_v2 as hv2
hv2.patch()
import run_10y_max_audit_fixed as compat
import run_10y_max_audit as audit

# Same frozen strategy and same preregistered audit gates; only execution volume units are corrected.
if __name__=='__main__':
    audit.main()
