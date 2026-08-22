from __future__ import annotations
import run_10y_hard_executor_v2 as hv2
hv2.patch()
import run_10y_external_audit_v2 as audit

if __name__=='__main__':
    audit.main()
