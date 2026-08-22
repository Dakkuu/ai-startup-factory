from __future__ import annotations
import run_execution_units_fixed as exfix

# Install the audited lots->shares conversion before the research module calls
# either exact daily-MTM or sparse discovery execution.
exfix.install()

import run_10y_alpha500_size_value as research

if __name__ == '__main__':
    research.main()
