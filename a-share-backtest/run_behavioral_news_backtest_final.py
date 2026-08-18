import re
import pandas as pd
import run_behavioral_news_backtest as b
import run_behavioral_news_backtest_full_universe  # applies full-universe candidate/agent patches

_original_load = b.load_all_membership

_STOCK_RE = re.compile(r'^(?:SH(?:600|601|603|605|688)\d{3}|SZ(?:000|001|002|003|300|301)\d{3}|BJ\d{6})$')

def load_stock_only_membership(cal):
    raw, _ = _original_load(cal)
    df = raw[raw.code.astype(str).str.match(_STOCK_RE)].copy()
    dates = cal[(cal>=b.START)&(cal<=b.END)]
    counts=[int(((df.start<=d)&(df.end>=d)).sum()) for d in dates]
    union=df[(df.end>=b.START)&(df.start<=b.END)].code.nunique()
    entered=df[(df.start>b.START)&(df.start<=b.END)].code.nunique()
    exited=df[(df.end>=b.START)&(df.end<b.END)].code.nunique()
    if union < 3000 or min(counts) < 2800 or exited < 5:
        raise RuntimeError(f'FAIL-CLOSED stock-only universe suspicious union={union} daily={min(counts)}..{max(counts)} exits={exited}')
    # Verify index-like codes are excluded.
    bad=df[df.code.str.match(r'^(SH000|SZ399)')]
    if len(bad):
        raise RuntimeError(f'FAIL-CLOSED index codes leaked into stock universe: {bad.code.head().tolist()}')
    audit={
        'instrument_file':'all.txt_stock_only',
        'union_members':union,'entered':entered,'exited':exited,
        'min_daily_members':min(counts),'max_daily_members':max(counts),
    }
    print('stock-only universe audit',audit,flush=True)
    return df,audit

b.load_all_membership = load_stock_only_membership

if __name__=='__main__':
    b.main()
