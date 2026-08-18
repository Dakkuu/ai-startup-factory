import run_behavioral_news_backtest as b

_original = b.prepare_candidates

def prepare_candidates_fixed(prev, news_today):
    # Preserve identical data/strategy logic; only remove pandas label/index ambiguity.
    if 'code' in getattr(prev.index, 'names', []):
        prev = prev.reset_index(drop=True)
    if 'code' in getattr(news_today.index, 'names', []):
        news_today = news_today.reset_index(drop=True)
    return _original(prev, news_today)

b.prepare_candidates = prepare_candidates_fixed

if __name__ == '__main__':
    b.main()
