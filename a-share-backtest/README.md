# A-share backtest sandbox

Isolated research sandbox for point-in-time A-share strategy backtests. The workflow fetches public market data in GitHub Actions, enforces T+1 execution, transaction costs, basic limit-lock constraints, and avoids same-day look-ahead.

Current experiment window: 2025-08-01 through 2026-08-18. Historical CSI500 membership is queried by date through BaoStock so the tradable universe is not reconstructed from today's constituents.
