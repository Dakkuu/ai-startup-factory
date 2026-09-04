# T1 Inventory Exhaustion Alpha (T1-IE) — Preregistration 2026-09-04

Status: preregistered before first backtest of this mechanism.

## Research objective
Create a short-horizon A-share alpha from market microstructure and trading rules rather than reuse popular indicators such as RSI, MACD, moving-average crossovers, generic momentum, generic reversal, breakout, or conventional limit-up continuation.

This is NOT claimed to be globally unique. It is a self-developed, relatively non-standard mechanism. Public code commonly studies limit-up, failed limit-up, board continuation, and next-day premium, but this study specifically tests a two-day path motivated by T+1 inventory release and supply exhaustion.

## Economic mechanism
1. Event day E: price approaches the board-specific upper price limit, attracting late demand, but fails to retain the high and closes materially below the intraday high. Volume is elevated. This creates a cohort of late buyers who cannot sell until the following session because of T+1.
2. Flush day F=E+1: the stock trades weak/negative and releases that trapped inventory. If volume contracts sharply versus E while the close recovers materially from the session low, the hypothesis is that forced supply was absorbed and marginal selling pressure is exhausted.
3. Signal is generated only at the close of F. Entry is no earlier than the next session open.
4. The expected alpha horizon is 2-8 sessions. The mechanism is neither generic reversal nor limit-up continuation: the T+1 release day and absorption signature are required.

## Raw causal features
All prices are converted to raw-price units with Qlib factor where required. Relative volume uses reconstructed raw-share volume. No future prices enter the signal.

Event day E features:
- `event_pressure = (high_E / prev_close_E - 1) / board_limit(E)`
- `event_close_pressure = (close_E / prev_close_E - 1) / board_limit(E)`
- `event_fade = (high_E - close_E) / prev_close_E / board_limit(E)`
- `event_vol_ratio = raw_volume_E / median(raw_volume_{E-20:E-1})`

Flush day F features:
- `flush_gap_n = (open_F / close_E - 1) / board_limit(F)`
- `flush_ret_n = (close_F / close_E - 1) / board_limit(F)`
- `flush_low_n = (low_F / close_E - 1) / board_limit(F)`
- `flush_vol_ratio = raw_volume_F / raw_volume_E`
- `flush_recovery = (close_F - low_F) / max(high_F - low_F, epsilon)`

Liquidity:
- trailing 20-session raw turnover; keep the top 70% by cross section unless overridden.

## Base event definition
The first sweep uses only coarse mechanism variants, not a fine optimization grid.

Common rules:
- event_pressure >= threshold
- event_fade >= threshold
- event_close_pressure < close_seal_max (must fail to retain near-limit close)
- event_vol_ratio >= threshold
- flush_ret_n between configured min/max
- flush_gap_n <= configured max
- flush_vol_ratio <= threshold
- flush_recovery >= threshold
- flush_low_n <= configured maximum (must actually flush intraday)
- liquidity percentile <= 70%

## T1-IE score
Among same-day valid events, lower rank is better after ranking the following positive components:
- 25% event_pressure
- 20% event_fade
- 25% volume compression = 1 - flush_vol_ratio
- 25% flush_recovery
- 5% controlled flush depth = clip(-flush_ret_n, 0, 0.50)

No generic momentum or fundamental factor is allowed in V1 score.

## Pre-registered coarse signal variants
- `balanced`: pressure .75, fade .12, event volume 1.30, flush volume <= .80, recovery >= .55
- `strict_pressure`: pressure .85, fade .12, event volume 1.30, flush volume <= .80, recovery >= .55
- `strict_absorption`: pressure .75, fade .12, event volume 1.30, flush volume <= .65, recovery >= .65
- `deep_trap`: pressure .75, fade .20, event volume 1.50, flush volume <= .80, recovery >= .55
- `quiet_release`: pressure .75, fade .12, event volume 1.50, flush volume <= .65, recovery >= .55
- `high_recovery`: pressure .75, fade .12, event volume 1.30, flush volume <= .80, recovery >= .70

Shared initial bounds:
- close_seal_max = .90 board-limit normalized
- flush_ret_n in [-.50, .10]
- flush_gap_n <= .10
- flush_low_n <= -.08

## Portfolio parameters
Search only:
- position cap N in {5, 10, 15}
- signal memory / intended holding window in {3, 5, 8} sessions
- Entry threshold=.10, Keep=.30
- post-expiry execution retry rows=10 sessions
- equal target capital per slot
- no leverage

Signal-day event rank is mapped inside Entry10. During the memory window after the original signal, the candidate receives a synthetic rank between Entry10 and Keep30 so an existing position can be retained but a stale candidate cannot newly enter. On expiry it receives NaN rank while execution rows remain available for sell retries.

## Execution
- signal at flush-day close
- earliest trade next session open
- board/date limit-gap proxy
- no buy at blocked/locked upper-limit open
- no sell at blocked/locked lower-limit open
- 100-share lots
- 5% raw daily share volume participation cap
- existing fee and slippage model
- blocked exits continue consuming a slot

## Selection split
- Train/selection: 2016-08-02 through 2021-12-31
- Pseudo-OOS diagnostic: 2022-01-01 through 2026-07-29
- 2022-2026 must not select the variant, N, or memory window.

Selection on train only:
1. train CAGR > 0
2. train MDD > -45%
3. both 2016-2019 and 2020-2021 CAGR/return positive where computable
4. maximize train Calmar
5. tie-break by Sharpe, then lower turnover

## Required robustness outputs
- all pre-registered grid results
- selected equity/trades/timing
- full/train/pseudo metrics
- annual returns
- 2x and 4x transaction-cost stress
- reverse-score control
- event-definition ablation: event day near-limit failure without flush absorption confirmation
- candidate counts by year
- parameter dump and code hash

## Promotion gate
This alpha is not promoted unless:
- selected train CAGR > 0 and pseudo CAGR > 0
- train and pseudo Sharpe > 0
- full MDD > -45%
- 2x-cost CAGR > 0
- 4x-cost result is reported
- absorption-confirmed strategy outperforms the event-only ablation on train Calmar
- no timing violation

## Future parameterization requirement
All economically meaningful thresholds and portfolio settings must live in JSON config and be overridable from one runtime config file. Future tests should change config only; core engine should not require code edits.