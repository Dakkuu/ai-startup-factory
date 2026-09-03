# A-share Multi-Alpha System Preregistration — 2026-09-03

Status: preregistered before running the new multi-alpha search.

## Goal
Build one capital system that harvests multiple economically distinct alpha sources instead of optimizing one GEff sleeve. The system must contain short-, medium-, and long-horizon sleeves and must be evaluated as a portfolio, not only as standalone strategies.

## Data / execution rules
- Research window: 2016-08-02 through 2026-07-29.
- Selection window: 2016-08-02 through 2021-12-31 only.
- Pseudo-OOS diagnostic window: 2022-01-01 through 2026-07-29. This is research-contaminated, not clean OOS.
- All signals use information available on or before signal date; execution no earlier than next session.
- Use existing signal-pure A-share universe, 100-share lots, volume participation limits, explicit next-session execution, board/date price-limit proxy, and blocked execution logic.
- No leverage.
- No endpoint calibration.
- No selection based on 2022-2026 performance.

## Sleeve S — short horizon / behavioral-price-volume alpha
Economic source: short-horizon price pressure, reversal, breakout/continuation, and volume acceleration. No fundamental variables and no GEff score may enter this sleeve.

Signal families, fixed before search:
1. `rev5`: lower trailing 5-session return is better.
2. `mom20`: higher trailing 20-session return is better.
3. `breakout`: 50% high 20-session momentum + 30% closeness to prior 60-session high + 20% high relative volume (20/120).
4. `accel`: 60% high return acceleration (`ret20 - ret60/3`) + 40% high relative volume (20/120).

Search grid:
- holding horizon H in {5, 10, 20} sessions;
- N in {10, 15};
- entry/keep in {(5%,20%), (10%,30%)};
- evaluate every phase implied by each H and combine phases equally for candidate evaluation.

Selection objective on 2016-2021 only:
1. train CAGR > 0;
2. train MDD > -45%;
3. maximize train Calmar;
4. tie-break by train Sharpe, then lower turnover proxy.

## Sleeve M — medium horizon / GEff + PIT fundamental alpha
Economic source: residual momentum / low residual risk / efficiency combined with PIT cash-flow and quality-value information.

This sleeve is fixed before this multi-alpha search and is not retuned here:
- signal candidate: `mom_cfo10_qv10` = 80% technical GEff-like score + 10% CFO/assets + 10% quality-value;
- holding horizon H60;
- Entry10 / Keep30;
- rank tilt = 75% N10 sleeve + 25% N5 sleeve;
- staggered phases = {0,4,8}, equal capital across phases.

## Sleeve L — long horizon / value-quality-cash-flow alpha
Economic source: valuation and business quality. Momentum is not part of the long alpha score.

Fixed signal families:
1. `value`: PIT value3 rank (earnings yield, book yield, cash-flow yield).
2. `quality`: PIT ROE + gross margin + CFO/assets + accrual quality + cash conversion.
3. `value_quality`: equal blend of value and quality.

Search grid:
- H in {120, 180, 250} sessions;
- N in {15, 20, 30};
- Entry10 / Keep30 fixed;
- use four evenly spaced staggered phases for H>=120 (or all phases if fewer than four exist).

Selection objective on 2016-2021 only:
1. train CAGR > 0;
2. train MDD > -45%;
3. maximize train Calmar;
4. tie-break by train Sharpe.

## Multi-alpha capital allocation
After S and L are selected using 2016-2021 only, test only the following coarse fixed capital splits. No continuous optimizer is allowed in this round:
- A1 = 25% short / 50% medium / 25% long
- A2 = 30% short / 40% medium / 30% long
- A3 = 20% short / 50% medium / 30% long
- A4 = 20% short / 40% medium / 40% long
- A5 = 1/3 short / 1/3 medium / 1/3 long

Allocation selection uses 2016-2021 only and maximizes train Calmar subject to train CAGR > each sleeve's simple weighted-average CAGR minus 2 percentage points. Tie-break by train Sharpe.

Capital is modeled as separate fixed sleeves from the evaluation start; no performance-chasing reallocation.

## Required diversification diagnostics
For selected S/M/L:
- train and pseudo-OOS daily-return correlation matrices;
- correlation on benchmark down days;
- full-period and rolling 252-session pairwise correlation distribution;
- standalone CAGR/MDD/Sharpe;
- portfolio CAGR/MDD/Sharpe;
- portfolio without each sleeve to measure marginal contribution;
- annual returns by sleeve and portfolio;
- 2x and 4x transaction-cost stress for selected sleeves and selected allocation.

## Promotion gates
The multi-alpha system is only promoted to a serious shadow candidate if all are true:
1. each sleeve has positive 2022-2026 CAGR without being selected on that period;
2. selected portfolio 2022-2026 CAGR > 0;
3. selected portfolio train Calmar exceeds the medium sleeve alone;
4. selected portfolio full-history MDD is no worse than medium-alone MDD by more than 5 percentage points;
5. at least one of S-M or L-M full-period daily correlations is <= 0.60;
6. removing any one sleeve does not increase full-history Sharpe by more than 0.10 (otherwise that sleeve is not earning its capital budget);
7. selected allocation remains positive CAGR under 2x costs; 4x cost result is reported as stress, not necessarily a hard veto.

## Interpretation
This round seeks diversification of alpha sources, not the highest backtest CAGR. A high-return sleeve that is effectively another copy of medium GEff does not satisfy the objective.