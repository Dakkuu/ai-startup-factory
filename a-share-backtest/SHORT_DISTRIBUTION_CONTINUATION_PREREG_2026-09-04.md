# Short Distribution Continuation (SDC) — Post-Diagnostic Preregistration — 2026-09-04

Status: **POST-DIAGNOSTIC**. This hypothesis was proposed only after observing that the prior long event-only failed-board strategy lost heavily. Results below must never be described as clean OOS.

## Primary question
Does the exact opposite *position direction* of the previously failed event-only strategy have positive economic value when implemented as a true short book rather than a reverse ranking that remains long?

## Primary signal — frozen exact mirror
Use the already-generated `eventonly_strict_absorption.csv.gz` signal file from T1-IE V2. No signal thresholds are changed.

Primary portfolio configuration is frozen to the configuration inherited from the prior train-only T1-IE selection:
- source: event-only failed near-limit event (no T+1 absorption confirmation)
- variant thresholds: `strict_absorption` event thresholds (same event thresholds as the prior ablation)
- N = 15
- active memory = 8 trading sessions
- score priority = same `score_rank`, ascending (same names that the prior long engine would prefer)
- signal confirmed at event-day close
- earliest short sale = next trading-session open
- fresh repeated event in an already-short name refreshes the 8-session memory window
- after memory expiry, cover is attempted every session until executable; blocked cover continues occupying a slot
- no leverage: gross short market value <= portfolio equity
- equal target capital per slot
- 100-share raw lots
- 5% raw daily share-volume participation cap

## A-share short execution
This is a true short-position simulator:
- open = sell short; adverse slippage uses a lower sale price
- close = buy to cover; adverse slippage uses a higher cover price
- opening short cannot execute at a locked/blocked lower-limit open
- covering cannot execute at a locked/blocked upper-limit open
- normal stock transaction fees use the existing A-share fee model: sell-side costs at short opening, buy-side costs at cover
- short-sale proceeds are segregated and are not reused to lever the book
- baseline stock-borrow fee = 8% annualized, accrued daily on current short market value

Historical stock-specific lend availability is not available in the cached dataset. Therefore the primary test is an **economic short-alpha backtest under assumed borrow availability**, not a claim that every historical trade was borrowable. Borrow-rate and liquidity feasibility stresses are mandatory.

## Frozen diagnostics (not selection)
No optimizer is allowed in this round. Report only:
1. exact mirror: N15, memory8, event-only strict_absorption
2. same exact mirror with borrow rate 0%, 8%, 15%, 30%
3. same exact mirror at transaction cost 1x, 2x, 4x while borrow stays 8%
4. same exact mirror restricted to a liquid feasibility proxy `liq20 >= 2 * liq_threshold`; this proxy is NOT historical lend availability and is reported only as a capacity diagnostic
5. secondary mechanism control: true short of `signals_strict_absorption.csv.gz` (T+1 absorption-confirmed events), same N15/memory8, to test whether waiting for the former long confirmation helps or harms the short thesis

## Evaluation windows
- full: 2016-08-02 through 2026-07-29
- historical split 1: 2016-08-02 through 2021-12-31
- historical split 2: 2022-01-01 through 2026-07-29

Because this entire study is post-diagnostic, neither split is called clean OOS. They are regime-stability diagnostics only.

## Required outputs
- full/split CAGR, MDD, Sharpe, total return
- annual returns
- trade count, win rate, mean/median trade return, holding-time distribution
- transaction-cost stresses
- borrow-rate stresses
- liquid-proxy result
- T1-confirmed short control
- blocked-open/blocked-cover counts
- missing execution-row count
- positions still open at end
- maximum gross exposure / equity
- timing violations

## Interpretation gate
The reverse idea is considered economically interesting only if the exact mirror with 8% borrow and 1x costs has:
- full CAGR > 0
- both historical split CAGRs > 0
- full Sharpe > 0
- MDD > -45%
- 2x-cost CAGR > 0
- no timing violation
- no gross exposure above 1.0x

Even if it passes, it remains **POST-DIAGNOSTIC RESEARCH**, not certified deployable alpha, until historical lend availability is added and a genuinely untouched future sample exists.
