# Multi-Alpha V3 Post-Diagnostic Preregistration — 2026-09-03

This is explicitly post-diagnostic research after V2 showed: (i) unconditional rev5/mom20/breakout/acceleration are not viable short alpha under the hard A-share executor; (ii) the single best train-Calmar long candidate over-selected pure quality; (iii) medium GEff-F10QV10 rank-tilt/stagger remains the strong core. Nothing in V3 is clean OOS.

## Frozen evaluation discipline
- 2016-08-02..2021-12-31: selection only.
- 2022-01-01..2026-07-29: pseudo-OOS diagnostic only, never used to select candidate/weight within this round.
- Same signal-pure universe and hard A-share executor; next-session execution, board price-limit proxies, blocked trades, volume participation, 100-share lots, no leverage.
- Cost stress 2x and 4x after train-only selection.

## Short sleeve V3: conditioned pullback alpha
Rationale: V2 showed that unconditional losers keep losing and unconditional strength chasing is destroyed by A-share execution. V3 tests a narrower behavioral hypothesis: short-term pullbacks only inside an established intermediate uptrend, optionally conditioned on low risk/quiet volume.

Fixed families before V3 run:
1. `pullback60`: require ret60 > +5%; score 70% low ret5 + 30% high ret60.
2. `pullback120`: require ret120 > +5%; score 70% low ret5 + 30% high ret120.
3. `pullback_lowiv`: require ret60 > 0; score 60% low ret5 + 20% high ret60 + 20% low IVOL60.
4. `quiet_pullback`: require ret60 > 0; score 60% low ret5 + 20% high ret60 + 20% low relative volume(20/120).
5. `market_relative_pullback`: require ret60 > 0; score 70% low (ret5 - market_ret5) + 30% high ret60.

Grid per family:
- H in {10,20}
- N in {15,20}
- Entry10 / Keep30 fixed
- all phases equally ensembled for each candidate
- liquidity eligibility: top 70% by 20-day amount, same orientation as prior work

Train selection objective:
- require train CAGR > 0 and train MDD > -45% if any candidate passes;
- maximize train Calmar, then train Sharpe, then fewer trades.

## Long sleeve V3: family-diversified PIT valuation/quality
Rationale: V2's pure-quality train winner beat value by only a small train-Calmar margin but produced an unstable long sleeve. V3 reduces winner-selection risk by ensembling train-selected representatives from distinct long families.

For each of `value`, `quality`, `value_quality`, select that family's representative solely by V2 2016-2021 train Calmar then Sharpe. Re-run those representatives under identical executor.

Predefined long ensemble candidates:
- `L_all3_equal`: 1/3 value + 1/3 quality + 1/3 value_quality.
- `L_value_dominant_equal`: 1/2 value + 1/2 value_quality. Pure quality is excluded here because the medium sleeve already contains explicit quality information; this is an alpha-orthogonality design choice, not a pseudo-OOS selection.
- `L_value70_vq30`: 70% value + 30% value_quality.

Select long ensemble on 2016-2021 train Calmar then train Sharpe only.

## Frozen medium sleeve
Same as V2 and not retuned:
- GEff-F10QV10 = 80% technical + 10% CFO/assets + 10% QualityValue
- H60
- 75% N10 + 25% N5 rank tilt
- phases 0/4/8 equal
- Entry10 / Keep30

## V3 capital allocation candidates
Only if the selected short sleeve has positive train CAGR. Coarse fixed candidates:
- B1 = 10% short / 60% medium / 30% long
- B2 = 15% short / 55% medium / 30% long
- B3 = 15% short / 60% medium / 25% long
- B4 = 20% short / 50% medium / 30% long
- B5 = 10% short / 70% medium / 20% long

If no short candidate has positive train CAGR, V3 must report that no validated short alpha exists and must not force capital into it.

Allocation selection uses 2016-2021 train Calmar then Sharpe only.

## Required diagnostics / gates
- standalone S/M/L full/train/pseudo CAGR, MDD, Sharpe
- full/train/pseudo/down-day correlations and rolling 252d correlations
- remove-one-sleeve marginal contribution
- annual returns
- 2x/4x costs

Promote only if:
1. short selected candidate has positive train CAGR;
2. each sleeve pseudo CAGR > 0;
3. portfolio pseudo CAGR > 0;
4. portfolio train Calmar >= medium-alone train Calmar;
5. portfolio full Sharpe >= medium-alone full Sharpe - 0.03;
6. at least one non-medium sleeve correlation with medium <= 0.60;
7. removing short or long does not improve full Sharpe by >0.05;
8. 2x cost portfolio CAGR > 0.
