# Concepts

This document describes the statistical methodology behind `alpha-lens`.
It is aimed at quants who want to know *what* the diagnostics compute,
*why* they were chosen, and *how* to read the results.

The numbers `alpha-lens` produces are only as trustworthy as the
choices behind them. We document the choices.

## Table of contents
- [Returns conventions](#returns-conventions)
- [Sharpe and friends](#sharpe-and-friends)
- [Information Coefficient](#information-coefficient)
- [Regime detection](#regime-detection)
- [Drawdown attribution](#drawdown-attribution)
- [Factor attribution](#factor-attribution)
- [Deflated Sharpe Ratio](#deflated-sharpe-ratio)
- [Probability of Backtest Overfitting (PBO)](#probability-of-backtest-overfitting-pbo)
- [Minimum Backtest Length](#minimum-backtest-length)
- [Walk-forward and OOS validation](#walk-forward-and-oos-validation)
- [Bootstrap Sharpe CI](#bootstrap-sharpe-ci)
- [Cost sensitivity and break-even](#cost-sensitivity-and-break-even)
- [Production Readiness Score](#production-readiness-score)

---

## Returns conventions

All returns are **decimal**: `0.01` = 1%. The library warns if it
detects percent-encoded input (`5.0` meaning 5%) because the rest of the
pipeline silently produces garbage if you get this wrong.

All annualization uses **252 trading days**. If your data is at a
different frequency, pass `periods_per_year` explicitly where supported.

NaN handling: `alpha-lens` does not interpolate. Missing returns are
**forward-filled briefly** and then **dropped**, on the principle that
inventing returns is worse than losing observations.

## Sharpe and friends

Standard formulas. The implementation uses sample standard deviation
(`ddof=1`) and an absolute tolerance of `1e-12` for the zero-std check
to avoid blowing up on near-constant series (a real failure mode we hit
in testing).

The risk-free rate is subtracted *before* annualizing, not after.

## Information Coefficient

Rank IC, not Pearson IC. We compute the Spearman correlation between
the factor at time `t` and returns at time `t + h`, where `h` is the
forward horizon. **The forward shift is critical**: a common bug is to
correlate the factor with contemporaneous returns, which leaks
information from the future.

`alpha-lens` always shifts forward returns explicitly. We test this
with a factor that exactly equals today's return: under the correct
implementation, its IC against the next day's return is near zero (not
1.0).

## Regime detection

Two methods are supported.

**Rule-based** is the default. It uses three signals:
- **Trend**: short SMA vs long SMA of returns.
- **Volatility**: rolling realized vol vs its historical 75th percentile.
- **Drawdown**: trailing cumulative drawdown from a rolling 252-day peak.

The combination rules:
- High vol → `high_vol`.
- In drawdown &gt; 10% and trending down → `bear`.
- Trending up and not in high vol → `bull`.
- Else → `sideways`.

The rule-based detector is deliberately simple and explainable. It is
not meant to be optimal &mdash; it is meant to give the user a label
that they can intuitively check.

**HMM-based** (optional, requires `hmmlearn`) fits a 2-, 3-, or 4-state
Gaussian HMM to returns, picks the state count by BIC, and maps states
to labels by their mean-return ranking. HMM regimes are mathematically
elegant but harder to interpret. Use rule-based when you want to argue
with a PM about it.

## Drawdown attribution

Each drawdown event is tagged with a `dominant_regime`. The naive
approach is to take the mode of the regime labels during the peak-to-trough
window. This is **wrong**: if the peak occurred in a long sideways
period followed by a sharp bear-market crash, the mode is still
"sideways" even though the bear regime caused the actual damage.

We instead weight regime labels by the magnitude of *negative* returns
during the window: days that drove the drawdown count more than flat
days. This is a small detail, but it changes the regime tags in
testing from "everything is bull" (the dominant background regime) to
correctly identifying bear and high-vol events.

## Factor attribution

Standard OLS regression of *excess strategy returns* on *factor returns*.
We report:

- **β** for each factor and its t-statistic.
- **Annualized α** &mdash; the intercept × 252.
- **R²** &mdash; how much of the strategy's variance the factors explain.
- **Uniqueness score** = 1 − R². A pure alpha strategy scores near 1;
  a 1.5x-market strategy scores near 0.

If only a benchmark is supplied, we run a CAPM-style single-factor
regression. If a `factors` DataFrame is supplied, we run multi-factor.

Rolling-window attribution is also computed (closed-form OLS per
window, vectorized) so the report can show how factor loadings drift.

## Deflated Sharpe Ratio

The Sharpe ratio of a backtest is biased upward when the researcher
tries many strategies and reports only the best. Bailey & López de
Prado (2014) derived a correction.

The DSR computes the probability that the *true* (population) Sharpe
is greater than zero, given:
- The observed Sharpe.
- The number of trials the researcher tried (`n_trials_assumed`).
- The skewness and kurtosis of the returns (non-normality penalty).
- The sample size.

**The `n_trials_assumed` parameter matters a lot.** A 1.5 Sharpe over
3 years is statistically convincing if you tried one thing; it is
unconvincing if you tried 1000. The default in `alpha-lens` is 100,
which is a guess. **You should override it with what you actually
did.** Counting parameters tried, feature combinations explored, and
universes screened.

The output is a z-score and a p-value. We use the p-value (the
probability that the true Sharpe is ≤ 0) for the scoring layer.

## Probability of Backtest Overfitting (PBO)

PBO comes from Bailey et al. (2017). The idea: given a *set of strategy
variants*, partition the time series into N equal chunks, then for every
combination of half the chunks as in-sample and half as out-of-sample,
ask: **does the in-sample-best strategy beat the OOS median?**

PBO is the fraction of partitions where it does *not*. A real signal
ranks high in OOS as well; pure noise has PBO near 0.5 (random).

`alpha-lens` implements the combinatorially-symmetric cross-validation
(CSCV) variant with `n_partitions=16` by default. The user must supply
the strategy variants for PBO to be computed. If only the headline
strategy is supplied, PBO is reported as `None` and the scoring
component is dropped (and reweighted).

### Implementation note

The "rank above median" check uses percentile rank, not raw rank. With
20 strategies, a strategy with rank 11/20 has percentile rank 0.55,
which is above median. Using the integer rank with `> n/2` introduces
a small downward bias on PBO; using percentile rank does not. This is
a subtle but real bug we found in our first pass.

## Minimum Backtest Length

Also from Bailey & López de Prado (2014). For a strategy with observed
Sharpe `SR` discovered after `n_trials`, what's the minimum number of
years of data you'd need for the result to be statistically meaningful?

The formula:

```
MinBTL = (1 + (1 - γ)·E[max_n]² / SR²) / SR²
```

where `γ` is the Euler-Mascheroni constant and `E[max_n]` is the
expected maximum of n iid standard normal variables.

We flag the strategy as failing this test if the actual length is less
than `MinBTL`. This is one of the most under-used statistics in
quant; many published backtests fail it.

## Walk-forward and OOS validation

We provide two related diagnostics:

**70/30 chronological split.** Compute the Sharpe on the first 70% of
the data ("in-sample") and the last 30% ("out-of-sample"). The
**degradation ratio** (OOS / IS) should be close to 1; below 0.5 is a
red flag; negative means the strategy went from working to broken.

**5-window walk-forward.** Split the data into 5 equal chronological
chunks and compute the Sharpe in each. We do *not* re-train the
strategy in each window (we don't have access to its parameters); we
just measure performance. A strategy that worked in 4 of 5 windows is
more robust than one that hit a single huge winner.

The combined score (40% OOS level, 40% degradation, 20% walk-forward
consistency) rewards strategies that hold up across the timeline.

## Bootstrap Sharpe CI

We use **IID bootstrap** with 1000 resamples by default. For daily
returns of most strategies, autocorrelation is small enough that the
simple bootstrap is fine. For strategies with strong autocorrelation
(some HF or option strategies), a **stationary bootstrap** would be
more rigorous &mdash; this is on the roadmap.

We report the 95% CI (`(lower, upper)`) and the bootstrap standard
deviation. A CI that straddles zero is a serious warning sign.

## Cost sensitivity and break-even

If positions are supplied, we compute annualized two-way turnover from
position changes (`abs(positions_t - positions_{t-1}).sum(axis=1)`).
If not, we conservatively assume 1.0 daily turnover &mdash; one full
portfolio turn per day &mdash; which intentionally pessimistic for
strategies whose true turnover we don't know.

For each cost level in `[0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]` basis
points, we compute the resulting Sharpe after subtracting `cost × turnover`
each day. We then **bisect** for the cost level at which Sharpe = 0,
giving the **break-even cost**.

A 5bps break-even strategy will not survive retail execution.
A 50bps break-even strategy has institutional margin.

## Production Readiness Score

The headline number. It's a weighted average of seven sub-scores, each
0-100:

| Component | Weight | Captures |
|-----------|--------|----------|
| Overfitting Risk | 25% | DSR p-value, PBO, MinBTL |
| Out-of-Sample Validation | 20% | OOS level, degradation, walk-forward |
| Robustness | 15% | Bootstrap CI position, subsample stability |
| Regime Robustness | 10% | Cross-regime weighted Sharpe |
| Alpha Decay | 10% | IC magnitude, stability, half-life |
| Cost Sensitivity | 10% | Break-even bps |
| Performance | 10% | Sharpe and Calmar |

The weights reflect what *actually predicts production failure*:
overfitting and OOS degradation are the leading causes, so they
dominate. Raw performance is intentionally the lowest weight &mdash;
anyone can produce a high-Sharpe backtest with enough effort.

If a component cannot be computed (e.g. PBO requires strategy
variants, decay requires factor values), the score is re-normalized
across the remaining components rather than being zero-imputed. This
keeps the verdict honest: a 6-component score is not penalized for
missing the 7th.

### Verdict thresholds

| Score | Verdict | Action |
|-------|---------|--------|
| ≥ 80 | READY | Deploy with normal monitoring |
| 60-80 | CONDITIONAL | Paper-trade and address top risks |
| 40-60 | NOT_READY | Iterate; consider expanding the validation set |
| < 40 | REJECT | Likely an artifact of selection bias |

These thresholds are calibrated against a small benchmark of real and
synthetic strategies. Like the weights, they are opinionated. You can
override them via `ScoringConfig`.

### Why opinionated defaults?

Configurable libraries are nice in theory and useless in practice. A
PM who reads "the score depends on weights you can tune" understands
that they cannot trust the score. A PM who reads "the weights are
chosen to penalize overfitting heavily because that's what kills
backtests in production" understands what the score means and can
disagree with it.

The defaults are documented, the weights are configurable, but
out-of-the-box `alpha-lens` makes a strong, defensible claim about
what production readiness means.
