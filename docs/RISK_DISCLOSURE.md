# Risk Disclosure

**Read this before using any output of this platform.**

## 1. No strategy, indicator, or model predicts markets

Nothing in QuantForge — no moving average, no momentum signal, no machine
learning model, no regime classifier — has the demonstrated ability to predict
future price movements. These are deterministic transforms of past
observations. A rule that identified profitable configurations historically
describes what happened, not what will happen.

The platform's own outputs say this. Every performance report, every metrics
table, and every strategy description carries a disclaimer. They are not
boilerplate; they are the most important sentences in the output.

## 2. Backtest results are evidence about the past

A profitable backtest establishes one narrow thing: that a specific rule,
applied to a specific dataset, under specific cost and execution assumptions,
produced a profit over a specific historical window.

It does **not** establish:

- that the rule will work on data outside that window,
- that the rule will work after the period it was fitted to,
- that the cost and fill assumptions are achievable,
- that the market structure has not changed since,
- that the result exceeds what selection bias alone would produce.

The platform provides deflated Sharpe ratios, walk-forward analysis, Monte
Carlo simulation, and overfitting diagnostics precisely because a raw Sharpe
ratio is not sufficient evidence. Read [OVERFITTING.md](OVERFITTING.md).

## 3. Most published strategies fail out of sample

The academic literature is unusually consistent on this point. Strategies
documented in papers, blogs, and vendor research have historically failed to
replicate at the reported magnitude. Some of that failure is data mining
across configurations; some is the costs of implementation exceeding the
modelled edge.

The platform's synthetic data generator deliberately injects **no exploitable
structure**. If a strategy profits on it, that is a bug, not a discovery.

## 4. The specific risks of this platform

**Model risk.** A model fitted to a regime that does not recur behaves
arbitrarily. The regime engine, the ML models, and the strategy rules are all
fitted to history. Non-stationarity is not a risk to be managed; it is a
certainty.

**Execution risk.** The fill model is an approximation. Real execution
involves queue position, order-book depth, adverse selection, partial fills,
and counterparty behaviour. A strategy whose edge is 20 basis points is
entirely an execution-quality strategy, and the platform cannot tell you
whether you have that edge.

**Data risk.** Historical vendor data contains errors: survivorship bias,
look-ahead in restated fundamentals, revision handling, timezone errors,
corporate-action mistakes. The platform prevents *structural* leakage and
validates data quality, but it cannot correct a vendor's errors.

**Operational risk.** Systems fail at 3am. Connection loss, partial fills,
clock skew, database unavailability, and duplicate messages are normal
events, not exceptions. The kill switch and reconciliation machinery exist
because these will happen.

**Overfitting risk.** Every design decision informed by looking at results is
a degree of freedom. The deflated Sharpe adjustment is only correct if
`n_trials` counts *every* configuration evaluated, including the ones you
tried and discarded. Undercounting it produces a falsely impressive statistic.

**Counterparty and custody risk.** The platform does not custody assets, verify
broker solvency, or assess regulatory compliance.

## 5. The limits of the platform's own safeguards

The risk engine enforces position, sector, gross, net, leverage, drawdown, and
daily-loss limits. It cannot:

- detect a market that has stopped behaving like any historical period,
- distinguish a signal from a coincidence,
- prevent a gap larger than any modelled stop,
- protect against a broker failure or a custody failure,
- guarantee that a limit is calibrated correctly.

The live-trading gate is a **procedural** control, not a technical one. It
verifies that a human performed a checklist. It cannot verify that the human
was correct.

## 6. Position sizing does not prevent ruin

Fixed-fractional sizing bounds the loss from a single position. It does not
prevent ruin across a sequence of correlated losses, and it offers no
protection against a gap through a stop. Kelly sizing is bounded to a fraction
of full Kelly for a reason: the full fraction maximises both expected growth
and drawdown, and the input error in the win-probability estimate is much
larger than the arithmetic suggests.

No position-sizing rule makes a strategy with no edge safe. It only determines
how quickly an unprofitable strategy loses capital.

## 7. Model confidence is not certainty

Classifier outputs are model scores. A probability of 0.7 means the model
assigns 70% to a class, not that the outcome occurs 70% of the time. Models
are miscalibrated out of sample, particularly after the regime they were
trained on has passed. The platform provides Brier scores, reliability
diagrams, and calibration-drift monitoring for this reason.

## 8. Sentiment and alternative data are weak signals

Social sentiment is measured on platforms with selection bias, bot activity,
and changing user bases. News sentiment reflects the tone of coverage, which
responds to events rather than predicting them. Analyst revisions are widely
followed and correspondingly arbitraged. Treat these as features to be
validated, not as information to be trusted.

## 9. Risk metrics have assumptions that can fail

- **VaR** under a normal assumption understates tail risk for real return
  distributions, which have excess kurtosis. The platform reports the
  assumption alongside the number and offers a Student-t alternative.
- **Expected shortfall** depends on a tail that, by definition, has few
  observations behind it.
- **Correlation and covariance** estimates are unstable in high dimensions and
  degrade in crises exactly when they matter. The platform applies shrinkage
  and reports the condition number.
- **Square-root-of-time scaling** assumes independent, identically
  distributed, finite-variance returns. Real returns have fat tails and
  volatility clustering, both of which make it optimistic.
- **Beta** is an estimated regression coefficient, not a stable risk measure.

## 10. Simulation is not trading

Paper trading and backtesting use a simulated execution adapter. The
simulation cannot model:

- the difference between your intent and the market's response,
- information leakage you have and the model does not,
- your own reaction to a drawdown,
- operational failures during a volatile session,
- the market's reaction to a strategy becoming widely known.

Paper trading validates that the *plumbing* works. It does not validate that
the *strategy* works.

## 11. Seek independent verification

Before risking capital:

1. Have the code reviewed by someone who did not write it.
2. Replicate the results independently, from raw data.
3. Verify the data provenance independently.
4. Test the execution path in a broker sandbox with real order flow.
5. Start with capital you can afford to lose entirely.
6. Monitor live behaviour against expected behaviour, and be prepared to stop.

## 12. Summary

QuantForge is a research tool. It is built to make a wrong answer harder to
produce and easier to detect — not to produce a profitable strategy. A
platform that produced profitable strategies reliably would either have a
genuine edge, which is rare and fragile, or would be making a statistical
error that its own diagnostics were built to catch.

If a result from this platform looks too good, that is the signal to be
sceptical, not satisfied.
