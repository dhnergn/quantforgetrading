# Portfolio Dashboard

Blueprint Section 56 names 14 components. All 14 are implemented.

The design constraint is not the components — it is that **a dashboard is the
one place in a quant system where a technically correct number is easiest to
mislead with.** An equity curve with no baseline, a Sharpe ratio with no sample
size, a VaR with no confidence level, a sector pie that silently omits the
instruments with no sector metadata: each is trivial to compute, trivial to
render, and each misleads in a specific direction.

So every component carries its own basis, and every component that cannot be
computed is marked unavailable **with a reason** — never a zero, never an empty
chart. Those two failure modes are what make dashboards dangerous: a zero
drawdown and a blank pie both read as "nothing to report", and both are
indistinguishable from a bug.

---

## The 14 components

| Component | Computes | Needs |
|---|---|---|
| `equity_curve` | Rebased to 1.0, optional benchmark | equity curve |
| `daily_pnl` | Per-period P&L in currency, cumulative path | equity curve |
| `unrealized_pnl` | Open-position P&L, realised, fees | positions |
| `realized_pnl` | Closed trades net of fees | trade list |
| `exposure` | Gross, net, leverage as fractions of equity | positions |
| `positions` | Entry, current price, P&L per position | positions |
| `sector_exposure` | Grouped by sector, **with coverage** | positions + metadata |
| `asset_class_exposure` | Grouped by asset class | positions + metadata |
| `correlations` | Heatmap over held instruments | per-instrument returns |
| `portfolio_volatility` | Rolling annualised, window stated | equity curve |
| `var` | VaR **and** ES, with confidence and n | equity curve |
| `drawdown` | Underwater curve, max, current | equity curve |
| `summary_statistics` | Sharpe, Sortino, Calmar + counts | equity curve |
| `beta` | Beta and alpha, overlap stated | equity curve + benchmark |

Every one of the 14 is always present in the response, including the ones that
could not be computed. A reader who sees 13 of 14 can tell something is
missing; a reader who sees 13 rows has no way to know a 14th was expected.

---

## The decisions that matter

**An equity curve with no benchmark says so.** The component is still shown, but
its caveat reads "no benchmark supplied, so this shows what happened, not
whether it beat anything." A strategy's own curve is not evidence it beat
something.

**Both curves are rebased to 1.0.** Two equity curves on different starting
capital cannot be overlaid, and a chart whose vertical axis runs from 10,000 to
10,000,000 is comparing capital, not returns.

**VaR never appears without expected shortfall.** VaR states the loss that is
*not expected* to be exceeded; it says nothing about how much worse the tail is.
Quoting VaR alone is the standard way to understate tail risk.

**A small sample is flagged, not left to notice.** A Sharpe ratio from 12 daily
returns is a number, not an estimate, so the caveat says so in capitals:
`FEWER THAN 30 OBSERVATIONS`.

**Sector exposure reports its coverage fraction.** If 40% of positions carry a
sector label, folding the other 60% into the groups that do have one would
misrepresent the book. The unclassified share is its own bar, and when *nothing*
carries a sector the component is unavailable — a currency pair has no sector,
and assigning one would be an invention.

**Gross and net are both shown.** A book that is 200% long and 200% short is
200% gross and 0% net, and those are very different risks.

**Beta is unavailable without a benchmark.** Beta is a statement about behaviour
relative to a reference. There is no default reference, so inventing one is the
only way to produce a number, and that number would mean nothing.

**Constant returns refuse rather than report infinity.** A constant non-zero
return gives an infinite Sharpe; a constant zero return gives an undefined one.
Neither is a risk-adjusted performance figure, and both are usually a data error.

**Correlation needs enough overlap.** Below 20 periods where every instrument
has data, the component is unavailable — a matrix computed from a handful of
observations is confident noise, and a heatmap makes that especially easy to
miss. Instruments with constant returns are dropped, since they have no
correlation with anything including themselves, and a NaN cell renders as zero
in most charting libraries.

**Undefined is `null` at the JSON boundary.** A bare `NaN` in a payload is
invalid JSON, and some clients parse it as a number.

---

## HTTP

All four routes require `CAN_VIEW_RISK`. That is deliberate: a dashboard
discloses positions and risk, which is a different permission from the ability
to run research. A `RESEARCHER` cannot read it.

```
GET  /v1/dashboard/components              the index of 14
GET  /v1/dashboard/report?portfolio_id=…  everything available
GET  /v1/dashboard/{component}?portfolio_id=…
POST /v1/dashboard/register               make a portfolio resident
```

### There is no default portfolio

Every read requires an explicit id. A default would hand any authorised caller
whichever portfolio happened to be loaded — no id to guess, and nothing to audit
as an attempt. A missing parameter is a `400` that says why.

A `job_id` is accepted as an alias, but its ownership is checked **before** the
portfolio lookup, so a job id cannot be used to discover whether another user's
portfolio is resident.

### Unavailable is 200, not 404

A `404` claims the component does not exist. It does exist; this portfolio
cannot produce it. Those are materially different things for a reader to
believe, so the response is `200` with `available: false` and a reason.

### `register` carries only the equity curve

So it sets `positions` and `trades` to `None`, and the affected components
report themselves unavailable. This preserves a distinction that a chart
cannot:

| State | Reported as |
|---|---|
| Positions supplied, book is empty | **flat** — available, zero positions |
| Positions never supplied | **unavailable** — with a reason |

Without the distinction the endpoint would tell an operator their account holds
nothing when in fact nothing was supplied.

`register` also refuses a non-positive equity value, a curve with fewer than 2
points, and out-of-order timestamps. It does not sort them: sorting would
reorder the series the caller actually measured.

### `confidence` is validated, not clamped

A VaR reported at a level the caller did not ask for is worse than a rejected
request, so `confidence=1.5` is a `400` rather than a silent clamp to 0.99.

---

## The page

`/static/dashboard.html`, served unauthenticated. That is deliberate: the risk
disclosure has to be readable before anyone holds a token, and the page contains
no portfolio data — every number arrives from a token-authenticated `/v1` call.
Hiding the page would only hide the disclosure.

Design choices that follow from the same principle as the API:

- **The disclosure sits above the data, not in a footer.** A reader who
  screenshots one card still carries the warning.
- **Every card shows `n=` and the period** in a fixed line under its title.
- **Unavailable cards are visibly distinct** — dashed border, amber flag — and
  carry their reason in full.
- **`undefined` renders as the words "not defined" or "not finite"**, never as a
  blank cell and never as `0`. A blank cell reads as "no data"; `0` reads as a
  measured zero. A real `0` still renders as `0`.
- **Charts are hand-rolled SVG.** No chart library, so the page has no
  third-party supply chain and works offline once the service worker caches it.
  Each series is drawn against its own labelled scale; an unlabelled line chart
  is decoration.
- **The correlation heatmap diverges** — green for positive, red for negative,
  saturating at ±1, mid-grey at 0 — so co-movement and counter-movement are
  distinguishable and neither washes out.
- **`UNCLASSIFIED` exposure is grey**, so it does not read as a sector.
- **A degenerate series declines rather than drawing.** One point, or all-NaN,
  produces a sentence saying so instead of a chart with a divide-by-zero.
- **No inline `onclick`.** Inline handlers would need an `unsafe-inline` CSP
  carve-out, which weakens the policy for the whole page.

### What is verified, and what is not

| Claim | How |
|---|---|
| Page and assets are served | `test_dashboard_page.py::TestServed` |
| The disclosure is above the data region | `TestDisclosurePlacement` |
| No third-party script or style | `TestNoThirdPartyDependencies` |
| `undefined` never renders as `0` | `tests/js/dashboard_render_check.mjs` under node |
| One card per component | same |
| Unavailable components render their reason | same |
| Charts survive flat / single-point / all-NaN input | same |
| Assets survive packaging into a wheel | `test_packaging_assets.py` |

**Not verified:** visual layout, colour contrast, and whether the charts look
correct. Those need a browser, and the browser harness was not connected while
this was written. They are untested, not passed.

The node render check is skipped — explicitly, with a reported skip — when node
is absent, because a check that silently does not run is indistinguishable from
one that silently passes.

---

## Reading the output

Nothing here is a forecast. The figures describe a tested period under stated
assumptions. See [RISK_DISCLOSURE.md](RISK_DISCLOSURE.md).

Two habits worth forming:

1. **Read `n=` first.** A ratio from 30 observations and the same ratio from
   3,000 are printed the same way; the count is what tells them apart.
2. **Check the caveat.** Most components here carry a sentence about what their
   number does not tell you. That sentence is the point of the component.
