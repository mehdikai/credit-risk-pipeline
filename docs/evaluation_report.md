# Model Evaluation — Week 7

Evaluated on the held-out 20% test split (7,292 clients, same `random_state=42`
split as training) of `gold_features`, using the Logistic Regression scorecard
model from Week 6.

## Discrimination metrics

| Metric | Value |
|---|---|
| AUC | 0.657 |
| Gini | 0.315 |
| KS | 0.273 (at predicted PD = 0.510) |

These are modest, not strong — consistent with the Week 4 finding that only
6/14 original features cleared the IV usefulness threshold. The model has
real, statistically meaningful signal (AUC well above the 0.5 no-skill
baseline) but should not be presented as a high-precision discriminator.

## Confusion matrix at the KS-optimal threshold (PD = 0.510)

| | Predicted good | Predicted bad |
|---|---|---|
| **Actual good** | 4,406 | 2,763 |
| **Actual bad** | 42 | 81 |

At this threshold the model catches ~66% of actual defaults (81/123), but
at the cost of flagging ~39% of good clients as high-risk (2,763/7,169).
This threshold maximizes statistical separation (KS) but is too aggressive
for a real approval cutoff — hence the quantile-based bands below, which
are what would actually drive a business decision.

## Decision bands (quantile-based on predicted PD)

| Band | PD range | Count | Default rate |
|---|---|---|---|
| Approve | PD < 0.472 | 3,646 | 0.85% |
| Refer (manual review) | 0.472 <= PD < 0.633 | 2,917 | 2.37% |
| Decline | PD >= 0.633 | 729 | 3.16% |

Overall test-set default rate: 1.69% (matches the population base rate).
The Decline band concentrates default risk at **3.7x** the Approve band's
rate (3.16% vs 0.85%) — a real, useable lift for portfolio triage, even
though the model isn't strong enough to support a single hard accept/reject
cutoff. Bands were chosen at the 50th/90th PD percentiles rather than fixed
probability values, since the ~1.69% base rate makes fixed cutoffs
uninformative without reference to the score distribution.

## Artifacts

- `docs/eval_plots/roc_curve.png`
- `docs/eval_plots/ks_curve.png`
- `docs/eval_plots/score_distribution.png`
- `docs/evaluation_report.json` (machine-readable version of this report)
- Logged to MLflow, experiment `credit_risk_scorecard`, run `evaluation`
