# Asta Football Predictor — Phase 3 Historical Backtesting

This phase evaluates the baseline prediction model against completed historical matches.

## Anti-leakage design

Each historical match is predicted **before** its result is used to update team ratings. This prevents the model from learning the result it is supposed to predict.

## Files

- `scripts/fetch_history.py` — retrieves historical results from football-data.org
- `scripts/backtest.py` — sequential Elo-style backtest
- `.github/workflows/backtest.yml` — manual + weekly backtest workflow
- `data/historical_matches.json` — fetched historical results
- `data/backtest_report.json` — metrics and sample predictions

## Metrics

The report calculates:

- 1X2 accuracy
- Brier score
- Multiclass log loss
- Accuracy by competition
- Accuracy by confidence band

## Important

Back-test results are evidence about this historical sample only. They are not a guarantee of future prediction accuracy or betting profitability. Do not claim 85%/93% accuracy unless a sufficiently large, out-of-sample evaluation supports it.
