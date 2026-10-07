# Asta V3 FINAL

This is the final planned model version.

## Method
V3 uses an expanding-window, walk-forward multiclass logistic regression.

For every historical match:
1. Build features using only matches completed before that match.
2. Train on earlier observations only.
3. Predict the current match.
4. Score the prediction.
5. Only then add the current match to the historical state.

Features include:
- Elo difference
- home advantage
- rolling 5/10-match points
- rolling 5/10-match goals for/against and goal difference
- home/away rolling performance
- team experience count

Regularization and class balancing are used to reduce overfitting.

## Existing benchmarks
V1: 48.74% accuracy on 1,752 matches.
V2: 48.29% accuracy on 1,752 matches.

V3 must be judged by its own chronological out-of-sample report. No target accuracy is guaranteed.

## Final-version rule
Do not create V4 merely to chase a higher historical score. Future changes should be limited to:
- fixing bugs,
- correcting data problems,
- improving data quality,
- or adding genuinely new information with a separately documented validation test.

Historical backtest performance does not guarantee future accuracy or profitability.
