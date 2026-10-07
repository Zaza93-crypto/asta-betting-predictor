# Asta Football Predictor — Prediction Engine v1

## What this adds

- Reads the live fixture file created by `scripts/fetch_data.py`
- Produces Home/Draw/Away probabilities
- Applies HIGH/MEDIUM/LOW confidence labels
- Writes results to `data/predictions.json`
- Clearly marks the model as **NOT VALIDATED**
- Provides a starter workflow for automatic fixture fetching and prediction generation

## Important

This is a baseline model, not a proven 85% or 93% accurate system. The next development phase is historical back-testing using actual past results. Only then should accuracy, calibration, or profitability claims be made.

## Files

- `scripts/predict.py` — prediction engine
- `data/team_ratings.json` — starting team ratings; replace with validated ratings/statistics
- `data/predictions.json` — generated prediction output
- `.github/workflows/asta-predictor.yml` — example automation workflow

## Run locally

```bash
python scripts/predict.py
```

If your existing workflow already fetches fixtures, add the prediction step immediately after the fetch step:

```yaml
- name: Generate predictions
  run: python scripts/predict.py
```
