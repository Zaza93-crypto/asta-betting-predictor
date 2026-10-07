# Asta Betting Predictor V2

V2 adds a live-data architecture using API-Football and GitHub Actions.

## Important
The site is an analytics/prediction tool. Predictions are estimates and are not guaranteed. Do not publish unsupported accuracy claims.

## Setup
1. Create an API-Football account and obtain an API key.
2. In GitHub: **Settings → Secrets and variables → Actions → New repository secret**.
3. Name the secret exactly `API_FOOTBALL_KEY`.
4. Paste the API key as the value.
5. Push this project to the `main` branch.
6. Run **Actions → Update football data → Run workflow** once to test.
7. The workflow also runs every 6 hours and commits updated `data/fixtures.json`.
8. GitHub Pages serves the updated JSON to the website.

The free API-Football plan currently provides 100 requests/day. This workflow intentionally uses a small set of competitions and only requests predictions for returned fixtures.

## Next development
Add a proper local model/backtest, historical prediction storage, verified result settlement, and accuracy metrics. Do not treat provider predictions as proof of our own model's performance.
