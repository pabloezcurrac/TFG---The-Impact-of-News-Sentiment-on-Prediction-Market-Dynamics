# Polymarket Sentiment Analysis

Does news sentiment predict returns on binary prediction markets? This project builds an end-to-end pipeline that combines Polymarket price data with daily news sentiment to test that hypothesis via OLS regression.

Built as part of a Bachelor's thesis (TFG), IE, 2026.

---

## Research question

Does lagged news sentiment predict daily returns (`ret_yes`) on Polymarket binary markets?

---

## Pipeline overview

```
Polymarket URLs        GNews API
      |                    |
Gamma API + Goldsky    Keywords → Dedup
      |                    |
 Daily prices         news_count + sentiment
      |                    |
      └──────── Merge ──────┘
                    |
            Feature engineering
          (log, z-score, lags t-1)
                    |
            OLS per-slug + pooled
                  (HC1)
```

---

## Repository structure

```
polymarket-sentiment/
│
├── about me/
│
├── tfg_data_polymarket_and_gnews/
│
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/polymarket-sentiment.git
cd polymarket-sentiment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API keys
export GNEWS_API_KEY="your_key_here"

# 4. Run the pipeline
python src/fetch_markets.py
python src/fetch_news.py
python src/build_panel.py
python src/features.py
python src/models.py
```

---

## Data

- **Polymarket**: 5 binary markets retrieved via the Gamma API and Goldsky GraphQL endpoint. Variables: `p_yes`, `ret_yes`, `vol7`, `spread`.
- **GNews**: Up to 100 articles per day per market, retrieved using 3–14 keyword queries per slug. Deduplicated by URL. Sentiment scored as:

```
score_i = P(positive)·1 + P(neutral)·0 + P(negative)·(-1)
```

> Raw data files are not committed to this repository. Add your own API keys and re-run the pipeline to reproduce the dataset.

---

## Methods

Features are constructed as follows:

- `log_news`: log-transformed daily article count
- `z_sentiment`: z-score normalised sentiment score
- `lag_*`: one-day lag (t-1) applied to all predictors to avoid look-ahead bias

Two OLS models are estimated:

- **Per-slug**: separate regression for each of the 5 markets
- **Pooled**: single regression across all markets with slug fixed effects

Standard errors are heteroskedasticity-robust (HC1) in both cases.

---

## Limitations

- Only 5 markets: results may not generalise.
- OLS assumes linearity; non-linear effects are not captured.
- Sentiment model (GNews) is a black box — no access to raw classifier probabilities.
- Short time window limits statistical power.

---

## Requirements

```
pandas
numpy
statsmodels
requests
python-dotenv
jupyter
```

---

## License

MIT License. See `LICENSE` for details.

---

## Citation

Ezcurra Callejas, P. (2026). The impact of news sentiment on prediction market dynamics [Bachelor's thesis, IE University].


---

## Author

Pablo Ezcurra · pezcurra.ieu2021@student.ie.edu · Universidad Instituto de Empresa (IE), 2026
