# Information Field Theory – Wiener Filter Engine for ETFs

Applies information field theory (Enßlin 2013) to model ETF returns as a correlated random field on the graph of ETFs. Macro variables (VIX, DXY, yields) enter as external source fields. The Wiener filter computes the posterior uncertainty, which serves as a signal for regime conflict and potential alpha.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Macro variables: configurable list (VIX, DXY, T10Y2Y, yields)
- Graph construction via k‑NN from correlation distance
- Wiener filter solves linear system (prior precision + noise identity)
- Score = posterior variance per ETF
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-information-field-theory-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High posterior variance → macro conditions conflict with graph‑implied ETF correlations → possible regime shift or alpha opportunity.
- Low variance → market is well‑explained by macro and graph structure.

## Requirements

See `requirements.txt`.
