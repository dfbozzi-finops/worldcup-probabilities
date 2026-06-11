# World Cup 2026 — Hybrid Statistical Arbitrage System

A production-grade quantitative platform designed to detect statistical arbitrage opportunities in Polymarket's 2026 FIFA World Cup winner market.

The system combines classical statistical methods (Dixon-Coles) with modern Machine Learning (CatBoost) and Bayesian MCMC sampling, enriched by real-time macro-economic data, to identify mispriced teams.

## Architecture

The system operates across a 6-step pipeline:

1. **Data Ingestion**: Pulls 20+ years of historical international football matches from Kaggle.
2. **Statistical Modelling (Dixon-Coles)**: Fits a bivariate Poisson model (with time-decay and tau-correction) over 15,000+ matches to extract team-level Attack and Defence parameters.
3. **Macro Automation**: Ingests GDP, Population, and Climate data from the World Bank API with aggressive local caching.
4. **Machine Learning Ensemble**:
   - **CatBoost**: Gradient-boosted trees operating on FIFA rankings, DC parameters, and Macro features.
   - **Custom MCMC**: Metropolis-Hastings Bayesian sampler (built natively in NumPy/SciPy) providing full posterior probabilities and uncertainty bounds.
5. **Precomputed Monte Carlo Simulator**: Simulates the 104-match tournament 10,000 times using an O(1) ML Look-up Table to generate true model consensus probabilities in < 3 seconds.
6. **Arbitrage Engine & Execution Preview**:
   - Pulls live order book data from Polymarket's Gamma API.
   - Applies strict Viability Filters (rejects Odds > 100 or Consensus < 2%).
   - Calculates Kelly-criterion position sizing.
   - Generates a "Dry-Run" Trade Execution Preview table for safe, advisory-only manual trading.

## Backtesting & Validation

The system includes a rigorous historical backtesting harness (`src/backtester.py`).
When tested purely out-of-sample against the **2022 World Cup**, the model:
- Achieved a highly-calibrated **Brier Score of 0.0248**.
- Accurately ranked the eventual winner (Argentina) as the **#2 most likely champion** pre-tournament.

## Usage

**Prerequisites:**
- Python >= 3.13
- `uv` package manager

**Run Pipeline (Advisory/Dry-Run Mode):**
```bash
uv run main.py --once --dry-run
```

**Run Historical Backtest (2022):**
```bash
uv run python -m src.cli_backtest --year 2022
```

## Security & Execution
This system is designed in **Advisory-Only Mode**. It does not handle L1/L2 wallet signing or private keys. The Trade Executor outputs a preview table (Token ID, Size, Limit Price) to be executed manually on the Polymarket UI.

---
*Built for Quantitative Precision. Deployed for the 2026 World Cup.*
