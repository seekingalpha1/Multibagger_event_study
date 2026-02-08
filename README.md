# Multibagger Event Study

Event study analyzing **Multibagger events in the S&P 500**. A "Multibagger" is a stock that achieves 2x, 3x, 4x, ... 10x returns relative to a realistic entry price.

**Core question: How long should you hold your winners?**

The study tracks post-event performance after stocks cross these multiple thresholds, measuring forward returns, drawdowns, probabilities of reaching the next multiple, and more.

## Key Design Decisions

- **Survivorship-bias free**: Uses all historical S&P 500 members (including delisted companies), not just current constituents
- **Realistic entry price**: Rolling 5th percentile over 252 trading days (1 year) as the baseline — no cherry-picking of entry points
- **Cooldown period**: 90-day minimum gap between events for the same stock to avoid clustering
- **Multiple time horizons**: 1Y, 2Y, 3Y, 5Y, 10Y forward-looking periods to avoid cherry-picking of holding periods

## Project Structure

```
Multibagger_event_study/
├── download_sp500_data.py      # Phase 1: Data acquisition & cleaning
├── analyze_winner_stocks.py    # Phase 2: Event study analysis
├── summary_template.html       # HTML template for interactive results
└── README.md
```

## Requirements

```
pandas
numpy
yfinance
tqdm
openpyxl
```

## Workflow

### Phase 1: Download Data

```bash
# Initial download (all historical S&P 500 members, back to 1970)
python download_sp500_data.py download

# Test with limited tickers first
python download_sp500_data.py download --limit 10

# Custom start date
python download_sp500_data.py download --start-date 1980-01-01

# Disable automatic data cleaning
python download_sp500_data.py download --no-cleaning

# Incremental update (only new data since last download)
python download_sp500_data.py update

# Full refresh (re-download everything to capture Yahoo Finance corrections)
python download_sp500_data.py refresh

# Load and inspect existing data
python download_sp500_data.py load
```

**Output files:**
| File | Description |
|------|-------------|
| `sp500_historical_prices.pkl` | Main data file (OHLCV + Ticker + in_sp500 flag) |
| `sp500_metadata.pkl` | Metadata (last update, row count, date range, ticker list) |
| `all_tickers.txt` | List of all unique S&P 500 tickers ever |
| `failed_tickers.csv` | Tickers that failed to download |
| `data_cleaning_report.csv` | Summary of data quality issues found |

### Phase 2: Run Analysis

```bash
# Run with default parameters (output to analysis_results/)
python analyze_winner_stocks.py

# Run with a named output for robustness testing
python analyze_winner_stocks.py --run-name baseline

# Override analysis parameters for robustness tests
python analyze_winner_stocks.py --run-name cooldown_60 --cooldown-days 60
python analyze_winner_stocks.py --run-name percentile_10 --percentile 10
python analyze_winner_stocks.py --run-name window_504 --rolling-window 504

# Combine parameter overrides
python analyze_winner_stocks.py --run-name aggressive --cooldown-days 30 --percentile 3

# List all saved runs and their configurations
python analyze_winner_stocks.py --list-runs
```

**Output files (per run):**
| File | Description |
|------|-------------|
| `summary_statistics.csv` | Main results table (one row per multiple, all KPIs) |
| `summary_statistics.html` | Interactive HTML table with period/metric filtering |
| `detailed_results.csv` | One row per event with forward returns for all periods |
| `detected_events.csv` | All detected crossing events |
| `next_multiple_probabilities.csv` | Probability of reaching higher multiples |
| `run_config.json` | Configuration parameters used for this run |
| `ticker_analysis_*.xlsx` | Per-ticker detailed Excel workbook (optional) |

## Methodology

### Entry Price Definition

The entry price is defined as the **rolling 5th percentile of closing prices over 252 trading days** (approximately 1 year). This serves as a conservative, realistic estimate of when an investor might have entered the position.

### Event Detection

An event is triggered when the stock's closing price crosses a multiple threshold (2x, 3x, ..., 10x) relative to its rolling entry price **from below**. A 90-day cooldown between events for the same stock/multiple prevents double-counting clustered crossings.

### KPIs Calculated

For each (Multiple, Follow-up Period) combination:

| Category | Metrics |
|----------|---------|
| **Returns** | 25th, 50th, 75th percentile, mean total return |
| **CAGR** | Implied compound annual growth rates |
| **Risk** | Maximum Drawdown (mean, median, worst), annualized volatility |
| **Win Rate** | Percentage of events with positive forward return |
| **Sharpe Ratio** | Simplified mean return / mean volatility |
| **Next Multiple** | Probability of reaching next multiple, avg days to reach |
| **Higher Multiples** | Probability of reaching each higher multiple (e.g., from 3x: prob of 4x, 5x, ..., 10x) |
| **MDD Until Next** | Drawdown endured on the path to the next multiple (25th/50th/75th percentile) |
| **Return If Not Reached** | Return distribution for events that did NOT reach the next multiple |

### Data Cleaning

The download script automatically handles:
- Zero or negative prices
- Zero volume days
- Extreme outliers (>500% daily change)
- OHLC inconsistencies (High < Low, Close outside High/Low range)
- Duplicate dates

## Output Versioning for Robustness Tests

When running the analysis with `--run-name`, results are saved to a subdirectory under `analysis_results/`:

```
analysis_results/
├── baseline/
│   ├── summary_statistics.csv
│   ├── summary_statistics.html
│   ├── detailed_results.csv
│   ├── detected_events.csv
│   ├── next_multiple_probabilities.csv
│   └── run_config.json
├── cooldown_60/
│   ├── ...
│   └── run_config.json
└── percentile_10/
    ├── ...
    └── run_config.json
```

Each `run_config.json` stores the exact parameters used, making it easy to compare runs and reproduce results.

Use `python analyze_winner_stocks.py --list-runs` to see all saved runs with their parameters at a glance.

## Data Sources

- **Historical S&P 500 components**: [fja05680/sp500](https://github.com/fja05680/sp500) on GitHub
- **Price data**: Yahoo Finance via the `yfinance` library (back to 1970)
