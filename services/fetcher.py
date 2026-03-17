"""
fetcher.py — yfinance OHLCV loader for BTC-USD.
Returns a list of candle dicts consumed by vumanchu.py and vpa.py.
"""

import yfinance as yf
from typing import List, Dict


def fetch_ohlcv(
    ticker: str = "BTC-USD",
    period: str = "3mo",
    interval: str = "1d",
) -> List[Dict]:
    """
    Download OHLCV candles via yfinance.

    Args:
        ticker:   yfinance symbol, default "BTC-USD"
        period:   lookback window ("1mo", "3mo", "6mo", "1y", etc.)
        interval: candle size ("1d", "4h", "1h", etc.)

    Returns:
        List of dicts with keys: timestamp, open, high, low, close, volume.
        Returns empty list on any failure.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception:
        return []

    if df is None or df.empty:
        return []

    df = df.dropna()
    if df.empty:
        return []

    records: List[Dict] = []
    for ts, row in df.iterrows():
        try:
            records.append(
                {
                    "timestamp": str(ts),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    return records
