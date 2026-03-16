"""
vumanchu.py — VuManChu Cipher B indicator engine.

Computes:
  wt1 / wt2      : WaveTrend oscillator (LazyBear)
  mfi            : Money Flow Index (-100 to +100, VuManChu scale)
  rsi            : RSI-14
  zone           : oscillator zone label
  signal         : BUY / STRONG BUY / SELL / STRONG SELL / HOLD / WATCH
  buy_signal     : bool — green dot (WT cross up from oversold)
  sell_signal    : bool — red dot (WT cross down from overbought)
  strong_buy     : bool — gold dot (WT2 < -60 at cross)
  strong_sell    : bool — strong red (WT2 > +60 at cross)
  bullish_div    : bool — price lower low / WT1 higher low
  bearish_div    : bool — price higher high / WT1 lower high
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# WaveTrend parameters (LazyBear default)
_WT_CHANNEL_LEN = 9
_WT_AVG_LEN = 12
_WT_MA_LEN = 4      # smoothing for wt2
_WT_OVERSOLD = -53
_WT_OVERBOUGHT = 53
_WT_STRONG_OVERSOLD = -60
_WT_STRONG_OVERBOUGHT = 60

# MFI parameters
_MFI_PERIOD = 60

# Divergence lookback
_DIV_LOOKBACK = 20
_DIV_PIVOT_GAP = 3


def compute_vumanchu(candles: List[Dict]) -> Dict[str, Any]:
    df = pd.DataFrame(candles)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    rsi_val = _rsi(close, 14).iloc[-1]
    wt1_series, wt2_series = _wavetrend(high, low, close)
    mfi_series = _mfi(open_, close)

    wt1 = round(float(wt1_series.iloc[-1]), 2)
    wt2 = round(float(wt2_series.iloc[-1]), 2)
    mfi = round(float(mfi_series.iloc[-1]), 2)

    zone = _zone(wt2)

    buy_signal, sell_signal, strong_buy, strong_sell = _detect_crosses(
        wt1_series, wt2_series, lookback=3
    )

    bullish_div = _bullish_divergence(close, wt1_series)
    bearish_div = _bearish_divergence(close, wt1_series)

    if strong_buy or (buy_signal and bullish_div):
        signal = "STRONG BUY"
    elif buy_signal:
        signal = "BUY"
    elif strong_sell or (sell_signal and bearish_div):
        signal = "STRONG SELL"
    elif sell_signal:
        signal = "SELL"
    else:
        signal = "HOLD / WATCH"

    return {
        "wt1": wt1,
        "wt2": wt2,
        "mfi": mfi,
        "rsi": round(float(rsi_val), 2),
        "zone": zone,
        "signal": signal,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
        "strong_buy": strong_buy,
        "strong_sell": strong_sell,
        "bullish_divergence": bullish_div,
        "bearish_divergence": bearish_div,
    }


# ── Indicator maths ──────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _wavetrend(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """LazyBear WaveTrend oscillator."""
    hlc3 = (high + low + close) / 3
    esa = _ema(hlc3, _WT_CHANNEL_LEN)
    d = _ema((hlc3 - esa).abs(), _WT_CHANNEL_LEN)
    ci = (hlc3 - esa) / (0.015 * d.replace(0, np.nan)).ffill()
    wt1 = _ema(ci, _WT_AVG_LEN)
    wt2 = _sma(wt1, _WT_MA_LEN)
    return wt1, wt2


def _mfi(open_: pd.Series, close: pd.Series) -> pd.Series:
    """
    VuManChu Cipher B money flow: compares bull vs bear candle body size
    over a rolling window, normalised to -100…+100.
    """
    mf = close - open_
    pmf = mf.clip(lower=0).rolling(_MFI_PERIOD).sum()
    nmf = (-mf).clip(lower=0).rolling(_MFI_PERIOD).sum()
    denom = (pmf + nmf).replace(0, np.nan)
    return ((pmf - nmf) / denom * 100).fillna(0)


# ── Signal detection ─────────────────────────────────────────────────────────

def _detect_crosses(
    wt1: pd.Series, wt2: pd.Series, lookback: int = 3
) -> Tuple[bool, bool, bool, bool]:
    buy_signal = sell_signal = strong_buy = strong_sell = False

    wt1_vals = wt1.values
    wt2_vals = wt2.values
    n = len(wt1_vals)

    for i in range(max(1, n - lookback), n):
        prev_wt1, curr_wt1 = wt1_vals[i - 1], wt1_vals[i]
        prev_wt2, curr_wt2 = wt2_vals[i - 1], wt2_vals[i]

        cross_up = prev_wt1 <= prev_wt2 and curr_wt1 > curr_wt2
        cross_down = prev_wt1 >= prev_wt2 and curr_wt1 < curr_wt2

        if cross_up and curr_wt2 < _WT_OVERSOLD:
            buy_signal = True
            if curr_wt2 < _WT_STRONG_OVERSOLD:
                strong_buy = True

        if cross_down and curr_wt2 > _WT_OVERBOUGHT:
            sell_signal = True
            if curr_wt2 > _WT_STRONG_OVERBOUGHT:
                strong_sell = True

    return buy_signal, sell_signal, strong_buy, strong_sell


# ── Divergence detection ─────────────────────────────────────────────────────

def _bullish_divergence(close: pd.Series, wt1: pd.Series) -> bool:
    """Price lower low + WT1 higher low, both in oversold territory."""
    c = close.iloc[-_DIV_LOOKBACK:].values
    w = wt1.iloc[-_DIV_LOOKBACK:].values

    c_lows = _pivots_low(c)
    w_lows = _pivots_low(w)

    if len(c_lows) < 2 or len(w_lows) < 2:
        return False

    price_lower_low = c[c_lows[-1]] < c[c_lows[-2]]
    wt1_higher_low = w[w_lows[-1]] > w[w_lows[-2]]
    in_oversold = w[-1] < _WT_OVERSOLD

    return price_lower_low and wt1_higher_low and in_oversold


def _bearish_divergence(close: pd.Series, wt1: pd.Series) -> bool:
    """Price higher high + WT1 lower high, both in overbought territory."""
    c = close.iloc[-_DIV_LOOKBACK:].values
    w = wt1.iloc[-_DIV_LOOKBACK:].values

    c_highs = _pivots_high(c)
    w_highs = _pivots_high(w)

    if len(c_highs) < 2 or len(w_highs) < 2:
        return False

    price_higher_high = c[c_highs[-1]] > c[c_highs[-2]]
    wt1_lower_high = w[w_highs[-1]] < w[w_highs[-2]]
    in_overbought = w[-1] > _WT_OVERBOUGHT

    return price_higher_high and wt1_lower_high and in_overbought


def _pivots_low(arr: np.ndarray) -> List[int]:
    indices = []
    for i in range(_DIV_PIVOT_GAP, len(arr) - _DIV_PIVOT_GAP):
        if arr[i] == min(arr[i - _DIV_PIVOT_GAP : i + _DIV_PIVOT_GAP + 1]):
            if not indices or i - indices[-1] >= _DIV_PIVOT_GAP:
                indices.append(i)
    return indices


def _pivots_high(arr: np.ndarray) -> List[int]:
    indices = []
    for i in range(_DIV_PIVOT_GAP, len(arr) - _DIV_PIVOT_GAP):
        if arr[i] == max(arr[i - _DIV_PIVOT_GAP : i + _DIV_PIVOT_GAP + 1]):
            if not indices or i - indices[-1] >= _DIV_PIVOT_GAP:
                indices.append(i)
    return indices


# ── Zone ─────────────────────────────────────────────────────────────────────

def _zone(wt2: float) -> str:
    if wt2 <= _WT_STRONG_OVERSOLD:
        return "Strong Oversold"
    elif wt2 <= _WT_OVERSOLD:
        return "Oversold"
    elif wt2 >= _WT_STRONG_OVERBOUGHT:
        return "Strong Overbought"
    elif wt2 >= _WT_OVERBOUGHT:
        return "Overbought"
    else:
        return "Neutral"
