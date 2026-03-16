"""
vpa.py — Anna Coulling Volume Price Analysis + Support/Resistance engine.

Computes:
  dominant_signal        : Most significant VPA pattern in last 5 candles
  support_levels         : Key S/R floors, closest first
  resistance_levels      : Key S/R ceilings, closest first
  nearest_support        : Closest support below current price
  nearest_resistance     : Closest resistance above current price
  pct_to_support         : % distance to nearest support
  pct_to_resistance      : % distance to nearest resistance
  proximity_note         : Warning string if price within 1.5% of a level
  trend                  : EMA-stack trend label enum
  trend_label            : Human-readable EMA stack description
  ema_20 / ema_50 / ema_200 : Current EMA values
  recent_candles         : Last 5 candles with VPA metadata
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

_VOL_MA_PERIOD = 20     # rolling volume average
_SR_MERGE_PCT = 0.005   # merge S/R levels within 0.5%
_SR_PIVOT_WINDOW = 3    # bars either side to confirm a pivot
_SR_LOOKBACK = 90       # candles to scan for pivots
_PROX_WARN_PCT = 1.5    # flag proximity warning below this %

# VPA signal priority (higher = more significant)
_SIGNAL_PRIORITY: List[str] = [
    "selling_climax",
    "buying_climax",
    "stopping_volume",
    "topping_volume",
    "validated_bull",
    "validated_bear",
    "no_result",
    "distribution",
    "accumulation",
    "anomaly_up",
    "anomaly_down",
    "neutral",
]


def compute_vpa(candles: List[Dict], current_price: float) -> Dict[str, Any]:
    df = pd.DataFrame(candles)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)

    # EMA trend stack
    ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema_200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    trend, trend_label = _trend_from_ema(current_price, ema_20, ema_50, ema_200)

    # Volume ratios
    avg_vol = volume.rolling(_VOL_MA_PERIOD).mean().fillna(volume.mean())
    vol_ratio = (volume / avg_vol).fillna(1.0)

    # Per-candle VPA
    all_candle_meta: List[Dict] = []
    for i in range(len(df)):
        o = float(open_.iloc[i])
        h = float(high.iloc[i])
        l = float(low.iloc[i])
        c = float(close.iloc[i])
        vr = float(vol_ratio.iloc[i])

        sig = _classify_candle(o, h, l, c, vr)
        body_range = h - l
        body_pct = round(abs(c - o) / body_range * 100, 1) if body_range > 0 else 0.0

        ts_col = df["timestamp"].iloc[i] if "timestamp" in df.columns else str(i)
        all_candle_meta.append(
            {
                "timestamp": str(ts_col),
                "signal": sig,
                "vol_label": _vol_label(vr),
                "vol_ratio": round(vr, 2),
                "is_bull": c > o,
                "body_pct": body_pct,
            }
        )

    recent_candles = all_candle_meta[-5:]

    # Dominant signal from last 5
    dominant_signal = "neutral"
    for sig in _SIGNAL_PRIORITY:
        if any(c["signal"] == sig for c in recent_candles):
            dominant_signal = sig
            break

    # S/R levels
    support_levels, resistance_levels = _compute_sr(
        high.values, low.values, current_price
    )

    nearest_support: Optional[float] = support_levels[0] if support_levels else None
    nearest_resistance: Optional[float] = resistance_levels[0] if resistance_levels else None

    pct_to_support: Optional[float] = (
        round((current_price - nearest_support) / current_price * 100, 2)
        if nearest_support is not None
        else None
    )
    pct_to_resistance: Optional[float] = (
        round((nearest_resistance - current_price) / current_price * 100, 2)
        if nearest_resistance is not None
        else None
    )

    proximity_note = ""
    if pct_to_support is not None and pct_to_support < _PROX_WARN_PCT:
        proximity_note = (
            f"Price within {pct_to_support:.1f}% of support at ${nearest_support:,.0f}"
        )
    elif pct_to_resistance is not None and pct_to_resistance < _PROX_WARN_PCT:
        proximity_note = (
            f"Price within {pct_to_resistance:.1f}% of resistance at ${nearest_resistance:,.0f}"
        )

    return {
        "dominant_signal": dominant_signal,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "pct_to_support": pct_to_support,
        "pct_to_resistance": pct_to_resistance,
        "proximity_note": proximity_note,
        "trend": trend,
        "trend_label": trend_label,
        "ema_20": round(ema_20, 2),
        "ema_50": round(ema_50, 2),
        "ema_200": round(ema_200, 2),
        "recent_candles": recent_candles,
    }


# ── Candle classification (Coulling rules) ────────────────────────────────────

def _classify_candle(o: float, h: float, l: float, c: float, vol_ratio: float) -> str:
    full_range = h - l
    if full_range == 0:
        return "neutral"

    body = abs(c - o)
    body_pct = body / full_range
    upper_wick = (h - max(o, c)) / full_range
    lower_wick = (min(o, c) - l) / full_range
    is_bull = c > o

    is_wide = body_pct > 0.60
    is_narrow = body_pct < 0.25
    is_very_high_vol = vol_ratio >= 2.0
    is_high_vol = vol_ratio >= 1.5
    is_low_vol = vol_ratio < 0.70

    # Selling climax: very high volume, heavy lower wick, bear candle
    if is_very_high_vol and lower_wick > 0.30 and not is_bull:
        return "selling_climax"

    # Buying climax: very high volume, heavy upper wick, bull candle
    if is_very_high_vol and upper_wick > 0.30 and is_bull:
        return "buying_climax"

    # No result (absorption): very high volume, narrow body — reversal watch
    if is_very_high_vol and is_narrow:
        return "no_result"

    # Stopping volume: high vol, bull candle, lower wick absorbing sellers
    if is_high_vol and is_bull and lower_wick > 0.25:
        return "stopping_volume"

    # Topping volume: high vol, bear candle, upper wick rejecting buyers
    if is_high_vol and not is_bull and upper_wick > 0.25:
        return "topping_volume"

    # Validated bull: wide bull body + high volume
    if is_wide and is_bull and is_high_vol:
        return "validated_bull"

    # Validated bear: wide bear body + high volume
    if is_wide and not is_bull and is_high_vol:
        return "validated_bear"

    # Distribution: narrow body + high volume (smart money unloading)
    if is_narrow and is_high_vol:
        return "distribution"

    # Anomaly up: bull candle + low volume (suspect move)
    if is_bull and is_low_vol:
        return "anomaly_up"

    # Anomaly down: bear candle + low volume (shallow pullback)
    if not is_bull and is_low_vol:
        return "anomaly_down"

    # Accumulation: narrow doji + average volume
    if is_narrow:
        return "accumulation"

    return "neutral"


def _vol_label(vol_ratio: float) -> str:
    if vol_ratio >= 2.0:
        return "very_high"
    elif vol_ratio >= 1.5:
        return "high"
    elif vol_ratio >= 0.70:
        return "average"
    else:
        return "low"


# ── EMA trend stack ───────────────────────────────────────────────────────────

def _trend_from_ema(
    price: float, ema20: float, ema50: float, ema200: float
) -> Tuple[str, str]:
    if price > ema20 > ema50 > ema200:
        return "strong_uptrend", "Price > EMA20 > EMA50 > EMA200 — full bull alignment"
    elif price > ema50 > ema200:
        return "uptrend", "Price above EMA50 and EMA200 — above long-term trend"
    elif price < ema20 < ema50 < ema200:
        return "strong_downtrend", "Price < EMA20 < EMA50 < EMA200 — full bear alignment"
    elif price < ema50 < ema200:
        return "downtrend", "Price below EMA50 and EMA200 — below long-term trend"
    else:
        return "ranging", "Mixed EMA alignment — ranging or transitional"


# ── Support / Resistance ──────────────────────────────────────────────────────

def _compute_sr(
    high: np.ndarray,
    low: np.ndarray,
    current_price: float,
    max_levels: int = 5,
) -> Tuple[List[float], List[float]]:
    """
    Identify pivot highs and lows as S/R levels.
    Levels are merged when within SR_MERGE_PCT of each other.
    """
    lookback_high = high[-_SR_LOOKBACK:]
    lookback_low = low[-_SR_LOOKBACK:]
    w = _SR_PIVOT_WINDOW

    pivot_highs: List[float] = []
    pivot_lows: List[float] = []

    for i in range(w, len(lookback_high) - w):
        if lookback_high[i] == max(lookback_high[i - w : i + w + 1]):
            pivot_highs.append(lookback_high[i])
        if lookback_low[i] == min(lookback_low[i - w : i + w + 1]):
            pivot_lows.append(lookback_low[i])

    pivot_highs = _merge_levels(sorted(pivot_highs, reverse=True))
    pivot_lows = _merge_levels(sorted(pivot_lows))

    # Separate relative to current price (flipped levels act as the opposite)
    resistance = sorted([p for p in pivot_highs if p > current_price])
    support = sorted([p for p in pivot_lows if p < current_price], reverse=True)

    # Pivot highs below price → old resistance now acting as support
    old_res_as_support = sorted(
        [p for p in pivot_highs if p < current_price], reverse=True
    )[:3]
    # Pivot lows above price → old support now acting as resistance
    old_sup_as_resistance = sorted(
        [p for p in pivot_lows if p > current_price]
    )[:3]

    all_support = sorted(
        set(support[:max_levels] + old_res_as_support), reverse=True
    )[:max_levels]
    all_resistance = sorted(
        set(resistance[:max_levels] + old_sup_as_resistance)
    )[:max_levels]

    return (
        [round(p, 0) for p in all_support],
        [round(p, 0) for p in all_resistance],
    )


def _merge_levels(levels: List[float]) -> List[float]:
    if not levels:
        return []
    merged = [levels[0]]
    for level in levels[1:]:
        if merged[-1] == 0:
            merged.append(level)
            continue
        if abs(level - merged[-1]) / merged[-1] > _SR_MERGE_PCT:
            merged.append(level)
    return merged
