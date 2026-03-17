"""
onchain.py — CoinGecko + alternative.me data fetcher.
Provides: current_price, market_cap, volume_24h, ath, ath_change_pct,
          dominance, mvrv_ratio, halving_cycle_phase,
          fear_greed_index, fear_greed_label.
"""

import httpx
from datetime import date
from typing import Dict, Any, Optional

HALVING_DATE = date(2024, 4, 20)

_COINGECKO_COIN_URL = "https://api.coingecko.com/api/v3/coins/bitcoin"
_COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
_FNG_URL = "https://api.alternative.me/fng/?limit=1"

_COINGECKO_PARAMS = {
    "localization": "false",
    "tickers": "false",
    "market_data": "true",
    "community_data": "false",
    "developer_data": "false",
}


def fetch_onchain_data() -> Dict[str, Any]:
    """
    Fetch on-chain metrics and market sentiment.
    Each API call is independently guarded — partial data is returned
    rather than crashing the entire pipeline.
    MVRV is not available without a paid data provider and is returned as None.
    """
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    ath: Optional[float] = None
    ath_change_pct: Optional[float] = None
    dominance: Optional[float] = None
    fear_greed_index: Optional[int] = None
    fear_greed_label: Optional[str] = None

    with httpx.Client(timeout=20) as client:
        # ── CoinGecko coin data ──
        try:
            cg_resp = client.get(_COINGECKO_COIN_URL, params=_COINGECKO_PARAMS)
            cg_resp.raise_for_status()
            md = cg_resp.json().get("market_data", {})
            current_price = md.get("current_price", {}).get("usd")
            market_cap = md.get("market_cap", {}).get("usd")
            volume_24h = md.get("total_volume", {}).get("usd")
            ath = md.get("ath", {}).get("usd")
            ath_change_pct = md.get("ath_change_percentage", {}).get("usd")
        except Exception:
            pass  # fields remain None

        # ── CoinGecko global data (dominance) ──
        try:
            global_resp = client.get(_COINGECKO_GLOBAL_URL)
            global_resp.raise_for_status()
            global_data = global_resp.json().get("data", {})
            raw_dom = global_data.get("market_cap_percentage", {}).get("btc")
            dominance = round(raw_dom, 2) if raw_dom is not None else None
        except Exception:
            pass

        # ── Fear & Greed ──
        try:
            fng_resp = client.get(_FNG_URL)
            fng_resp.raise_for_status()
            fng_entry = fng_resp.json()["data"][0]
            fear_greed_index = int(fng_entry["value"])
            fear_greed_label = fng_entry["value_classification"]
        except Exception:
            pass

    halving_cycle_phase = _halving_phase(_months_since_halving())

    return {
        "current_price": current_price,
        "market_cap": market_cap,
        "volume_24h": volume_24h,
        "ath": ath,
        "ath_change_pct": round(ath_change_pct, 2) if ath_change_pct is not None else None,
        "dominance": dominance,
        "mvrv_ratio": None,  # requires Glassnode or Messari paid API
        "halving_cycle_phase": halving_cycle_phase,
        "fear_greed_index": fear_greed_index,
        "fear_greed_label": fear_greed_label,
    }


def _months_since_halving() -> int:
    today = date.today()
    return (today.year - HALVING_DATE.year) * 12 + (today.month - HALVING_DATE.month)


def _halving_phase(months: int) -> str:
    if months < 6:
        return "Early post-halving (supply shock forming)"
    elif months < 12:
        return "Mid post-halving (historically bullish accumulation)"
    elif months < 24:
        return "Bull market expansion phase"
    elif months < 30:
        return "Late cycle / distribution phase"
    elif months < 42:
        return "Bear market / capitulation phase"
    else:
        return "Pre-halving accumulation"
