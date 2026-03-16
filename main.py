"""
main.py — BTC Analyser orchestration entry point.

Usage:
    python main.py              # default: 3-month daily candles
    python main.py 6mo          # 6-month daily candles
    python main.py 1mo 4h       # 1-month 4-hour candles

Data flow:
    yfinance ──► fetcher.py ──► vumanchu.py  ──┐
                            └──► vpa.py      ──┤
    CoinGecko + alternative.me ──► onchain.py ──┤
                                                ▼
                                           analyst.py ──► JSON
"""

import json
import sys

from services.fetcher import fetch_ohlcv
from services.onchain import fetch_onchain_data
from services.vumanchu import compute_vumanchu
from services.vpa import compute_vpa
from services.analyst import generate_btc_analysis


def run_analysis(time_range: str = "3mo", interval: str = "1d") -> dict:
    print(f"[1/5] Fetching OHLCV data  ({time_range}, {interval})...")
    candles = fetch_ohlcv("BTC-USD", period=time_range, interval=interval)
    print(f"      {len(candles)} candles loaded.")

    print("[2/5] Fetching on-chain & sentiment data...")
    onchain = fetch_onchain_data()
    current_price: float = onchain["current_price"]
    print(f"      BTC/USD ${current_price:,.0f}  |  F&G {onchain['fear_greed_index']} ({onchain['fear_greed_label']})")

    print("[3/5] Computing VuManChu Cipher B...")
    vumanchu = compute_vumanchu(candles)
    print(f"      Signal: {vumanchu['signal']}  |  Zone: {vumanchu['zone']}  |  WT2: {vumanchu['wt2']}")

    print("[4/5] Computing VPA + S/R levels...")
    vpa = compute_vpa(candles, current_price)
    print(f"      Trend: {vpa['trend']}  |  Dominant: {vpa['dominant_signal']}")
    print(f"      Support: ${vpa['nearest_support']:,.0f}  |  Resistance: ${vpa['nearest_resistance']:,.0f}")

    payload = {
        **onchain,
        "time_range": time_range,
        "price_history": candles,
        "vumanchu": vumanchu,
        "vpa": vpa,
    }

    print("[5/5] Generating AI analysis...")
    analysis = generate_btc_analysis(payload)

    return analysis


if __name__ == "__main__":
    args = sys.argv[1:]
    time_range = args[0] if len(args) > 0 else "3mo"
    interval = args[1] if len(args) > 1 else "1d"

    result = run_analysis(time_range=time_range, interval=interval)
    print("\n" + "=" * 64)
    print(json.dumps(result, indent=2))
