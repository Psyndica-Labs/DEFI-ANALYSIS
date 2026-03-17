"""
app.py — Flask web server for the BTC Analyser dashboard.

Usage:
    pip install flask
    python app.py

Then open: http://localhost:8080
"""

import json
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from services.fetcher import fetch_ohlcv
from services.onchain import fetch_onchain_data
from services.vumanchu import compute_vumanchu
from services.vpa import compute_vpa
from services.analyst import generate_btc_analysis

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyse")
def analyse():
    time_range = request.args.get("range", "3mo")
    interval = request.args.get("interval", "1d")

    try:
        candles = fetch_ohlcv("BTC-USD", period=time_range, interval=interval)
        onchain = fetch_onchain_data()
        current_price = onchain["current_price"]
        vumanchu = compute_vumanchu(candles)
        vpa = compute_vpa(candles, current_price)

        payload = {
            **onchain,
            "time_range": time_range,
            "price_history": candles,
            "vumanchu": vumanchu,
            "vpa": vpa,
        }

        analysis = generate_btc_analysis(payload)

        return jsonify(
            {
                "status": "ok",
                "onchain": {k: v for k, v in onchain.items() if k != "price_history"},
                "vumanchu": vumanchu,
                "vpa": {k: v for k, v in vpa.items() if k != "recent_candles"},
                "recent_candles": vpa.get("recent_candles", []),
                "analysis": analysis,
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, port=port)
