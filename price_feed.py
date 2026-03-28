"""
Price Feed — Real-time BTC price from Binance aggTrade WebSocket.

Uses aggTrade (not kline) for lowest latency — every individual trade.
Research: millisecond-level price data critical for racing market makers.
"""

import json
import time
import logging
import threading

import websocket

from config import BINANCE_WS_URL, LOG_DIR

logger = logging.getLogger("feed")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(f"{LOG_DIR}/feed.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)


class BtcFeed:
    """
    Real-time BTC/USDT price via Binance aggTrade WebSocket.

    aggTrade fires on every trade execution — lower latency than kline.
    The bot uses this to detect momentum (price move from window open)
    within milliseconds of it happening.
    """

    def __init__(self):
        self.price: float = 0.0
        self.last_update: float = 0.0
        self._ws = None
        self._thread = None
        self._running = False

    def start(self):
        """Start the WebSocket listener in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("BTC price feed starting...")

        # Wait for first price
        deadline = time.time() + 10
        while self.price == 0 and time.time() < deadline:
            time.sleep(0.1)

        if self.price > 0:
            logger.info(f"BTC price feed live: ${self.price:,.2f}")
        else:
            logger.warning("BTC price feed: no price received in 10s")

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _run(self):
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    BINANCE_WS_URL,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.warning(f"WS error: {e}")
            if self._running:
                time.sleep(1)

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            self.price = float(data["p"])
            self.last_update = time.time()
        except (KeyError, ValueError):
            pass

    def _on_error(self, ws, error):
        logger.warning(f"WS error: {error}")

    def _on_close(self, ws, code, msg):
        logger.info(f"WS closed: {code} {msg}")
