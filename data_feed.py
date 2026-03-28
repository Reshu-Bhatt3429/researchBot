"""
Real-time BTC price data ingestion.

Streams 1-minute klines from Binance via WebSocket and maintains
a rolling price history buffer for indicator computation.
Also provides REST fallback for historical data bootstrapping.
"""

import asyncio
import json
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import aiohttp
import websockets

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class PriceFeed:
    """
    Maintains a rolling buffer of 1-minute BTC candles and provides
    real-time price snapshots for the signal engine.
    """
    config: Config
    candles: deque = field(default_factory=lambda: deque(maxlen=500))
    current_price: float = 0.0
    last_update_ts: float = 0.0
    _ws: Optional[object] = None
    _running: bool = False

    async def bootstrap_history(self):
        """Fetch historical 1m candles via REST to seed indicators."""
        url = f"{self.config.binance_rest_url}/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "limit": self.config.price_history_minutes,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to fetch history: {resp.status}")
                    return
                data = await resp.json()

        for k in data:
            candle = Candle(
                timestamp=k[0] / 1000.0,
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
            )
            self.candles.append(candle)

        if self.candles:
            self.current_price = self.candles[-1].close
            self.last_update_ts = time.time()
            logger.info(
                f"Bootstrapped {len(self.candles)} candles, "
                f"latest price: ${self.current_price:,.2f}"
            )

    async def start_stream(self):
        """Connect to Binance WebSocket for real-time 1m kline updates."""
        self._running = True
        while self._running:
            try:
                async with websockets.connect(
                    self.config.binance_ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    logger.info("Connected to Binance WebSocket")
                    async for msg in ws:
                        if not self._running:
                            break
                        await self._process_message(msg)
            except (websockets.ConnectionClosed, ConnectionError) as e:
                logger.warning(f"WebSocket disconnected: {e}, reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)

    async def _process_message(self, msg: str):
        """Parse incoming kline WebSocket message."""
        data = json.loads(msg)
        k = data.get("k", {})
        if not k:
            return

        self.current_price = float(k["c"])
        self.last_update_ts = time.time()

        # If candle is closed, append to history
        if k.get("x", False):
            candle = Candle(
                timestamp=k["t"] / 1000.0,
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
            )
            self.candles.append(candle)

    def stop(self):
        self._running = False

    def get_closes(self, n: Optional[int] = None) -> np.ndarray:
        """Return array of close prices, optionally last n."""
        closes = [c.close for c in self.candles]
        if n is not None:
            closes = closes[-n:]
        return np.array(closes, dtype=np.float64)

    def get_volumes(self, n: Optional[int] = None) -> np.ndarray:
        volumes = [c.volume for c in self.candles]
        if n is not None:
            volumes = volumes[-n:]
        return np.array(volumes, dtype=np.float64)

    def get_highs(self, n: Optional[int] = None) -> np.ndarray:
        highs = [c.high for c in self.candles]
        if n is not None:
            highs = highs[-n:]
        return np.array(highs, dtype=np.float64)

    def get_lows(self, n: Optional[int] = None) -> np.ndarray:
        lows = [c.low for c in self.candles]
        if n is not None:
            lows = lows[-n:]
        return np.array(lows, dtype=np.float64)

    def get_ohlcv_matrix(self, n: Optional[int] = None) -> np.ndarray:
        """Return (n, 5) matrix of [open, high, low, close, volume]."""
        data = list(self.candles)
        if n is not None:
            data = data[-n:]
        return np.array(
            [[c.open, c.high, c.low, c.close, c.volume] for c in data],
            dtype=np.float64,
        )

    @property
    def five_min_window_open_price(self) -> Optional[float]:
        """
        Get the opening price of the current 5-minute epoch window.
        Markets open at timestamps divisible by 300.
        """
        now = time.time()
        window_start = now - (now % self.config.market_duration_seconds)

        # Find the candle closest to window start
        for candle in reversed(list(self.candles)):
            if candle.timestamp <= window_start:
                return candle.close  # Close of candle at/before window start
        return None
