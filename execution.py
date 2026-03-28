"""
Execution Layer: Polymarket CLOB via py-clob-client + Gnosis Safe signing.

Uses the same ClobClient initialization pattern as gabagoolBot:
- ApiCreds with HMAC L2 credentials
- Gnosis Safe signature type (type=2)
- Funder address for proxy wallet
- GTC limit orders (maker) as primary, FOK (taker) as fallback
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from config import Config

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    order_id: str = ""
    token_id: str = ""
    side: str = ""
    price: float = 0.0
    size: float = 0.0
    status: str = "pending"
    filled_size: float = 0.0
    timestamp: float = 0.0


@dataclass
class ExecutionEngine:
    """
    Handles all interaction with Polymarket CLOB API via py-clob-client.

    Matches gabagoolBot's proven initialization:
    - ClobClient with Gnosis Safe signing (signature_type=2)
    - GTC limit orders for maker rebate
    - FOK orders for high-confidence snipes
    - Rate limiting to avoid API bans
    """
    config: Config
    _client: Optional[object] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_call: float = 0.0
    _min_interval: float = 0.1  # 10 req/s max
    _pending_orders: dict = field(default_factory=dict)
    _execution_log: list = field(default_factory=list)
    _dry_orders: dict = field(default_factory=dict)

    def setup(self) -> bool:
        """
        Initialize the CLOB client with credentials from .env.
        Mirrors gabagoolBot's Executor.setup() exactly.
        """
        if self.config.paper_trading:
            logger.info("PAPER TRADING MODE -- no real orders")
            return True

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=self.config.polymarket_api_key,
                api_secret=self.config.polymarket_api_secret,
                api_passphrase=self.config.polymarket_passphrase,
            )
            self._client = ClobClient(
                host=self.config.clob_base_url,
                chain_id=self.config.chain_id,
                key=self.config.private_key,
                creds=creds,
                funder=self.config.funder_address or None,
                signature_type=self.config.signature_type,
            )
            logger.info("CLOB client initialized (Gnosis Safe signing)")
            return True
        except ImportError:
            logger.error(
                "py-clob-client not installed. Run: pip install py-clob-client"
            )
            return False
        except Exception as e:
            logger.error(f"CLOB setup failed: {e}")
            return False

    def _rate_limit(self):
        """Throttle API calls to avoid rate limiting."""
        now = time.time()
        wait = self._min_interval - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    # -----------------------------------------------------------------
    #  Market Discovery
    # -----------------------------------------------------------------

    def get_active_btc_5min_markets(self) -> list:
        """
        Find active BTC 5-minute binary markets.
        These resolve at Unix epoch timestamps divisible by 300.
        """
        if self.config.paper_trading or not self._client:
            return []

        self._rate_limit()
        try:
            # Fetch active markets and filter for BTC 5-min
            markets = self._client.get_markets()
            btc_markets = []

            data = markets if isinstance(markets, list) else markets.get("data", [])
            for m in data:
                question = (m.get("question", "") or "").lower()
                if (
                    ("bitcoin" in question or "btc" in question)
                    and ("5" in question or "five" in question)
                    and ("up" in question or "down" in question or "higher" in question)
                ):
                    btc_markets.append(m)

            logger.info(f"Found {len(btc_markets)} active BTC 5-min markets")
            return btc_markets
        except Exception as e:
            logger.error(f"Market search error: {e}")
            return []

    def get_orderbook(self, token_id: str) -> dict:
        """Fetch current orderbook for a binary token."""
        if self.config.paper_trading or not self._client:
            return {"bids": [], "asks": []}

        self._rate_limit()
        try:
            book = self._client.get_order_book(token_id)
            return book
        except Exception as e:
            logger.error(f"Orderbook error: {e}")
            return {"bids": [], "asks": []}

    # -----------------------------------------------------------------
    #  GTC Limit Orders (primary -- earns maker rebate)
    # -----------------------------------------------------------------

    def buy_limit_gtc(
        self,
        token_id: str,
        size_usdc: float,
        price: float,
        neg_risk: bool = False,
    ) -> Optional[str]:
        """
        Place a resting GTC limit BUY order (maker).
        Earns 20% fee rebate when filled.
        """
        if price <= 0 or price >= 1.0:
            logger.error(f"Invalid GTC price {price}")
            return None

        # Minimum 5 tokens per Polymarket requirement
        qty = size_usdc / price
        if qty < 5.0:
            qty = 5.0
            size_usdc = qty * price

        tick = self.config.tick_size
        price = round(price / tick) * tick

        if self.config.paper_trading:
            import random
            order_id = f"PAPER_GTC_{int(time.time() * 1000)}_{random.randint(100, 999)}"
            logger.info(
                f"[PAPER] GTC BUY {qty:.1f} tokens @ ${price:.2f} "
                f"= ${size_usdc:.2f}"
            )
            order = Order(
                order_id=order_id,
                token_id=token_id,
                side="BUY",
                price=price,
                size=qty,
                status="paper",
                timestamp=time.time(),
            )
            self._pending_orders[order_id] = order
            return order_id

        if not self._client:
            logger.error("CLOB client not initialized")
            return None

        with self._lock:
            self._rate_limit()
            try:
                from py_clob_client.order_builder.constants import BUY

                signed_order = self._client.create_and_sign_order(
                    {
                        "token_id": token_id,
                        "price": price,
                        "size": qty,
                        "side": BUY,
                        "fee_rate_bps": 0,  # Maker
                    },
                    neg_risk=neg_risk,
                )
                resp = self._client.post_order(signed_order)
                order_id = resp.get("orderID", "")

                start_time = time.time()
                order = Order(
                    order_id=order_id,
                    token_id=token_id,
                    side="BUY",
                    price=price,
                    size=qty,
                    status="live",
                    timestamp=start_time,
                )
                self._pending_orders[order_id] = order
                self._execution_log.append({
                    "action": "gtc_buy",
                    "order_id": order_id,
                    "price": price,
                    "size": qty,
                    "latency_ms": (time.time() - start_time) * 1000,
                })

                logger.info(
                    f"GTC BUY {qty:.1f} @ ${price:.2f} = ${size_usdc:.2f} | "
                    f"ID: {order_id[:16]}..."
                )
                return order_id

            except Exception as e:
                logger.error(f"GTC order failed: {e}")
                return None

    # -----------------------------------------------------------------
    #  FOK Market Orders (fallback for high-confidence snipes)
    # -----------------------------------------------------------------

    def buy_fok(
        self,
        token_id: str,
        size_usdc: float,
        worst_price: float,
        neg_risk: bool = False,
    ) -> Optional[str]:
        """
        Place a Fill-or-Kill market order (taker).
        Incurs up to 1.56% fee but guarantees immediate fill.
        """
        if worst_price <= 0 or worst_price >= 1.0:
            logger.error(f"Invalid FOK price {worst_price}")
            return None

        qty = size_usdc / worst_price
        if qty < 5.0:
            qty = 5.0

        if self.config.paper_trading:
            import random
            order_id = f"PAPER_FOK_{int(time.time() * 1000)}_{random.randint(100, 999)}"
            logger.info(
                f"[PAPER] FOK BUY {qty:.1f} tokens @ ${worst_price:.2f}"
            )
            return order_id

        if not self._client:
            return None

        with self._lock:
            self._rate_limit()
            try:
                from py_clob_client.order_builder.constants import BUY

                signed_order = self._client.create_and_sign_order(
                    {
                        "token_id": token_id,
                        "price": worst_price,
                        "size": qty,
                        "side": BUY,
                        "fee_rate_bps": 156,  # Worst case taker fee
                        "taker_address": self.config.funder_address,
                    },
                    order_type="FOK",
                    neg_risk=neg_risk,
                )
                resp = self._client.post_order(signed_order, order_type="FOK")
                order_id = resp.get("orderID", "")

                logger.info(f"FOK BUY {qty:.1f} @ ${worst_price:.2f} | ID: {order_id[:16]}...")
                return order_id

            except Exception as e:
                logger.error(f"FOK order failed: {e}")
                return None

    # -----------------------------------------------------------------
    #  Order Management
    # -----------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if self.config.paper_trading:
            if order_id in self._pending_orders:
                self._pending_orders[order_id].status = "cancelled"
            return True

        if not self._client:
            return False

        self._rate_limit()
        try:
            self._client.cancel(order_id)
            if order_id in self._pending_orders:
                self._pending_orders[order_id].status = "cancelled"
            logger.info(f"Cancelled order {order_id[:16]}...")
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def cancel_all(self) -> int:
        """Cancel all pending orders."""
        cancelled = 0
        for order_id, order in list(self._pending_orders.items()):
            if order.status == "live":
                if self.cancel_order(order_id):
                    cancelled += 1
        logger.info(f"Cancelled {cancelled} orders")
        return cancelled

    def cancel_stale_orders(self):
        """Cancel orders exceeding timeout."""
        now = time.time()
        timeout = self.config.order_timeout_seconds
        for order_id, order in list(self._pending_orders.items()):
            if order.status == "live" and (now - order.timestamp) > timeout:
                self.cancel_order(order_id)

    def check_order_filled(self, order_id: str) -> Optional[bool]:
        """
        Check if an order has been filled.
        Returns True (filled), False (open/cancelled), None (error).
        """
        if self.config.paper_trading:
            return None  # Paper trades resolve via simulation

        if not self._client:
            return None

        self._rate_limit()
        try:
            order_data = self._client.get_order(order_id)
            status = order_data.get("status", "")
            size_matched = float(order_data.get("size_matched", 0))

            if order_id in self._pending_orders:
                self._pending_orders[order_id].filled_size = size_matched
                self._pending_orders[order_id].status = status

            return status == "MATCHED" or size_matched > 0
        except Exception as e:
            logger.error(f"Order check failed: {e}")
            return None

    def get_balance(self) -> float:
        """Get current USDC balance on Polymarket."""
        if self.config.paper_trading or not self._client:
            return 0.0

        self._rate_limit()
        try:
            balance = self._client.get_balance()
            return float(balance) if balance else 0.0
        except Exception as e:
            logger.error(f"Balance check failed: {e}")
            return 0.0

    def parse_orderbook(self, book: dict) -> dict:
        """Parse orderbook into trading metrics."""
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        bid_size = float(bids[0]["size"]) if bids else 0.0
        ask_size = float(asks[0]["size"]) if asks else 0.0

        spread = best_ask - best_bid
        mid = (best_ask + best_bid) / 2.0

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spread": spread,
            "mid": mid,
            "spread_pct": spread / mid if mid > 0 else 1.0,
        }

    @property
    def execution_stats(self) -> dict:
        if not self._execution_log:
            return {"total_orders": 0}

        latencies = [e["latency_ms"] for e in self._execution_log]
        return {
            "total_orders": len(self._execution_log),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "pending_orders": sum(
                1 for o in self._pending_orders.values() if o.status == "live"
            ),
        }
