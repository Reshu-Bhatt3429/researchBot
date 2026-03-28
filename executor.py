"""
Executor — Order placement on Polymarket CLOB.

Mirrors gabagoolBot's proven executor pattern:
- py-clob-client with ApiCreds + ClobClient
- GTC limit orders (maker, earns 20% rebate) as primary
- FOK market orders for fast directional entries
- Gnosis Safe signature type support

Research alpha: sub-100ms execution, fixed limit pricing to skip orderbook
"""

import time
import random
import logging
import threading

from config import (
    POLYMARKET_PRIVATE_KEY, POLYMARKET_API_KEY, POLYMARKET_API_SECRET,
    POLYMARKET_API_PASSPHRASE, POLYMARKET_FUNDER_ADDRESS,
    POLYMARKET_SIGNATURE_TYPE, CHAIN_ID, CLOB_HOST, LIVE_TRADING,
    TICK_SIZE, API_RATE_LIMIT, DEFAULT_BANKROLL_USDC, LOG_DIR,
)

logger = logging.getLogger("executor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(f"{LOG_DIR}/executor.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(sh)


class Executor:
    """
    Places BUY/SELL orders on the Polymarket CLOB.
    In dry-run mode (LIVE_TRADING=false) all orders are simulated.
    """

    def __init__(self):
        self._client = None
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._min_interval = 1.0 / API_RATE_LIMIT

        # Dry-run order simulation
        self._dry_orders: dict[str, dict] = {}

    def setup(self) -> bool:
        if not LIVE_TRADING:
            logger.info("DRY RUN -- no real orders will be placed")
            return True

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=POLYMARKET_API_KEY,
                api_secret=POLYMARKET_API_SECRET,
                api_passphrase=POLYMARKET_API_PASSPHRASE,
            )
            self._client = ClobClient(
                host=CLOB_HOST,
                chain_id=CHAIN_ID,
                key=POLYMARKET_PRIVATE_KEY,
                creds=creds,
                funder=POLYMARKET_FUNDER_ADDRESS or None,
                signature_type=POLYMARKET_SIGNATURE_TYPE,
            )
            logger.info(
                f"CLOB client initialized | SigType={POLYMARKET_SIGNATURE_TYPE} | "
                f"Funder={POLYMARKET_FUNDER_ADDRESS[:10]}..."
            )
            return True
        except ImportError:
            logger.error("py-clob-client not installed: pip install py-clob-client")
            return False
        except Exception as e:
            logger.error(f"CLOB setup failed: {e}")
            return False

    def _rate_limit(self):
        now = time.time()
        wait = self._min_interval - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    # -----------------------------------------------------------------
    #  GTC Limit Orders (primary — maker, earns 20% fee rebate)
    # -----------------------------------------------------------------

    def buy_limit_gtc(self, token_id: str, size_usdc: float, price: float,
                      neg_risk: bool = False) -> str | None:
        """
        Place a resting GTC limit BUY order.
        Sits on book until filled or canceled.
        """
        if price <= 0 or price >= 1.0:
            logger.error(f"Invalid GTC price {price}")
            return None

        qty = size_usdc / price
        if qty < 5.0:
            qty = 5.0
            size_usdc = qty * price

        if not LIVE_TRADING:
            order_id = f"DRY_GTC_{int(time.time() * 1000)}_{random.randint(100, 999)}"
            self._dry_orders[order_id] = {
                "size": qty,
                "price": price,
                "placed_at": time.time(),
                "fill_delay": random.uniform(15, 90),
                "will_fill": random.random() < 0.65,
            }
            logger.info(
                f"[DRY] GTC BUY {qty:.1f} tokens @ ${price:.2f} "
                f"= ${size_usdc:.2f} | {token_id[:16]}..."
            )
            return order_id

        if not self._client:
            return None

        with self._lock:
            self._rate_limit()
            try:
                from py_clob_client.order_builder.constants import BUY
                from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions

                order_args = OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=qty,
                    side=BUY,
                    fee_rate_bps=0,
                )
                options = PartialCreateOrderOptions(
                    tick_size=TICK_SIZE,
                    neg_risk=neg_risk,
                )
                signed = self._client.create_and_sign_order(order_args, options)
                resp = self._client.post_order(signed, orderType=OrderType.GTC)
                order_id = resp.get("orderID", "")

                logger.info(
                    f"GTC BUY {qty:.1f} @ ${price:.2f} = ${size_usdc:.2f} | "
                    f"ID: {order_id[:16]}..."
                )
                return order_id

            except Exception as e:
                logger.error(f"GTC order failed: {e}")
                return None

    # -----------------------------------------------------------------
    #  FOK Market Orders (fast directional entry)
    # -----------------------------------------------------------------

    def buy_fok(self, token_id: str, size_usdc: float, price: float,
                neg_risk: bool = False) -> str | None:
        """
        Place a Fill-or-Kill order at fixed limit price.
        Research: skip orderbook fetch, use fixed $0.55 limit — saves ~300ms.
        """
        if price <= 0 or price >= 1.0:
            logger.error(f"Invalid FOK price {price}")
            return None

        qty = size_usdc / price
        if qty < 5.0:
            qty = 5.0

        if not LIVE_TRADING:
            order_id = f"DRY_FOK_{int(time.time() * 1000)}_{random.randint(100, 999)}"
            # Simulate immediate fill at given price
            self._dry_orders[order_id] = {
                "size": qty,
                "price": price,
                "placed_at": time.time(),
                "fill_delay": 0,
                "will_fill": True,
            }
            logger.info(
                f"[DRY] FOK BUY {qty:.1f} tokens @ ${price:.2f} "
                f"= ${size_usdc:.2f} | {token_id[:16]}..."
            )
            return order_id

        if not self._client:
            return None

        with self._lock:
            self._rate_limit()
            try:
                from py_clob_client.order_builder.constants import BUY
                from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions

                order_args = OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=qty,
                    side=BUY,
                )
                options = PartialCreateOrderOptions(
                    tick_size=TICK_SIZE,
                    neg_risk=neg_risk,
                )
                signed = self._client.create_and_sign_order(order_args, options)
                resp = self._client.post_order(signed, orderType=OrderType.FOK)
                order_id = resp.get("orderID", "")

                logger.info(
                    f"FOK BUY {qty:.1f} @ ${price:.2f} | ID: {order_id[:16]}..."
                )
                return order_id

            except Exception as e:
                logger.error(f"FOK order failed: {e}")
                return None

    # -----------------------------------------------------------------
    #  SELL Orders (for early exit / profit taking)
    # -----------------------------------------------------------------

    def sell_fok(self, token_id: str, qty: float, price: float,
                 neg_risk: bool = False) -> str | None:
        """Sell tokens via FOK for early profit exit."""
        if price <= 0 or price >= 1.0:
            return None

        if not LIVE_TRADING:
            order_id = f"DRY_SELL_{int(time.time() * 1000)}_{random.randint(100, 999)}"
            logger.info(f"[DRY] SELL {qty:.1f} tokens @ ${price:.2f}")
            return order_id

        if not self._client:
            return None

        with self._lock:
            self._rate_limit()
            try:
                from py_clob_client.order_builder.constants import SELL
                from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions

                order_args = OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=qty,
                    side=SELL,
                )
                options = PartialCreateOrderOptions(
                    tick_size=TICK_SIZE,
                    neg_risk=neg_risk,
                )
                signed = self._client.create_and_sign_order(order_args, options)
                resp = self._client.post_order(signed, orderType=OrderType.FOK)
                return resp.get("orderID", "")

            except Exception as e:
                logger.error(f"SELL order failed: {e}")
                return None

    # -----------------------------------------------------------------
    #  Order Management
    # -----------------------------------------------------------------

    def get_order_status(self, order_id: str) -> dict:
        """Check fill status. Returns {"status": str, "size_matched": float}."""
        if not LIVE_TRADING:
            dry = self._dry_orders.get(order_id)
            if not dry:
                return {"status": "UNKNOWN", "size_matched": 0}
            elapsed = time.time() - dry["placed_at"]
            if dry["will_fill"] and elapsed >= dry["fill_delay"]:
                return {"status": "MATCHED", "size_matched": dry["size"]}
            elif not dry["will_fill"] and elapsed >= dry["fill_delay"]:
                return {"status": "CANCELED", "size_matched": 0}
            return {"status": "LIVE", "size_matched": 0}

        if not self._client:
            return {"status": "UNKNOWN", "size_matched": 0}

        self._rate_limit()
        try:
            resp = self._client.get_order(order_id)
            return {
                "status": resp.get("status", "UNKNOWN"),
                "size_matched": float(resp.get("size_matched", 0)),
            }
        except Exception as e:
            logger.warning(f"Status check error: {e}")
            return {"status": "ERROR", "size_matched": 0}

    def cancel(self, order_id: str) -> bool:
        """Cancel a resting order."""
        if not LIVE_TRADING:
            if order_id in self._dry_orders:
                del self._dry_orders[order_id]
            logger.info(f"[DRY] Cancelled {order_id}")
            return True

        if not self._client:
            return False

        self._rate_limit()
        try:
            self._client.cancel(order_id)
            logger.info(f"Cancelled {order_id[:16]}...")
            return True
        except Exception as e:
            logger.warning(f"Cancel failed: {e}")
            return False

    def cancel_all(self):
        """Cancel all resting orders."""
        if not LIVE_TRADING:
            count = len(self._dry_orders)
            self._dry_orders.clear()
            logger.info(f"[DRY] Cancelled all {count} orders")
            return

        if self._client:
            try:
                self._client.cancel_all()
                logger.info("Cancelled all orders")
            except Exception as e:
                logger.warning(f"Cancel all failed: {e}")

    def get_balance(self) -> float:
        """Get USDC balance on Polymarket."""
        if not LIVE_TRADING:
            return DEFAULT_BANKROLL_USDC

        if not self._client:
            return 0.0

        self._rate_limit()
        try:
            bal = self._client.get_balance_allowance()
            return float(bal.get("balance", 0)) if isinstance(bal, dict) else 0.0
        except Exception as e:
            logger.warning(f"Balance check failed: {e}")
            return 0.0
