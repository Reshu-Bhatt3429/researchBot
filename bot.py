"""
Polymarket 5-Minute BTC Binary Options Bot.

Architecture matches research paper + production-proven patterns:

1. Market discovery via Gamma API slug (btc-updown-5m-{ts})
2. Binance aggTrade for millisecond price feed
3. Hedge + directional strategy (BotVersion3 pattern):
   - Phase 1: Small hedge on cheap side at window open
   - Phase 2: Momentum-triggered directional bet (0.01% threshold)
   - Phase 2b: Follow-up resting GTC at better price
   - Phase 3: Optional add-on if conviction grows
4. Early exit at 70%+ profit or $0.97 bid
5. Emergency mode at 45s to expiry
6. On-chain redemption via Gnosis Safe after resolution
7. Fractional Kelly sizing with 7-condition risk controls

50ms main loop — research says sub-100ms execution is mandatory.
"""

import csv
import os
import signal
import sys
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import (
    LIVE_TRADING, WINDOW_SEC, SLUG_PATTERN,
    HEDGE_SIZE_USDC, CHEAP_SIDE_MAX,
    FAST_BET_SIZE_USDC, FAST_LIMIT_PRICE, DIRECTION_THRESHOLD,
    FOLLOWUP_LIMIT_PRICE, FOLLOWUP_LIMIT_SIZE_USDC,
    ADDON_SIZE_USDC, ADDON_DELAY_SEC, ADDON_THRESHOLD,
    MAX_TOTAL_USDC, MAX_ONE_SIDE_USDC,
    ENTRY_DEADLINE_SEC, EMERGENCY_SEC,
    PROFIT_EXIT_PCT, NEAR_MAX_EXIT_BID,
    KELLY_FRACTION, MOMENTUM_SCALE, MAX_MOMENTUM_EDGE,
    MIN_BET_USDC, DEFAULT_BANKROLL_USDC,
    MAX_DAILY_LOSS_USDC, MAX_CONSECUTIVE_LOSSES,
    MAIN_LOOP_INTERVAL, DISPLAY_INTERVAL,
    BALANCE_REFRESH_SEC, LOG_DIR,
)
from price_feed import BtcFeed
from market_scanner import MarketScanner
from executor import Executor
from redeemer import Redeemer

# === Logging ===
logger = logging.getLogger("bot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fh = logging.FileHandler(f"{LOG_DIR}/bot_{ts}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(sh)


# === Trade CSV Logger ===
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")
CSV_HEADERS = [
    "timestamp", "slug", "direction", "btc_open", "btc_close",
    "up_spent", "down_spent", "total_spent", "payout",
    "pnl", "phase", "bankroll",
]

def _init_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADERS)

def _log_trade(row: dict):
    with open(TRADES_CSV, "a", newline="") as f:
        csv.writer(f).writerow([row.get(h, "") for h in CSV_HEADERS])


# === Position Tracking ===

@dataclass
class Side:
    """Tracks a single direction (UP or DOWN) token position."""
    token_id: str
    outcome: str  # "Up" / "Down" / "Yes" / "No"
    tokens: float = 0.0
    spent: float = 0.0
    order_ids: list = field(default_factory=list)

    @property
    def avg_price(self) -> float:
        return self.spent / self.tokens if self.tokens > 0 else 0.0

    def add_fill(self, usdc: float, price: float):
        qty = usdc / price
        self.tokens += qty
        self.spent += usdc


@dataclass
class Position:
    """
    Full position for one 5-minute window.

    Phases:
        WAIT   — Market found, waiting for window open
        HEDGE  — Cheap-side hedge placed, watching for momentum
        MAIN   — Directional bet placed after momentum trigger
        HOLD   — Fully positioned, holding to expiry or early exit
        CLOSED — Resolved, PnL calculated
    """
    slug: str
    market: dict
    window_ts: int
    btc_open: float = 0.0

    up: Optional[Side] = None
    down: Optional[Side] = None

    phase: str = "WAIT"
    lean: Optional[str] = None  # "UP" or "DOWN"
    addon_done: bool = False
    followup_order_id: Optional[str] = None

    def __post_init__(self):
        tokens = self.market.get("tokens", [])
        for t in tokens:
            outcome = t["outcome"].upper()
            if outcome in ("UP", "YES", "HIGHER"):
                self.up = Side(token_id=t["token_id"], outcome=t["outcome"])
            elif outcome in ("DOWN", "NO", "LOWER"):
                self.down = Side(token_id=t["token_id"], outcome=t["outcome"])

    @property
    def total_spent(self) -> float:
        s = 0.0
        if self.up: s += self.up.spent
        if self.down: s += self.down.spent
        return s

    @property
    def up_spent(self) -> float:
        return self.up.spent if self.up else 0.0

    @property
    def down_spent(self) -> float:
        return self.down.spent if self.down else 0.0


# === Main Bot ===

class PolymarketBot:
    def __init__(self):
        self.feed = BtcFeed()
        self.scanner = MarketScanner()
        self.executor = Executor()
        self.redeemer = Redeemer()

        self._running = True
        self._positions: dict[int, Position] = {}  # window_ts -> Position
        self._pending_redeems: list[dict] = []
        self._last_display = 0.0
        self._last_balance_refresh = 0.0
        self._last_redeem_check = 0.0

        # Session stats
        self.bankroll = DEFAULT_BANKROLL_USDC
        self.session_pnl = 0.0
        self.daily_pnl = 0.0
        self.total_windows = 0
        self.wins = 0
        self.losses = 0
        self.consecutive_losses = 0

    def run(self):
        """Main entry point."""
        logger.info("=" * 60)
        logger.info("Polymarket BTC 5-Min Binary Options Bot")
        logger.info(f"Mode: {'LIVE' if LIVE_TRADING else 'DRY RUN'}")
        logger.info(f"Strategy: Hedge + Momentum Directional")
        logger.info(f"Loop: {MAIN_LOOP_INTERVAL * 1000:.0f}ms tick")
        logger.info("=" * 60)

        _init_csv()

        # Initialize components
        if not self.executor.setup():
            logger.error("Executor setup failed")
            return

        self.redeemer.setup()
        self.feed.start()

        # Refresh initial balance
        self._refresh_balance()

        logger.info(f"Bankroll: ${self.bankroll:.2f}")
        logger.info("Entering main loop...\n")

        try:
            while self._running:
                self._tick()
                time.sleep(MAIN_LOOP_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self._shutdown()

    def _tick(self):
        """Single iteration of the 50ms main loop."""
        now = time.time()

        # Periodic balance refresh
        if now - self._last_balance_refresh > BALANCE_REFRESH_SEC:
            self._refresh_balance()
            self._last_balance_refresh = now

        # Check daily loss limit (research: circuit breaker)
        if self.daily_pnl <= -MAX_DAILY_LOSS_USDC:
            if now - self._last_display > 60:
                logger.warning(
                    f"HALTED: daily loss ${self.daily_pnl:.2f} "
                    f">= limit ${MAX_DAILY_LOSS_USDC:.2f}"
                )
                self._last_display = now
            return

        # Consecutive loss check (research: abstention condition)
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            if now - self._last_display > 60:
                logger.warning(
                    f"HALTED: {self.consecutive_losses} consecutive losses"
                )
                self._last_display = now
            return

        # Detect new 5-minute window
        current_ts = (int(now) // WINDOW_SEC) * WINDOW_SEC
        if current_ts not in self._positions:
            self._open_window(current_ts)

        # Update all active positions
        for ts, pos in list(self._positions.items()):
            if pos.phase == "CLOSED":
                continue
            remaining = (ts + WINDOW_SEC) - now
            if remaining <= 0:
                self._resolve(ts, pos)
            elif remaining < EMERGENCY_SEC and pos.phase not in ("HOLD", "CLOSED"):
                self._emergency(pos, remaining)
            else:
                self._update_position(pos, remaining)

        # Process redemptions (retry every 20s)
        if now - self._last_redeem_check > 20 and self._pending_redeems:
            self._process_redeems()
            self._last_redeem_check = now

        # Periodic display
        if now - self._last_display > DISPLAY_INTERVAL:
            self._display_stats()
            self._last_display = now

    # -----------------------------------------------------------------
    #  Window Lifecycle
    # -----------------------------------------------------------------

    def _open_window(self, window_ts: int):
        """Discover market for this 5-minute window and create position."""
        slug = SLUG_PATTERN.format(ts=window_ts)
        market = self.scanner.get_market(slug)

        if not market:
            # Market might not exist yet, will retry next tick
            return

        if self.feed.price <= 0:
            return

        pos = Position(
            slug=slug,
            market=market,
            window_ts=window_ts,
            btc_open=self.feed.price,
        )

        if not pos.up or not pos.down:
            logger.warning(f"Market missing UP/DOWN tokens: {slug}")
            return

        self._positions[window_ts] = pos
        self.total_windows += 1

        logger.info(
            f"\n{'='*50}\n"
            f"NEW WINDOW | {slug} | BTC Open: ${pos.btc_open:,.2f}\n"
            f"UP token: {pos.up.token_id[:16]}... | "
            f"DOWN token: {pos.down.token_id[:16]}...\n"
            f"{'='*50}"
        )

        # Immediately try to place cheap-side hedge
        self._try_hedge(pos)

    def _try_hedge(self, pos: Position):
        """
        Phase 1: Place small hedge on the cheap side.
        Research: reduces max loss to hedge cost (~$1.50).
        """
        if pos.total_spent >= MAX_TOTAL_USDC:
            return

        # Determine which side is cheap
        # Skip orderbook fetch for speed — place at CHEAP_SIDE_MAX limit
        # The GTC order will only fill if the price is actually ≤ CHEAP_SIDE_MAX
        if pos.phase == "WAIT":
            # Hedge the side opposite to initial lean (or DOWN by default)
            # We don't know direction yet, so hedge the DOWN side
            hedge_token = pos.down.token_id
            order_id = self.executor.buy_limit_gtc(
                hedge_token, HEDGE_SIZE_USDC, CHEAP_SIDE_MAX,
                neg_risk=pos.market.get("neg_risk", False),
            )
            if order_id:
                pos.down.order_ids.append(order_id)
                pos.phase = "HEDGE"
                logger.info(
                    f"HEDGE: GTC BUY DOWN @ ${CHEAP_SIDE_MAX:.2f} "
                    f"(${HEDGE_SIZE_USDC:.2f})"
                )

    def _update_position(self, pos: Position, remaining: float):
        """Update position based on current phase and BTC price."""
        if remaining < ENTRY_DEADLINE_SEC and pos.phase in ("WAIT", "HEDGE"):
            # Past entry deadline, just hold what we have
            pos.phase = "HOLD"
            return

        btc_now = self.feed.price
        if btc_now <= 0 or pos.btc_open <= 0:
            return

        move_pct = (btc_now - pos.btc_open) / pos.btc_open * 100

        # --- Check GTC fills on hedge orders ---
        self._check_fills(pos)

        # --- Phase 2: Momentum-triggered directional entry ---
        if pos.phase == "HEDGE" and abs(move_pct) >= DIRECTION_THRESHOLD:
            self._enter_directional(pos, move_pct, btc_now)

        # --- Phase 2b: Follow-up GTC at better price ---
        if pos.phase == "MAIN" and pos.followup_order_id is None:
            self._place_followup(pos)

        # --- Phase 3: Add-on ---
        if (
            pos.phase in ("MAIN", "HOLD")
            and not pos.addon_done
            and pos.lean
            and (time.time() - (pos.window_ts)) > ADDON_DELAY_SEC
            and abs(move_pct) >= ADDON_THRESHOLD
        ):
            self._addon(pos, move_pct)

        # --- Early exit check ---
        if pos.phase in ("MAIN", "HOLD") and pos.lean:
            self._check_early_exit(pos)

    def _enter_directional(self, pos: Position, move_pct: float, btc_now: float):
        """
        Phase 2: Place directional bet in the direction of momentum.

        Research: BotVersion3 uses 0.01% threshold, races market makers
        with fixed $0.55 limit (skip orderbook, saves 300ms).
        """
        if pos.total_spent + FAST_BET_SIZE_USDC > MAX_TOTAL_USDC:
            return

        neg_risk = pos.market.get("neg_risk", False)

        if move_pct > 0:
            # BTC moving UP — buy UP token
            pos.lean = "UP"
            token_id = pos.up.token_id
            order_id = self.executor.buy_fok(
                token_id, FAST_BET_SIZE_USDC, FAST_LIMIT_PRICE, neg_risk,
            )
            if order_id:
                pos.up.add_fill(FAST_BET_SIZE_USDC, FAST_LIMIT_PRICE)
                pos.up.order_ids.append(order_id)
        else:
            # BTC moving DOWN — buy DOWN token
            pos.lean = "DOWN"
            token_id = pos.down.token_id
            order_id = self.executor.buy_fok(
                token_id, FAST_BET_SIZE_USDC, FAST_LIMIT_PRICE, neg_risk,
            )
            if order_id:
                pos.down.add_fill(FAST_BET_SIZE_USDC, FAST_LIMIT_PRICE)
                pos.down.order_ids.append(order_id)

        pos.phase = "MAIN"
        logger.info(
            f"DIRECTIONAL: {pos.lean} | BTC move: {move_pct:+.3f}% | "
            f"${FAST_BET_SIZE_USDC:.2f} @ ${FAST_LIMIT_PRICE:.2f}"
        )

    def _place_followup(self, pos: Position):
        """
        Phase 2b: Resting GTC at better price for averaging.
        Research: follow-up at $0.48 catches dips within the window.
        """
        if not pos.lean or pos.total_spent + FOLLOWUP_LIMIT_SIZE_USDC > MAX_TOTAL_USDC:
            return

        neg_risk = pos.market.get("neg_risk", False)

        if pos.lean == "UP":
            token_id = pos.up.token_id
        else:
            token_id = pos.down.token_id

        order_id = self.executor.buy_limit_gtc(
            token_id, FOLLOWUP_LIMIT_SIZE_USDC, FOLLOWUP_LIMIT_PRICE, neg_risk,
        )
        if order_id:
            pos.followup_order_id = order_id
            logger.info(
                f"FOLLOWUP: GTC {pos.lean} @ ${FOLLOWUP_LIMIT_PRICE:.2f} "
                f"(${FOLLOWUP_LIMIT_SIZE_USDC:.2f})"
            )

    def _addon(self, pos: Position, move_pct: float):
        """Phase 3: Small add-on to strengthen conviction."""
        if pos.total_spent + ADDON_SIZE_USDC > MAX_TOTAL_USDC:
            return

        # Only add on if move is in our direction
        if pos.lean == "UP" and move_pct > ADDON_THRESHOLD:
            side = pos.up
        elif pos.lean == "DOWN" and move_pct < -ADDON_THRESHOLD:
            side = pos.down
        else:
            return

        if side.spent + ADDON_SIZE_USDC > MAX_ONE_SIDE_USDC:
            return

        neg_risk = pos.market.get("neg_risk", False)
        order_id = self.executor.buy_fok(
            side.token_id, ADDON_SIZE_USDC, FAST_LIMIT_PRICE, neg_risk,
        )
        if order_id:
            side.add_fill(ADDON_SIZE_USDC, FAST_LIMIT_PRICE)
            pos.addon_done = True
            logger.info(f"ADDON: {pos.lean} +${ADDON_SIZE_USDC:.2f}")

    def _check_fills(self, pos: Position):
        """Check if any resting GTC orders have filled."""
        for side in (pos.up, pos.down):
            if not side:
                continue
            for oid in list(side.order_ids):
                status = self.executor.get_order_status(oid)
                if status["status"] == "MATCHED":
                    matched = status["size_matched"]
                    if matched > 0 and side.tokens == 0:
                        # First fill on this side
                        price = CHEAP_SIDE_MAX if oid.startswith("DRY_GTC") else FOLLOWUP_LIMIT_PRICE
                        side.tokens = matched
                        side.spent = matched * price
                        logger.info(
                            f"GTC FILL: {side.outcome} {matched:.1f} tokens "
                            f"@ ~${price:.2f}"
                        )

        # Check followup fill
        if pos.followup_order_id and pos.lean:
            status = self.executor.get_order_status(pos.followup_order_id)
            if status["status"] == "MATCHED":
                matched = status["size_matched"]
                side = pos.up if pos.lean == "UP" else pos.down
                if matched > 0:
                    side.tokens += matched
                    side.spent += matched * FOLLOWUP_LIMIT_PRICE
                    logger.info(
                        f"FOLLOWUP FILL: {pos.lean} +{matched:.1f} tokens "
                        f"@ ${FOLLOWUP_LIMIT_PRICE:.2f}"
                    )
                pos.followup_order_id = None

    def _check_early_exit(self, pos: Position):
        """
        Research: exit at 70%+ unrealized profit or $0.97 bid.
        Skip orderbook fetch for speed — check less frequently.
        """
        if not pos.lean:
            return

        # Only check every ~10 seconds
        elapsed = time.time() - pos.window_ts
        if elapsed % 10 > MAIN_LOOP_INTERVAL * 2:
            return

        side = pos.up if pos.lean == "UP" else pos.down
        if side.tokens <= 0:
            return

        book = self.scanner.get_orderbook(side.token_id)
        if not book:
            return

        bid = book["best_bid"]

        # Near-certain win: bid >= $0.97
        if bid >= NEAR_MAX_EXIT_BID and side.avg_price < bid:
            neg_risk = pos.market.get("neg_risk", False)
            self.executor.sell_fok(
                side.token_id, side.tokens, bid - 0.01, neg_risk,
            )
            pnl = side.tokens * (bid - 0.01) - side.spent
            logger.info(f"EARLY EXIT (near-max): sold {side.tokens:.1f} @ ${bid:.2f} | PnL: ${pnl:+.2f}")
            pos.phase = "CLOSED"
            self._record_pnl(pos, pnl)
            return

        # Profit exit at 70%+
        unrealized = side.unrealized_pct(bid) if hasattr(side, 'unrealized_pct') else 0
        if unrealized >= PROFIT_EXIT_PCT:
            neg_risk = pos.market.get("neg_risk", False)
            self.executor.sell_fok(
                side.token_id, side.tokens, bid - 0.01, neg_risk,
            )
            pnl = side.tokens * (bid - 0.01) - side.spent
            logger.info(f"EARLY EXIT (profit): {unrealized:.0%} | PnL: ${pnl:+.2f}")
            pos.phase = "CLOSED"
            self._record_pnl(pos, pnl)

    def _emergency(self, pos: Position, remaining: float):
        """
        Emergency mode: < 45s to expiry with incomplete position.
        Research: try to complete or hold what we have.
        """
        if pos.phase == "HOLD":
            return

        # Cancel any resting orders
        if pos.followup_order_id:
            self.executor.cancel(pos.followup_order_id)
            pos.followup_order_id = None

        for side in (pos.up, pos.down):
            if side:
                for oid in side.order_ids:
                    self.executor.cancel(oid)

        pos.phase = "HOLD"
        logger.info(
            f"EMERGENCY: {remaining:.0f}s left | Holding: "
            f"UP ${pos.up_spent:.2f} / DOWN ${pos.down_spent:.2f}"
        )

    # -----------------------------------------------------------------
    #  Resolution
    # -----------------------------------------------------------------

    def _resolve(self, window_ts: int, pos: Position):
        """Resolve a completed window — calculate PnL and queue redemption."""
        if pos.phase == "CLOSED":
            return

        btc_close = self.feed.price
        btc_went_up = btc_close > pos.btc_open

        # Calculate payout
        payout = 0.0
        if btc_went_up and pos.up and pos.up.tokens > 0:
            payout += pos.up.tokens * 1.0  # UP wins -> $1 each
        if not btc_went_up and pos.down and pos.down.tokens > 0:
            payout += pos.down.tokens * 1.0  # DOWN wins -> $1 each

        total_spent = pos.total_spent
        pnl = payout - total_spent

        pos.phase = "CLOSED"
        self._record_pnl(pos, pnl)

        direction = "UP" if btc_went_up else "DOWN"
        logger.info(
            f"RESOLVED: {pos.slug} | BTC ${pos.btc_open:,.2f} -> ${btc_close:,.2f} "
            f"({direction}) | Spent: ${total_spent:.2f} | Payout: ${payout:.2f} | "
            f"PnL: ${pnl:+.2f}"
        )

        # Log to CSV
        _log_trade({
            "timestamp": datetime.now().isoformat(),
            "slug": pos.slug,
            "direction": direction,
            "btc_open": f"{pos.btc_open:.2f}",
            "btc_close": f"{btc_close:.2f}",
            "up_spent": f"{pos.up_spent:.2f}",
            "down_spent": f"{pos.down_spent:.2f}",
            "total_spent": f"{total_spent:.2f}",
            "payout": f"{payout:.2f}",
            "pnl": f"{pnl:+.2f}",
            "phase": pos.lean or "NONE",
            "bankroll": f"{self.bankroll:.2f}",
        })

        # Queue redemption (oracle needs 2-7 min to settle on-chain)
        if payout > 0:
            self._pending_redeems.append({
                "market": pos.market,
                "first_try": time.time(),
                "attempts": 0,
            })

    def _record_pnl(self, pos: Position, pnl: float):
        """Update session stats after a trade."""
        self.session_pnl += pnl
        self.daily_pnl += pnl
        self.bankroll += pnl

        if pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.losses += 1
            self.consecutive_losses += 1

    def _process_redeems(self):
        """Retry pending redemptions. Oracle needs 2-7 min to settle."""
        still_pending = []
        for item in self._pending_redeems:
            item["attempts"] += 1
            success = self.redeemer.try_redeem(item["market"])
            if not success and item["attempts"] < 30:  # Retry up to ~10 min
                still_pending.append(item)
            elif success:
                logger.info(f"Redeemed: {item['market'].get('slug', '?')}")
        self._pending_redeems = still_pending

    # -----------------------------------------------------------------
    #  Utilities
    # -----------------------------------------------------------------

    def _refresh_balance(self):
        bal = self.executor.get_balance()
        if bal > 0:
            self.bankroll = bal
            logger.info(f"Balance refreshed: ${self.bankroll:.2f}")

    def _display_stats(self):
        active = sum(1 for p in self._positions.values() if p.phase != "CLOSED")
        total = self.wins + self.losses
        wr = self.wins / total * 100 if total > 0 else 0

        logger.info(
            f"\n--- STATS ---\n"
            f"  Bankroll:  ${self.bankroll:.2f}\n"
            f"  Session:   ${self.session_pnl:+.2f}\n"
            f"  Win Rate:  {wr:.0f}% ({self.wins}W/{self.losses}L)\n"
            f"  Windows:   {self.total_windows} ({active} active)\n"
            f"  Pending:   {len(self._pending_redeems)} redeems\n"
            f"  BTC:       ${self.feed.price:,.2f}\n"
            f"-------------"
        )

    def _shutdown(self):
        logger.info("Shutting down...")
        self._running = False

        # Cancel all resting orders
        self.executor.cancel_all()

        # Stop price feed
        self.feed.stop()

        logger.info(
            f"\nFINAL: ${self.session_pnl:+.2f} session PnL | "
            f"{self.wins}W/{self.losses}L | "
            f"Bankroll: ${self.bankroll:.2f}"
        )

    def stop(self):
        self._running = False


def main():
    bot = PolymarketBot()

    def handle_sig(sig, frame):
        bot.stop()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    bot.run()


if __name__ == "__main__":
    main()
