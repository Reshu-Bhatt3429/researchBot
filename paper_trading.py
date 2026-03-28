"""
Paper Trading Engine with SQLite database.

Research alpha:
- Logs all synthetic executions for statistical validation
- Tracks portfolio state, P&L, and position history
- Enables iterative parameter tuning without risking capital
- Schema matches research's PostgreSQL design but uses SQLite for simplicity
"""

import sqlite3
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from signal_engine import Signal, Direction

logger = logging.getLogger(__name__)


@dataclass
class PaperTrader:
    """
    Simulates Polymarket execution with realistic fee modeling.

    Matches the paper_trades / portfolio schema from research
    (OpenClaw polymarket-autopilot).
    """
    config: Config
    db_path: str = "paper_trades.db"
    _conn: Optional[sqlite3.Connection] = None

    def __post_init__(self):
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create tables matching research schema."""
        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,
                market_name TEXT,
                strategy TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                fee REAL,
                pnl REAL,
                signal_confidence REAL,
                signal_edge REAL,
                signal_features TEXT,
                btc_open_price REAL,
                btc_close_price REAL,
                resolution TEXT,
                timestamp REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_value REAL,
                cash REAL,
                positions TEXT,
                win_count INTEGER,
                loss_count INTEGER,
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brier_score REAL,
                log_loss REAL,
                win_rate REAL,
                roi REAL,
                total_trades INTEGER,
                bankroll REAL,
                timestamp REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS abstentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reason TEXT,
                signal_confidence REAL,
                signal_edge REAL,
                btc_price REAL,
                timestamp REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        self._conn.commit()

    def simulate_trade(
        self,
        signal: Signal,
        size_usd: float,
        entry_price: float,
        btc_open_price: float,
        btc_close_price: float,
        market_id: str = "",
        market_name: str = "BTC 5-min Up/Down",
    ) -> dict:
        """
        Simulate a paper trade with realistic Polymarket fee modeling.

        Returns trade result dict with P&L.
        """
        # Determine binary outcome
        btc_went_up = btc_close_price > btc_open_price

        if signal.direction == Direction.UP:
            won = btc_went_up
        else:
            won = not btc_went_up

        # Calculate P&L with fees
        fee = self.config.taker_fee_at_price(entry_price) * size_usd
        if self.config.use_maker_orders:
            fee = self.config.effective_maker_fee(entry_price) * size_usd

        if won:
            # Binary payout: $1 per share, cost was entry_price per share
            shares = size_usd / entry_price
            gross_pnl = shares * (1.0 - entry_price)  # Win: receive $1 - cost
            pnl = gross_pnl - fee
            exit_price = 1.0
            resolution = "WIN"
        else:
            # Lose entire position
            pnl = -size_usd - fee
            exit_price = 0.0
            resolution = "LOSS"

        # Record trade
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO paper_trades
            (market_id, market_name, strategy, direction, entry_price,
             exit_price, quantity, fee, pnl, signal_confidence, signal_edge,
             signal_features, btc_open_price, btc_close_price, resolution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_id,
                market_name,
                "gmadl_composite",
                signal.direction.value,
                entry_price,
                exit_price,
                size_usd,
                fee,
                pnl,
                signal.confidence,
                signal.edge,
                json.dumps(signal.features),
                btc_open_price,
                btc_close_price,
                resolution,
            ),
        )
        self._conn.commit()

        result = {
            "won": won,
            "pnl": pnl,
            "fee": fee,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size_usd": size_usd,
            "btc_open": btc_open_price,
            "btc_close": btc_close_price,
            "direction": signal.direction.value,
            "resolution": resolution,
        }

        logger.info(
            f"Paper trade: {resolution} | {signal.direction.value} | "
            f"BTC {btc_open_price:.2f} -> {btc_close_price:.2f} | "
            f"PnL: ${pnl:+.4f} | Fee: ${fee:.4f}"
        )

        return result

    def record_abstention(
        self, reason: str, confidence: float, edge: float, btc_price: float
    ):
        """Log when the bot decides not to trade (for analysis)."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO abstentions (reason, signal_confidence, signal_edge, btc_price)
            VALUES (?, ?, ?, ?)
            """,
            (reason, confidence, edge, btc_price),
        )
        self._conn.commit()

    def record_portfolio_snapshot(
        self, total_value: float, cash: float, positions: dict, wins: int, losses: int
    ):
        """Periodic portfolio state snapshot."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO portfolio (total_value, cash, positions, win_count, loss_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (total_value, cash, json.dumps(positions), wins, losses),
        )
        self._conn.commit()

    def record_model_metrics(self, metrics: dict):
        """Log model health metrics (Brier, LogLoss, etc.)."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_metrics
            (brier_score, log_loss, win_rate, roi, total_trades, bankroll)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                metrics.get("brier_score"),
                metrics.get("log_loss"),
                metrics.get("win_rate"),
                metrics.get("roi"),
                metrics.get("total_trades"),
                metrics.get("bankroll"),
            ),
        )
        self._conn.commit()

    def get_trade_history(self, limit: int = 50) -> list:
        """Fetch recent paper trades."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM paper_trades ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_summary_stats(self) -> dict:
        """Compute aggregate paper trading statistics."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT COUNT(*) as n FROM paper_trades")
        total = cursor.fetchone()["n"]

        if total == 0:
            return {"total_trades": 0, "message": "No trades yet"}

        cursor.execute(
            "SELECT COUNT(*) as n FROM paper_trades WHERE resolution='WIN'"
        )
        wins = cursor.fetchone()["n"]

        cursor.execute("SELECT SUM(pnl) as total_pnl FROM paper_trades")
        total_pnl = cursor.fetchone()["total_pnl"] or 0

        cursor.execute("SELECT SUM(fee) as total_fees FROM paper_trades")
        total_fees = cursor.fetchone()["total_fees"] or 0

        cursor.execute("SELECT AVG(signal_confidence) as avg_conf FROM paper_trades")
        avg_conf = cursor.fetchone()["avg_conf"] or 0

        cursor.execute("SELECT COUNT(*) as n FROM abstentions")
        total_abstentions = cursor.fetchone()["n"]

        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "net_pnl": total_pnl,
            "avg_confidence": avg_conf,
            "total_abstentions": total_abstentions,
            "roi": total_pnl / self.config.initial_bankroll,
        }

    def close(self):
        if self._conn:
            self._conn.close()
