"""
Configuration — Polymarket 5-Minute BTC Binary Options Bot.

Parameters calibrated from:
- Research paper findings (GMADL, Kelly, abstention rules, fee structure)
- BotVersion3 production results (+$9.50 session, 55%+ WR)
- gabagoolBot architecture (Gamma API, Safe redemption, temporal arb)
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ═══════════════════════════════════════════════════════════════
#  POLYMARKET API
# ═══════════════════════════════════════════════════════════════

POLYMARKET_PRIVATE_KEY    = os.environ.get("POLYGON_PRIVATE_KEY", "").strip()
POLYMARKET_API_KEY        = os.environ.get("POLYMARKET_API_KEY", "").strip()
POLYMARKET_API_SECRET     = os.environ.get("POLYMARKET_API_SECRET", "").strip()
POLYMARKET_API_PASSPHRASE = os.environ.get("POLYMARKET_API_PASSPHRASE", "").strip()
POLYMARKET_FUNDER_ADDRESS = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "").strip()
POLYMARKET_SIGNATURE_TYPE = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "2"))

CHAIN_ID   = int(os.environ.get("CHAIN_ID", "137"))
CLOB_HOST  = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"

LIVE_TRADING = os.environ.get("LIVE_TRADING", "false").lower() in ("true", "1", "yes")

# ═══════════════════════════════════════════════════════════════
#  MARKETS
# ═══════════════════════════════════════════════════════════════

WINDOW_SEC = 300  # 5-minute markets (ts % 300 == 0)
SLUG_PATTERN = "btc-updown-5m-{ts}"

# ═══════════════════════════════════════════════════════════════
#  PRICE FEED (Binance)
# ═══════════════════════════════════════════════════════════════

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
BINANCE_REST_URL = "https://api.binance.com/api/v3"

# ═══════════════════════════════════════════════════════════════
#  STRATEGY PARAMETERS
#  (calibrated from BotVersion3 production results)
# ═══════════════════════════════════════════════════════════════

# Phase 1 — Hedge (cheap side insurance, placed immediately at window open)
# Research: hedging reduces max loss from full position to hedge cost
HEDGE_SIZE_USDC   = 1.50   # Small hedge on the cheap side
CHEAP_SIDE_MAX    = 0.42   # Only hedge when token price < this

# Phase 2 — Directional bet (after momentum confirms direction)
# Research: momentum threshold filters noise from random walk
FAST_BET_SIZE_USDC = 2.50  # Primary directional entry (fixed, skip Kelly initially)
FAST_LIMIT_PRICE   = 0.55  # Fixed limit — skip orderbook fetch, saves ~300ms
DIRECTION_THRESHOLD = 0.01 # 0.01% BTC move triggers entry (race market makers)

# Phase 2b — Follow-up resting order (better average price if dip occurs)
FOLLOWUP_LIMIT_PRICE     = 0.48  # Resting GTC at better price
FOLLOWUP_LIMIT_SIZE_USDC = 1.50  # Follow-up size

# Phase 3 — Add-on (optional, strengthens conviction)
ADDON_SIZE_USDC   = 0.50
ADDON_DELAY_SEC   = 120    # After 120s
ADDON_THRESHOLD   = 0.20   # Need >0.20% move

# Position limits
MAX_TOTAL_USDC    = 5.00   # Max per asset per window
MAX_ONE_SIDE_USDC = 4.00   # Never over-concentrate

# ═══════════════════════════════════════════════════════════════
#  TIMING
# ═══════════════════════════════════════════════════════════════

ENTRY_DEADLINE_SEC  = 60   # Don't enter in last 60s of window
EMERGENCY_SEC       = 45   # Emergency mode at 45s to expiry
MAIN_LOOP_INTERVAL  = 0.05 # 50ms tick (research: sub-100ms mandatory)
DISPLAY_INTERVAL    = 30.0 # Stats display every 30s

# ═══════════════════════════════════════════════════════════════
#  EARLY EXIT
# ═══════════════════════════════════════════════════════════════

PROFIT_EXIT_PCT     = 0.70  # Sell at 70%+ unrealized profit
NEAR_MAX_EXIT_BID   = 0.97  # Exit when bid hits $0.97

# ═══════════════════════════════════════════════════════════════
#  KELLY CRITERION SIZING
#  (Research: Fractional Kelly mandatory for heavy-tail BTC)
# ═══════════════════════════════════════════════════════════════

KELLY_FRACTION    = 0.50    # Half-Kelly (research recommends 0.25-0.50)
MOMENTUM_SCALE    = 0.10    # Edge per 1% BTC move
MAX_MOMENTUM_EDGE = 0.15    # Cap at 65% win probability
MIN_BET_PCT       = 0.005   # 0.5% bankroll minimum
MAX_BET_PCT       = 0.05    # 5% bankroll maximum
MIN_BET_USDC      = 0.50    # Hard floor

# ═══════════════════════════════════════════════════════════════
#  RISK CONTROLS
#  (Research: 7-condition abstention circuit breaker)
# ═══════════════════════════════════════════════════════════════

MAX_DAILY_LOSS_USDC    = 50.0   # Halt after $50 daily loss
MAX_CONSECUTIVE_LOSSES = 7      # Halt after 7 consecutive losses
MAX_DRAWDOWN_PCT       = 0.15   # Halt at 15% drawdown from peak

# ═══════════════════════════════════════════════════════════════
#  FEE STRUCTURE (from research)
# ═══════════════════════════════════════════════════════════════

PEAK_TAKER_FEE = 0.0156  # 1.56% at $0.50 midpoint
MAKER_REBATE_PCT = 0.20  # 20% rebate for makers
TICK_SIZE = "0.01"

# ═══════════════════════════════════════════════════════════════
#  EXECUTION
# ═══════════════════════════════════════════════════════════════

ORDER_TIMEOUT_SEC = 20
FILL_TIMEOUT_SEC  = 30
API_RATE_LIMIT    = 10      # requests/sec

# ═══════════════════════════════════════════════════════════════
#  BALANCE & BANKROLL
# ═══════════════════════════════════════════════════════════════

BALANCE_REFRESH_SEC   = 300     # Refresh every 5 min
DEFAULT_BANKROLL_USDC = 50.0    # Fallback

# ═══════════════════════════════════════════════════════════════
#  POLYGON ON-CHAIN (for redemption)
# ═══════════════════════════════════════════════════════════════

POLYGON_RPCS = [
    os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com"),
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon.llamarpc.com",
]

CTF_ADDRESS  = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# ═══════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def taker_fee_at_price(price: float) -> float:
    """Dynamic taker fee: peaks at 1.56% at $0.50, 0% at extremes."""
    return PEAK_TAKER_FEE * (1.0 - abs(2.0 * price - 1.0))
