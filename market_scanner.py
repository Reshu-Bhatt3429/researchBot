"""
Market Scanner — Discovers BTC 5-min binary markets on Polymarket.

Uses Gamma API with predictable slug pattern (proven in gabagoolBot):
  slug = btc-updown-5m-{(now // 300) * 300}

This is the ONLY reliable way to find these markets.
The CLOB /markets endpoint does NOT support keyword search for these.
"""

import json
import logging
import requests

from config import GAMMA_HOST, CLOB_HOST, LOG_DIR

logger = logging.getLogger("scanner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(f"{LOG_DIR}/scanner.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(sh)


class MarketScanner:
    """Finds BTC binary markets by slug and fetches order books."""

    def __init__(self):
        self._session = requests.Session()
        self._cache: dict[str, dict] = {}

    def get_market(self, slug: str) -> dict | None:
        """
        Fetch market data from Gamma API by slug.

        Returns dict with keys: slug, question, condition_id, tokens,
        neg_risk, end_str — or None if not found.
        """
        if slug in self._cache:
            cached = self._cache[slug]
            if not cached.get("closed"):
                return cached

        try:
            r = self._session.get(
                f"{GAMMA_HOST}/events",
                params={"slug": slug},
                timeout=10,
            )
            r.raise_for_status()
            events = r.json()
        except Exception as e:
            logger.error(f"Gamma fetch error ({slug}): {e}")
            return None

        if not events:
            return None

        event = events[0]
        for mkt in event.get("markets", []):
            if mkt.get("closed"):
                continue

            try:
                token_ids = (
                    json.loads(mkt["clobTokenIds"])
                    if isinstance(mkt["clobTokenIds"], str)
                    else mkt["clobTokenIds"]
                )
                outcomes = (
                    json.loads(mkt["outcomes"])
                    if isinstance(mkt["outcomes"], str)
                    else mkt["outcomes"]
                )
            except Exception:
                continue

            tokens = [
                {"token_id": tid, "outcome": o}
                for tid, o in zip(token_ids, outcomes)
            ]
            if not tokens:
                continue

            end_str = mkt.get("endDate") or mkt.get("end_date_iso", "")

            market = {
                "slug": slug,
                "question": mkt.get("question", event.get("title", "")),
                "condition_id": (
                    mkt.get("conditionId") or mkt.get("condition_id", "")
                ),
                "tokens": tokens,
                "neg_risk": mkt.get("negRisk", False),
                "end_str": end_str,
            }
            self._cache[slug] = market
            logger.info(f"Found: {market['question']}")
            return market

        return None

    def get_orderbook(self, token_id: str) -> dict | None:
        """
        Fetch best bid/ask for a token from the CLOB.
        Returns {"best_bid": float, "best_ask": float, "spread": float}
        """
        try:
            r = self._session.get(
                f"{CLOB_HOST}/book",
                params={"token_id": token_id},
                timeout=5,
            )
            r.raise_for_status()
            book = r.json()
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            best_bid = max((float(b["price"]) for b in bids), default=0.0)
            best_ask = min((float(a["price"]) for a in asks), default=1.0)
            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": best_ask - best_bid,
            }
        except Exception as e:
            logger.warning(f"Orderbook error ({token_id[:12]}): {e}")
            return None

    def seconds_remaining(self, market: dict) -> float | None:
        """Seconds until a market expires."""
        end_str = market.get("end_str", "")
        if not end_str:
            return None
        try:
            from datetime import datetime, timezone
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (end_dt - now).total_seconds()
        except Exception:
            return None
