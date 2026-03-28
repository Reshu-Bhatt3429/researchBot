"""
Auto-Redeemer — Converts resolved winning tokens back to USDC.

CRITICAL component missing from initial bot (fatal gap vs research).

On Polymarket, when a binary market resolves, winning tokens become
redeemable for $1.00 USDC each via the CTF contract's redeemPositions().

For proxy wallet (signature_type 2 = Gnosis Safe), tokens are held in the Safe.
We must route redeemPositions through Safe's execTransaction so
msg.sender = Safe (which holds the tokens), not the EOA signer.

Copied from gabagoolBot's proven production redeemer.
"""

import logging
from web3 import Web3
from eth_account import Account

from config import (
    POLYMARKET_PRIVATE_KEY, POLYMARKET_FUNDER_ADDRESS,
    POLYMARKET_SIGNATURE_TYPE, CHAIN_ID, POLYGON_RPCS,
    CTF_ADDRESS, USDC_ADDRESS, LOG_DIR,
)

logger = logging.getLogger("redeemer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(f"{LOG_DIR}/redeemer.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(sh)


# Minimal ABI for CTF redeemPositions
CTF_ABI = [
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "id", "type": "uint256"},
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "name": "payoutDenominator",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Gnosis Safe v1.3.0 ABI
SAFE_ABI = [
    {
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "operation", "type": "uint8"},
            {"name": "safeTxGas", "type": "uint256"},
            {"name": "baseGas", "type": "uint256"},
            {"name": "gasPrice", "type": "uint256"},
            {"name": "gasToken", "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "signatures", "type": "bytes"},
        ],
        "name": "execTransaction",
        "outputs": [{"name": "success", "type": "bool"}],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "nonce",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "operation", "type": "uint8"},
            {"name": "safeTxGas", "type": "uint256"},
            {"name": "baseGas", "type": "uint256"},
            {"name": "gasPrice", "type": "uint256"},
            {"name": "gasToken", "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "_nonce", "type": "uint256"},
        ],
        "name": "getTransactionHash",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
]


class Redeemer:
    """
    Redeems resolved Polymarket positions back to USDC.
    Routes through Gnosis Safe execTransaction for proxy wallets.
    """

    def __init__(self):
        self._w3 = None
        self._account = None
        self._ctf = None
        self._safe = None
        self._wallet_address = None
        self._use_safe = False

    def setup(self) -> bool:
        """Connect to Polygon and set up contract instances."""
        if not POLYMARKET_PRIVATE_KEY:
            logger.error("No private key configured")
            return False

        # Connect to Polygon (failover across RPCs)
        for rpc in POLYGON_RPCS:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                if w3.is_connected():
                    self._w3 = w3
                    logger.info(f"Connected to Polygon via {rpc}")
                    break
            except Exception:
                continue

        if not self._w3:
            logger.error("Failed to connect to any Polygon RPC")
            return False

        self._account = Account.from_key(POLYMARKET_PRIVATE_KEY)

        # Determine wallet mode
        if POLYMARKET_SIGNATURE_TYPE in (1, 2) and POLYMARKET_FUNDER_ADDRESS:
            self._wallet_address = Web3.to_checksum_address(
                POLYMARKET_FUNDER_ADDRESS
            )
            self._use_safe = True
            self._safe = self._w3.eth.contract(
                address=self._wallet_address, abi=SAFE_ABI
            )
            logger.info(
                f"Redeemer ready (Safe mode) | "
                f"Safe: {self._wallet_address} | "
                f"Signer: {self._account.address}"
            )
        else:
            self._wallet_address = self._account.address
            self._use_safe = False
            logger.info(f"Redeemer ready (EOA mode) | Wallet: {self._wallet_address}")

        self._ctf = self._w3.eth.contract(
            address=Web3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI,
        )
        return True

    def redeem(self, condition_id: str, neg_risk: bool = False) -> bool:
        """
        Redeem resolved positions for a given condition ID.
        Routes through Safe for proxy wallets.
        """
        if not self._w3 or not self._ctf:
            logger.warning("Redeemer not initialized")
            return False

        if not condition_id:
            return False

        # Normalize condition_id to bytes32
        if isinstance(condition_id, str):
            if not condition_id.startswith("0x"):
                condition_id = "0x" + condition_id
            cond_bytes = bytes.fromhex(condition_id[2:].zfill(64))
        else:
            cond_bytes = condition_id

        try:
            # Check if market has resolved on-chain
            payout_denom = self._ctf.functions.payoutDenominator(
                cond_bytes
            ).call()
            if payout_denom == 0:
                logger.info(f"Market not yet resolved: {condition_id[:18]}...")
                return False

            if self._use_safe:
                return self._redeem_via_safe(cond_bytes, condition_id)
            else:
                return self._redeem_direct(cond_bytes, condition_id)

        except Exception as e:
            logger.error(f"Redeem failed: {e}")
            return False

    def _redeem_via_safe(self, cond_bytes: bytes, condition_id: str) -> bool:
        """Redeem by routing redeemPositions through Gnosis Safe execTransaction."""
        # Encode inner redeemPositions calldata
        inner_data = self._ctf.functions.redeemPositions(
            Web3.to_checksum_address(USDC_ADDRESS),
            b"\x00" * 32,       # parentCollectionId
            cond_bytes,          # conditionId
            [1, 2],             # indexSets (both outcomes)
        )._encode_transaction_data()

        # Get Safe nonce
        safe_nonce = self._safe.functions.nonce().call()

        # Get transaction hash from Safe contract
        ctf_addr = Web3.to_checksum_address(CTF_ADDRESS)
        zero_addr = "0x0000000000000000000000000000000000000000"

        safe_tx_hash = self._safe.functions.getTransactionHash(
            ctf_addr, 0, inner_data, 0, 0, 0, 0,
            zero_addr, zero_addr, safe_nonce,
        ).call()

        # Sign with EOA (Safe owner)
        sig_obj = self._account.unsafe_sign_hash(safe_tx_hash)
        signature = (
            sig_obj.r.to_bytes(32, "big")
            + sig_obj.s.to_bytes(32, "big")
            + sig_obj.v.to_bytes(1, "big")
        )

        # Build outer execTransaction
        tx = self._safe.functions.execTransaction(
            ctf_addr, 0, inner_data, 0, 0, 0, 0,
            zero_addr, zero_addr, signature,
        ).build_transaction({
            "from": self._account.address,
            "nonce": self._w3.eth.get_transaction_count(self._account.address),
            "gas": 350_000,
            "gasPrice": self._w3.eth.gas_price,
            "chainId": CHAIN_ID,
        })

        # Sign and send
        signed = self._account.sign_transaction(tx)
        tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info(
            f"Safe redeem tx sent: {tx_hash.hex()[:20]}... "
            f"(condition: {condition_id[:18]}..., nonce: {safe_nonce})"
        )

        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        if receipt["status"] == 1:
            logger.info(f"Redeemed via Safe | Gas used: {receipt['gasUsed']}")
            return True
        else:
            logger.warning(f"Safe redeem tx reverted: {tx_hash.hex()[:20]}...")
            return False

    def _redeem_direct(self, cond_bytes: bytes, condition_id: str) -> bool:
        """Redeem directly from EOA (for non-proxy wallet setups)."""
        tx = self._ctf.functions.redeemPositions(
            Web3.to_checksum_address(USDC_ADDRESS),
            b"\x00" * 32,
            cond_bytes,
            [1, 2],
        ).build_transaction({
            "from": self._account.address,
            "nonce": self._w3.eth.get_transaction_count(self._account.address),
            "gas": 200_000,
            "gasPrice": self._w3.eth.gas_price,
            "chainId": CHAIN_ID,
        })

        signed = self._account.sign_transaction(tx)
        tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info(f"Redeem tx sent: {tx_hash.hex()[:20]}...")

        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
        if receipt["status"] == 1:
            logger.info(f"Redeemed | Gas used: {receipt['gasUsed']}")
            return True
        else:
            logger.warning(f"Redeem tx reverted: {tx_hash.hex()[:20]}...")
            return False

    def try_redeem(self, market: dict) -> bool:
        """Convenience: try to redeem a market's positions. Safe to call pre-resolution."""
        condition_id = market.get("condition_id", "")
        neg_risk = market.get("neg_risk", False)
        if not condition_id:
            return False
        return self.redeem(condition_id, neg_risk)
