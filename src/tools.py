"""
The two required tools.

Both are deterministic Python. The LLM never computes a number - it only
explains numbers these functions return. Every figure is labelled with its
source (statement or live market) because a finance answer that mixes the
two without saying so is wrong even when the arithmetic is right.
"""
from __future__ import annotations

import math

from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import yfinance as yf
from langchain_core.tools import tool

from src.ingest import Holding

# ---------------------------------------------------------------------------
# Session state. The agent must not be asked to pass an entire portfolio
# through a tool argument, so the app sets it once after ingestion.
# ---------------------------------------------------------------------------
_HOLDINGS: list[Holding] = []
_PRICES: dict[str, float] = {}
_PRICE_ERRORS: dict[str, str] = {}

CONCENTRATION_THRESHOLD = 25.0     # percent of portfolio value in one position
LOSS_THRESHOLD = -10.0             # percent


def set_session(holdings: list[Holding], prices: dict[str, float], errors: dict[str, str] | None = None) -> None:
    global _HOLDINGS, _PRICES, _PRICE_ERRORS
    _HOLDINGS = holdings
    _PRICES = prices
    _PRICE_ERRORS = errors or {}


def has_session() -> bool:
    return bool(_HOLDINGS)


def fetch_prices(symbols):
    """Return ({symbol: price}, {symbol: reason}) for the given tickers.

    Yahoo's quoteSummary endpoint (used by Ticker.info and fast_info) now
    rejects unauthenticated requests and yields NaN, so the last available
    daily close from the chart endpoint is used instead. This function never
    raises: a symbol that cannot be priced is reported in the errors dict and
    is excluded from every downstream total.
    """
    prices, errors = {}, {}
    uniq = list(dict.fromkeys(symbols))
    if not uniq:
        return prices, errors

    try:
        frame = yf.download(uniq, period="5d", interval="1d", auto_adjust=False,
                            progress=False, group_by="ticker", threads=True)
    except Exception:
        frame = None

    def from_frame(sym):
        if frame is None or getattr(frame, "empty", True):
            return None
        try:
            col = frame["Close"] if len(uniq) == 1 else frame[sym]["Close"]
            col = col.dropna()
            return float(col.iloc[-1]) if len(col) else None
        except Exception:
            return None

    def from_history(sym):
        try:
            hist = yf.Ticker(sym).history(period="5d", auto_adjust=False)
            if "Close" not in hist:
                return None
            closes = hist["Close"].dropna()
            return float(closes.iloc[-1]) if len(closes) else None
        except Exception:
            return None

    for sym in uniq:
        px = from_frame(sym)
        if px is None:
            px = from_history(sym)
        if px is None or not math.isfinite(px) or px <= 0:
            errors[sym] = "no market data returned"
        else:
            prices[sym] = px
    return prices, errors

# ---------------------------------------------------------------------------
# Result shapes. These are plain dataclasses rather than dicts so that a typo
# in a field name fails immediately instead of silently rendering as blank in
# the UI. Fields are Optional where a holding could not be priced - that case
# is reported, never filled in with a zero.
# ---------------------------------------------------------------------------
@dataclass
class Position:
    symbol: str
    name: str
    sector: str
    quantity: float
    buy_price: float
    current_price: float | None
    invested: float
    current_value: float | None
    gain_absolute: float | None
    gain_percent: float | None
    weight_percent: float | None
    priced: bool


@dataclass
class Analysis:
    positions: list[Position]
    total_invested: float
    total_value: float
    total_gain: float
    total_gain_percent: float
    concentration_flags: list[str]
    loss_flags: list[str]
    sector_weights: dict[str, float]
    unpriced: list[str]
    threshold: float
    generated_at: str


def compute_analysis(threshold: float = CONCENTRATION_THRESHOLD) -> Analysis:
    """Pure arithmetic over the parsed holdings. No model involved."""
    positions: list[Position] = []
    total_invested = 0.0
    total_value = 0.0
    unpriced: list[str] = []

    for h in _HOLDINGS:
        invested = h.quantity * h.buy_price
        total_invested += invested
        price = _PRICES.get(h.symbol)
        if price is None:
            unpriced.append(h.symbol)
            positions.append(Position(
                h.symbol, h.name, h.sector, h.quantity, h.buy_price, None,
                round(invested, 2), None, None, None, None, False,
            ))
            continue
        value = h.quantity * price
        total_value += value
        positions.append(Position(
            h.symbol, h.name, h.sector, h.quantity, h.buy_price, round(price, 2),
            round(invested, 2), round(value, 2),
            round(value - invested, 2),
            round(((value - invested) / invested) * 100, 2) if invested else 0.0,
            None, True,
        ))

    # Weights are of PRICED value only, and that is stated in the output.
    for p in positions:
        if p.priced and total_value > 0:
            p.weight_percent = round((p.current_value / total_value) * 100, 2)

    sector_weights: dict[str, float] = {}
    for p in positions:
        if p.priced and p.weight_percent is not None:
            sector_weights[p.sector] = round(sector_weights.get(p.sector, 0.0) + p.weight_percent, 2)

    concentration_flags = [
        f"{p.name} ({p.symbol}) is {p.weight_percent}% of the portfolio, above the {threshold}% threshold"
        for p in positions if p.weight_percent is not None and p.weight_percent > threshold
    ]
    loss_flags = [
        f"{p.name} ({p.symbol}) is down {abs(p.gain_percent)}% against purchase price"
        for p in positions if p.gain_percent is not None and p.gain_percent <= LOSS_THRESHOLD
    ]

    priced_invested = sum(p.invested for p in positions if p.priced)
    total_gain = round(total_value - priced_invested, 2)

    return Analysis(
        positions=positions,
        total_invested=round(total_invested, 2),
        total_value=round(total_value, 2),
        total_gain=total_gain,
        total_gain_percent=round((total_gain / priced_invested) * 100, 2) if priced_invested else 0.0,
        concentration_flags=concentration_flags,
        loss_flags=loss_flags,
        sector_weights=sector_weights,
        unpriced=unpriced,
        threshold=threshold,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


# ---------------------------------------------------------------------------
# Tool 1 (required): portfolio analysis - runs automatically on ingestion
# ---------------------------------------------------------------------------
@tool
def analyze_portfolio(concentration_threshold_pct: float = 25.0) -> str:
    """Analyse the uploaded portfolio statement with deterministic arithmetic.

    Computes, for every holding in the statement the user just uploaded:
      - invested amount in INR (quantity x purchase price, from the STATEMENT)
      - current value in INR (quantity x latest close, from LIVE MARKET DATA)
      - gain or loss in INR and as a percentage
      - each position's weight as a PERCENTAGE of total current portfolio value
    Flags any position whose weight exceeds concentration_threshold_pct
    (percentage, default 25.0) and any position down 10 percent or more.

    All money values are INR. All weights and gains are percentages, not
    absolute amounts. Returns plain text; it does not give investment advice.
    """
    if not has_session():
        return "No portfolio has been uploaded yet, so there is nothing to analyse."

    a = compute_analysis(concentration_threshold_pct)
    lines = [
        f"PORTFOLIO ANALYSIS (generated {a.generated_at})",
        f"Total invested (statement): INR {a.total_invested:,.2f}",
        f"Current value (live prices): INR {a.total_value:,.2f}",
        f"Overall gain/loss: INR {a.total_gain:,.2f} ({a.total_gain_percent:+.2f}%)",
        "",
        "PER HOLDING:",
    ]
    for p in a.positions:
        if not p.priced:
            lines.append(f"- {p.name} ({p.symbol}): invested INR {p.invested:,.2f}. No live price available.")
            continue
        lines.append(
            f"- {p.name} ({p.symbol}): {p.quantity:g} shares, bought at INR {p.buy_price:,.2f}, "
            f"now INR {p.current_price:,.2f}. Gain/loss INR {p.gain_absolute:,.2f} "
            f"({p.gain_percent:+.2f}%). Weight {p.weight_percent}% of portfolio."
        )

    lines.append("")
    if a.concentration_flags:
        lines.append(f"CONCENTRATION FLAGS (threshold {a.threshold}%):")
        lines += [f"! {f}" for f in a.concentration_flags]
    else:
        lines.append(f"No position exceeds the {a.threshold}% concentration threshold.")

    if a.loss_flags:
        lines.append("")
        lines.append("UNDERPERFORMANCE FLAGS (down 10% or more):")
        lines += [f"! {f}" for f in a.loss_flags]

    if a.unpriced:
        lines.append("")
        lines.append("Could not price: " + ", ".join(a.unpriced) +
                     ". These are excluded from value and weight calculations.")

    if a.sector_weights:
        lines.append("")
        lines.append("SECTOR WEIGHTS: " + ", ".join(
            f"{k} {v}%" for k, v in sorted(a.sector_weights.items(), key=lambda x: -x[1])
        ))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2 (required): live market data
# ---------------------------------------------------------------------------
@tool
def get_market_data(ticker: str) -> str:
    """Look up the CURRENT market price and recent movement for one ticker symbol.

    Use this when the user asks about today's price, recent movement, or news
    for a specific stock. The ticker must be an exchange symbol such as
    'INFY.NS' for the Indian NSE or 'AAPL' for the US market.

    Returns the latest close price, the previous close, and the day's change as
    a percentage. Prices for .NS symbols are in INR; US symbols are in USD, and
    the currency is stated in the output. This is LIVE market data, not a figure
    from the uploaded statement.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return "No ticker was supplied."
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="5d")
        if hist is None or hist.empty:
            return (f"No market data was found for '{symbol}'. Check the symbol - "
                    f"Indian listings usually need a .NS or .BO suffix.")
        closes = hist["Close"].dropna()
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) > 1 else last
        change = ((last - prev) / prev * 100) if prev else 0.0
        currency = "INR" if symbol.endswith((".NS", ".BO")) else "USD"
        name = symbol
        try:
            name = t.info.get("shortName") or symbol
        except Exception:
            pass
        return (f"LIVE MARKET DATA for {name} ({symbol}): "
                f"latest close {currency} {last:,.2f}, previous close {currency} {prev:,.2f}, "
                f"change {change:+.2f}%. Source: live market feed, not the uploaded statement.")
    except Exception as e:
        return f"Could not fetch live data for '{symbol}': {str(e)[:120]}"


TOOLS = [analyze_portfolio, get_market_data]
