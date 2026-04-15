#!/usr/bin/env python3
"""
live/bridge.py — AllocatorRunner bridge for m1

Reads live state from Singularity / Divergence / Horizon, assembles
MarketState from yfinance, calls AllocatorRunner.step(), and writes
state/allocator_output.json every cycle.

Telegram alerts (via Astra Telepathica bot, same creds as Singularity):
  - Personal DM  : budget changes >10%, risk flags, strategy going absent
  - Alert channel: errors, solver non-optimal, watchdog triggers

Usage:
    python live/bridge.py             # single shot
    python live/bridge.py --loop 300  # run every 5 minutes

Path overrides (env vars):
    SINGULARITY_DIR   default: ~/strategies/singularity
    DIVERGENCE_DIR    default: ~/strategies/divergence
    HORIZON_DIR       default: ~/strategies/horizon
    ALLOCATOR_STATE   default: ./state/allocator_output.json
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yfinance as yf

from allocator.adapters import DivergenceNative, HorizonNative, SingularityNative
from allocator.runner import AllocatorRunner
from allocator.schemas import MarketState
from live.telegram import send_alert, send_personal

log = logging.getLogger("bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SINGULARITY_DIR = Path(os.getenv("SINGULARITY_DIR", str(Path.home() / "strategies" / "singularity")))
DIVERGENCE_DIR  = Path(os.getenv("DIVERGENCE_DIR",  str(Path.home() / "strategies" / "divergence")))
HORIZON_DIR     = Path(os.getenv("HORIZON_DIR",     str(Path.home() / "strategies" / "horizon")))
ALLOCATOR_STATE = Path(os.getenv("ALLOCATOR_STATE", "./state/allocator_output.json"))

STALE_SECONDS   = 600   # treat file/row as absent if older than 10 min
DEFAULT_SPREAD  = 0.001 # XAUUSD typical spread normalised

# Alert thresholds
_BUDGET_SWING_THRESHOLD = 0.10   # notify if any budget moves >10%
_MAX_ABSENT_CYCLES      = 3      # alert after N consecutive absent cycles


# ---------------------------------------------------------------------------
# Singularity reader
# ---------------------------------------------------------------------------

_SESSION_HOURS_UTC: dict[str, tuple[int, int]] = {
    "ASIA":         (22, 6),
    "ASIA_DEFAULT": (22, 6),
    "LONDON":       (7,  16),
    "LONDON_LONG":  (7,  16),
    "NY":           (13, 22),
}


def _current_session_name(hour_utc: int) -> str:
    for name, (start, end) in _SESSION_HOURS_UTC.items():
        if start < end:
            if start <= hour_utc < end:
                return name
        else:
            if hour_utc >= start or hour_utc < end:
                return name
    return "OFF"


def read_singularity() -> Optional[SingularityNative]:
    path = SINGULARITY_DIR / "state" / "session_map.json"
    data = _load_json(path, stale_seconds=STALE_SECONDS)
    if data is None:
        return None
    sessions: list[dict] = data.get("sessions", [])
    if not sessions:
        return None
    hour_utc = datetime.now(timezone.utc).hour
    session  = next(
        (s for s in sessions if s.get("name", "").upper() == _current_session_name(hour_utc).upper()),
        sessions[0],
    )
    shadow   = session.get("shadow_mode", True)
    win_rate = float(session.get("wr_pct", 0.0)) / 100.0
    kelly    = 0.0 if shadow else max(0.0, win_rate - 0.5) * 2.0
    return SingularityNative(
        kelly_fraction=kelly,
        win_rate=win_rate if win_rate > 0 else 0.5,
        spread=DEFAULT_SPREAD,
    )


# ---------------------------------------------------------------------------
# Divergence reader
# ---------------------------------------------------------------------------

def _reconstruct_gamma(confidence: float, divergence: float) -> float:
    if confidence < 0.6 or divergence > 0.4:
        return 0.0
    return 0.5 + (confidence - 0.6) * (0.5 / 0.4)


def read_divergence() -> Optional[DivergenceNative]:
    path = DIVERGENCE_DIR / "logs" / "consensus_log.jsonl"
    last = _last_jsonl_line(path, stale_seconds=STALE_SECONDS)
    if last is None:
        return None
    try:
        consensus  = str(last.get("consensus", "NEUTRAL"))
        confidence = float(last.get("confidence", 0.0))
        divergence = float(last.get("divergence", 1.0))
        return DivergenceNative(
            gamma=_reconstruct_gamma(confidence, divergence),
            regime=consensus,
            persona_divergence_score=divergence,
            spread=DEFAULT_SPREAD,
        )
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Horizon reader
# ---------------------------------------------------------------------------

def read_horizon() -> Optional[HorizonNative]:
    db_path = HORIZON_DIR / "logs" / "predictions.db"
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute("SELECT confidence, timestamp FROM predictions ORDER BY id DESC LIMIT 1")
        row = con.cursor().fetchone() or cur.fetchone()
        con.close()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    confidence, ts_str = row
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts_str)).total_seconds()
        if age > STALE_SECONDS:
            return None
    except (ValueError, TypeError):
        pass
    return HorizonNative(confidence=float(confidence), spread=DEFAULT_SPREAD)


# ---------------------------------------------------------------------------
# MarketState builder
# ---------------------------------------------------------------------------

def _safe_last(series) -> Optional[float]:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) > 0 else None


def build_market_state(timestamp: datetime, forecasts: list) -> MarketState:
    breadth = float(sum(1 for f in forecasts if f > 0) / max(len(forecasts), 1))
    try:
        raw = yf.download(
            ["^VIX", "SPY", "GLD", "TLT", "GC=F"],
            period="5d", interval="1h",
            progress=False, auto_adjust=True,
        )["Close"].ffill()

        vix_last   = _safe_last(raw["^VIX"]) if "^VIX" in raw else None
        volatility = min(1.0, (vix_last or 20.0) / 100.0)

        asset_cols  = [c for c in ["SPY", "GLD", "TLT"] if c in raw]
        if asset_cols:
            assets      = raw[asset_cols].pct_change().dropna()
            dispersion  = float(assets.iloc[-1].std()) if len(assets) > 0 else 0.05
            corr_vals   = assets.tail(20).corr().values
            upper_tri   = corr_vals[np.triu_indices_from(corr_vals, k=1)]
            correlation = float(np.clip(np.nanmean(upper_tri), -1.0, 1.0))
        else:
            dispersion, correlation = 0.05, 0.3

        trend = 0.0
        if "GC=F" in raw:
            gc = raw["GC=F"].dropna()
            if len(gc) >= 21:
                ema9  = gc.ewm(span=9,  adjust=False).mean()
                ema21 = gc.ewm(span=21, adjust=False).mean()
                trend = float(np.clip((ema9.iloc[-1] - ema21.iloc[-1]) / gc.iloc[-1], -1.0, 1.0))
    except Exception as exc:
        log.warning("market_state: yfinance failed (%s), using defaults", exc)
        volatility, dispersion, correlation, trend = 0.20, 0.05, 0.3, 0.0

    return MarketState(
        timestamp=timestamp, volatility=volatility, dispersion=dispersion,
        correlation=correlation, breadth=breadth, trend=trend,
        slippage=DEFAULT_SPREAD, liquidity=500.0,
    )


# ---------------------------------------------------------------------------
# Watchdog state
# ---------------------------------------------------------------------------

_prev_budgets:    dict[str, float] = {}
_absent_counts:   dict[str, int]   = {"singularity": 0, "divergence": 0, "horizon": 0}
_consecutive_errors = 0


def _check_alerts(result, sing, div, hor) -> None:
    global _prev_budgets, _absent_counts, _consecutive_errors

    # --- Solver health ---
    if result.solver_status not in ("optimal", "optimal_inaccurate"):
        msg = f"⚠️ ALLOCATOR solver={result.solver_status}\nBudgets: {result.budgets}"
        send_alert(msg)
        log.warning("solver non-optimal: %s", result.solver_status)

    # --- Risk flags ---
    for rf in result.risk_flags:
        if rf.flag.value in ("de_risk", "kill"):
            send_personal(
                f"🚨 ALLOCATOR {rf.flag.value.upper()} — {rf.strategy_id}\n"
                f"Reason: {rf.reason or 'threshold breach'}\n"
                f"Budget: {result.budgets.get(rf.strategy_id, 0):.3f}"
            )

    # --- Budget swing >10% ---
    if _prev_budgets:
        swings = [
            (sid, abs(result.budgets.get(sid, 0) - _prev_budgets.get(sid, 0)))
            for sid in result.budgets
        ]
        big = [(sid, delta) for sid, delta in swings if delta >= _BUDGET_SWING_THRESHOLD]
        if big:
            lines = "\n".join(f"  {sid}: {_prev_budgets.get(sid,0):.3f} → {result.budgets[sid]:.3f}" for sid, _ in big)
            send_personal(f"📊 ALLOCATOR REBALANCE\n{lines}\nCash: {result.cash:.3f} [{result.solver_status}]")

    _prev_budgets = dict(result.budgets)

    # --- Absent strategy counters ---
    for name, val in [("singularity", sing), ("divergence", div), ("horizon", hor)]:
        if val is None:
            _absent_counts[name] += 1
            if _absent_counts[name] == _MAX_ABSENT_CYCLES:
                send_personal(f"⚠️ ALLOCATOR: {name} absent for {_MAX_ABSENT_CYCLES} cycles — check strategy health")
        else:
            if _absent_counts[name] >= _MAX_ABSENT_CYCLES:
                send_personal(f"✅ ALLOCATOR: {name} recovered")
            _absent_counts[name] = 0


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

_runner = AllocatorRunner()


def run_step() -> None:
    global _consecutive_errors
    ts   = datetime.now(timezone.utc).replace(microsecond=0)
    sing = read_singularity()
    div  = read_divergence()
    hor  = read_horizon()

    log.info("inputs — singularity=%s divergence=%s horizon=%s",
             "ok" if sing else "absent",
             "ok" if div  else "absent",
             "ok" if hor  else "absent")

    forecasts = [
        sing["kelly_fraction"] if sing else 0.0,
        div["gamma"] * (1 if div and div["regime"] == "RISK_ON" else
                        -1 if div and div["regime"] == "RISK_OFF" else 0) if div else 0.0,
        hor["confidence"] if hor else 0.0,
    ]

    market_state = build_market_state(ts, forecasts)
    result       = _runner.step(ts, market_state, singularity=sing, divergence=div, horizon=hor)

    _check_alerts(result, sing, div, hor)
    _consecutive_errors = 0

    output = {
        "timestamp":       ts.isoformat(),
        "budgets":         result.budgets,
        "cash":            result.cash,
        "solver_status":   result.solver_status,
        "objective_value": result.objective_value,
        "risk_flags": [
            {"strategy_id": rf.strategy_id, "flag": rf.flag.value, "reason": rf.reason}
            for rf in result.risk_flags
        ],
    }

    ALLOCATOR_STATE.parent.mkdir(parents=True, exist_ok=True)
    ALLOCATOR_STATE.write_text(json.dumps(output, indent=2))

    log.info(
        "budgets — sing=%.3f div=%.3f hor=%.3f cash=%.3f [%s]",
        result.budgets.get("singularity", 0),
        result.budgets.get("divergence",  0),
        result.budgets.get("horizon",     0),
        result.cash,
        result.solver_status,
    )


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, stale_seconds: int) -> Optional[dict]:
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > stale_seconds:
        log.warning("%s: stale", path.name)
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _last_jsonl_line(path: Path, stale_seconds: int) -> Optional[dict]:
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > stale_seconds:
        log.warning("%s: stale", path.name)
        return None
    try:
        text = path.read_text().strip()
        return json.loads(text.split("\n")[-1]) if text else None
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=0,
                        help="Run every N seconds (0 = single shot)")
    args = parser.parse_args()

    if args.loop > 0:
        log.info("starting loop, interval=%ds", args.loop)
        while True:
            try:
                run_step()
            except Exception as exc:
                _consecutive_errors += 1
                log.error("step failed: %s", exc, exc_info=True)
                if _consecutive_errors in (1, 5, 10):
                    send_alert(
                        f"🚨 ALLOCATOR BRIDGE ERROR (#{_consecutive_errors})\n"
                        f"{type(exc).__name__}: {exc}"
                    )
            time.sleep(args.loop)
    else:
        run_step()
