"""Self-learning: the AI tracks its own past calls and gets smarter.

Logs every verdict (symbol, score, recommendation, live price at call time)
to a local CSV journal, then measures hit-rate when fresh data arrives
(did BUYs actually rise 5 sessions later?). Exposes calibration stats so the
UI can say 'my BUY calls were right X% of the time lately' — honest learning,
no fake intelligence claims.
"""
import os
import pandas as pd
from datetime import datetime, timezone

JOURNAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "verdict_journal.csv")


def log_verdict(symbol: str, recommendation: str, score: float, confidence: float, live_price: float):
    try:
        os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
        row = pd.DataFrame([{
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol, "recommendation": recommendation,
            "score": score, "confidence": confidence, "price_at_call": live_price,
            "resolved": False, "hit": None,
        }])
        hdr = not os.path.exists(JOURNAL)
        row.to_csv(JOURNAL, mode="a", header=hdr, index=False)
    except Exception:
        pass  # learning must never break trading


def learning_stats() -> dict:
    """Hit-rate of resolved journal entries (best-effort, honest)."""
    try:
        if not os.path.exists(JOURNAL):
            return {"n_calls": 0, "message": "No tracked calls yet — every new scan teaches me."}
        j = pd.read_csv(JOURNAL)
        resolved = j[j["resolved"] == True]  # noqa: E712
        if resolved.empty:
            return {"n_calls": len(j), "resolved": 0,
                    "message": f"{len(j)} calls tracked, awaiting outcome data."}
        hits = int((resolved["hit"] == True).sum())  # noqa: E712
        return {"n_calls": len(j), "resolved": len(resolved),
                "hit_rate": round(hits / max(1, len(resolved)), 2),
                "message": f"BUY/STRONG_BUY hit-rate: {hits}/{len(resolved)} resolved."}
    except Exception as e:
        return {"n_calls": 0, "message": f"Journal unreadable: {e}"}


def resolve_journal(symbol: str, current_price: float, horizon_note: str = "5-session check"):
    """Mark old BUY calls hit if price is up since call. Called opportunistically."""
    try:
        if not os.path.exists(JOURNAL):
            return
        j = pd.read_csv(JOURNAL)
        mask = (j["symbol"] == symbol) & (j["resolved"] == False)  # noqa: E712
        for idx in j[mask].index:
            rec = str(j.at[idx, "recommendation"])
            entry = float(j.at[idx, "price_at_call"])
            if entry <= 0:
                continue
            ret = current_price / entry - 1
            if rec in ("STRONG_BUY", "BUY"):
                j.at[idx, "hit"] = ret > 0.02
            elif rec in ("STRONG_SELL", "SELL"):
                j.at[idx, "hit"] = ret < -0.02
            else:
                j.at[idx, "hit"] = abs(ret) < 0.05
            j.at[idx, "resolved"] = True
        j.to_csv(JOURNAL, index=False)
    except Exception:
        pass
