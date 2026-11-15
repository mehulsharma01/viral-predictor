"""Pipeline: measure -> rules -> (Claude checklist) -> scores -> report dict."""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import rules
from .extract import extract, keyframes

DATA = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = DATA / "model.json"


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def features(report: dict) -> dict:
    """Numbers the calibrated model learns from. Keep names stable: they go in outcomes.csv."""
    m = report["measured"]
    ai = report.get("ai") or {}
    tail_share = (m["static_tail_seconds"] or 0) / m["duration"]
    return {
        "action_start": min(m["action_start"] if m["action_start"] is not None else 6.0, 6.0),
        "early_flashes": sum(1 for t in m["flashes"] if t < 5),
        "hook_shots": m["hook_shots"],
        "static_tail_share": round(tail_share, 3),
        "quiet_start": int(m["quiet_start"]),
        "duration": m["duration"],
        "ai_hook_score": ai.get("hook_score", -1),
        "ai_movement_1s": int(ai["movement_in_first_second"]) if ai else -1,
        "ai_curiosity_gap": int(ai["curiosity_gap"]) if ai else -1,
        "ai_share_trigger": int(bool(ai.get("share_trigger"))) if ai else -1,
        "ai_original": {"yes": 1, "no": 0}.get(ai.get("original_footage"), -1) if ai else -1,
    }


def _calibrated_skip(feats: dict) -> tuple[float | None, int]:
    if not MODEL_PATH.exists():
        return None, 0
    model = json.loads(MODEL_PATH.read_text())
    sk = model.get("skip_model")
    if not sk:
        return None, model.get("n", 0)
    z = sk["intercept"] + sum(sk["coef"].get(k, 0) * (feats.get(k, 0) - sk["mean"].get(k, 0)) / (sk["std"].get(k, 1) or 1)
                              for k in sk["coef"])
    return 1 / (1 + math.exp(-z)), model.get("n", 0)


def _logged_count() -> int:
    f = DATA / "outcomes.csv"
    return max(0, len(f.read_text().strip().splitlines()) - 1) if f.exists() else 0


def analyze(path: str, use_ai: bool = True, context: str = "", out_dir: str | None = None,
            transcribe: bool = True) -> dict:
    name = Path(path).stem
    out = Path(out_dir or Path("reports") / name)
    out.mkdir(parents=True, exist_ok=True)

    sig = extract(path, transcribe=transcribe)
    params = rules.load_params(str(MODEL_PATH))
    found = rules.find(sig)
    curve = rules.predict_retention(sig, params)
    act = rules.action_start(sig)
    tail = rules.static_tail(sig)
    frames = keyframes(path, str(out / "frames"))

    measured = {
        "duration": sig.duration,
        "cuts": sig.cuts,
        "flashes": sig.flashes,
        "shots": [{"start": s.start, "end": s.end, "motion": s.motion} for s in sig.shots],
        "action_start": act,
        "hook_shots": sum(1 for s in sig.shots if act is not None and s.end <= act + 0.01),
        "static_tail_seconds": round(tail[1] - tail[0], 2) if tail else 0,
        "quiet_start": rules.quiet_start(sig),
        "audio_db": list(zip(sig.audio_times, sig.audio_db)),
        "transcript": sig.transcript,
    }

    ai = None
    ai_error = None
    if use_ai:
        try:
            from .judge import judge
            j = judge(frames, {k: measured[k] for k in
                               ("duration", "cuts", "flashes", "shots", "action_start",
                                "static_tail_seconds", "quiet_start", "transcript")}, context)
            ai = j.model_dump()
            for i in j.issues:
                found.append(rules.Finding("ai", i.start, i.end, i.severity, i.problem, "Claude, from the frames", i.fix, "ai"))
        except Exception as e:  # report still works without the AI layer
            ai_error = f"{type(e).__name__}: {e}"

    content_end = tail[0] if tail else sig.duration
    ret1, ret3 = rules.at(curve, 1), rules.at(curve, 3)
    ret_end_content = rules.at(curve, content_end)
    report = {
        "video": str(path), "name": name, "measured": measured, "ai": ai, "ai_error": ai_error,
        "findings": rules.findings_dicts(found),
    }
    feats = features(report)
    cal_skip, n_cal = _calibrated_skip(feats)
    prior_skip = 1 - ret3
    skip = cal_skip if cal_skip is not None else prior_skip

    hook = _clamp(((1 - skip) - 0.15) / 0.55) * 100
    if ai:
        hook = 0.6 * hook + 0.4 * ai["hook_score"] * 10
    hold = _clamp((ret_end_content / max(ret3, 1e-6)) / 0.6) * 100
    parts = {"hook": (hook, 0.45), "hold": (hold, 0.20)}
    if ai:
        parts["value"] = (ai["value_score"] * 10, 0.15)
        share = 50 * bool(ai["share_trigger"]) + 30 * bool(ai["save_trigger"]) + 20 * (ai["original_footage"] == "yes")
        parts["share"] = (share, 0.20)
    overall = sum(v * w for v, w in parts.values()) / sum(w for _, w in parts.values())
    notes = []
    if ai and ai["original_footage"] == "no":
        overall = min(overall, 60)
        notes.append("Footage looks reposted from other creators. Instagram shows reused clips to fewer non-followers, so the score is capped at 60.")
    highs = [f for f in report["findings"] if f["severity"] == 3]
    if not ai:
        notes.append("Measured timing only. Hook text, share reason and originality were not checked; run with Claude before posting.")
    verdict = ("Timing passes; run the Claude check before posting" if not ai and overall >= 70 and not highs else
               "Post it" if overall >= 70 and not highs else
               "Fix the high-severity items first" if overall >= 50 else "Re-edit before posting")

    report.update({
        "predicted": {
            "skip_rate": round(skip * 100, 1),
            "skip_rate_source": "calibrated on your reels" if cal_skip is not None else "prior rules (not yet calibrated)",
            "retention_1s": round(ret1 * 100, 1),
            "retention_3s": round((1 - skip) * 100, 1),
            "retention_end_of_content": round(ret_end_content * 100, 1),
            "curve": curve,
        },
        "scores": {k: round(v) for k, (v, _) in parts.items()} | {"overall": round(overall)},
        "verdict": verdict,
        "notes": notes,
        "calibration": {"reels_logged": _logged_count(), "calibrated": cal_skip is not None},
        "features": feats,
        "frames": frames,
    })
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    from .report import write_html
    write_html(report, out / "report.html")
    return report
