"""Step 2a: rule checks on the measured signals, plus a predicted retention curve.

Every finding carries a time range and the evidence (numbers) behind it, so it
can be checked against the real Instagram retention graph after posting.

The weights below are PRIORS (starting guesses from short-form video practice).
`calibrate.py` replaces them with weights fitted on your own posted reels.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

from .extract import Signals

PRIORS = {
    "base_hazard": 0.09,        # share of remaining viewers lost per second on a "clean" reel
    "early_boost": 1.6,         # extra loss in the first seconds (everyone decides fast)
    "early_tau": 1.5,           # seconds over which the early boost fades
    "no_action_mult": 1.9,      # before the first sustained action / demo
    "flash_mult": 3.0,          # 0.4 s after a fade-to-white / washed-out frame
    "static_tail_mult": 2.2,    # static end card / nothing changing
    "quiet_audio_mult": 1.25,   # first second much quieter than the rest
}
ACTIVE_MOTION = 7.0     # within-shot motion that counts as "something is happening"
MIN_ACTION_RUN = 3.0    # seconds of continuous activity that count as the demo/payoff
STATIC_MOTION = 3.0


@dataclass
class Finding:
    id: str
    start: float
    end: float
    severity: int          # 1 low, 2 medium, 3 high
    title: str
    evidence: str
    fix: str
    source: str = "rules"  # rules | ai


def load_params(model_path: str | None = None) -> dict:
    p = dict(PRIORS)
    if model_path and Path(model_path).exists():
        p.update(json.loads(Path(model_path).read_text()).get("hazard", {}))
    return p


def action_start(sig: Signals) -> float | None:
    """Start of the first run of active shots lasting MIN_ACTION_RUN seconds."""
    run_start, run_len = None, 0.0
    for s in sig.shots:
        if s.motion >= ACTIVE_MOTION:
            if run_start is None:
                run_start, run_len = s.start, 0.0
            run_len += s.duration
            if run_len >= MIN_ACTION_RUN:
                return run_start
        else:
            run_start, run_len = None, 0.0
    return None


def static_tail(sig: Signals) -> tuple[float, float] | None:
    if not sig.shots:
        return None
    last = sig.shots[-1]
    if last.motion < STATIC_MOTION and last.duration >= 2.0:
        return last.start, sig.duration
    return None


def quiet_start(sig: Signals) -> bool:
    if len(sig.audio_db) < 4:
        return False
    rest = sorted(sig.audio_db[2:])
    median = rest[len(rest) // 2]
    return sig.audio_db[0] < median - 6


def find(sig: Signals) -> list[Finding]:
    f: list[Finding] = []
    act = action_start(sig)
    hook_shots = [s for s in sig.shots if act is None or s.end <= act + 0.01]

    if act is None:
        f.append(Finding("no_action", 0, sig.duration, 3, "Nothing sustained happens on screen",
                         "No stretch of 3+ seconds with real movement was found.",
                         "Show the product or result being used, starting in the first second."))
    elif act > 1.0:
        sev = 3 if act >= 2.5 else 2
        detail = ", ".join(f"{s.start:.1f}-{s.end:.1f}s (motion {s.motion})" for s in hook_shots)
        f.append(Finding("late_action", 0, act, sev,
                         f"The demo starts at {act:.1f}s. Before that, little moves.",
                         f"First sustained movement at {act:.1f}s. Shots before it: {detail}.",
                         f"Open on the moment at {act:.1f}s. Put the hook text over the action, not over static faces."))

    if act and len(hook_shots) >= 2 and all(s.motion < ACTIVE_MOTION or s.duration < 1.5 for s in hook_shots):
        f.append(Finding("static_montage", 0, act, 2,
                         f"Hook is a montage of {len(hook_shots)} short, mostly still shots",
                         f"{len(hook_shots)} shots in {act:.1f}s, each {min(s.duration for s in hook_shots):.1f}-"
                         f"{max(s.duration for s in hook_shots):.1f}s long.",
                         "Use one shot that shows the promise happening. A still face is a promise; the action is the proof."))

    early_flashes = [t for t in sig.flashes if t < 5]
    if sig.flashes:
        f.append(Finding("flash_transitions", sig.flashes[0], sig.flashes[-1] + 0.4,
                         3 if early_flashes else 1,
                         f"{len(sig.flashes)} fade-to-white transition(s)",
                         "Washed-out frames at " + ", ".join(f"{t:.1f}s" for t in sig.flashes) + ".",
                         "Use hard cuts. A white flash looks like the video is loading or broken."))

    tail = static_tail(sig)
    if tail:
        length = tail[1] - tail[0]
        share = length / sig.duration
        f.append(Finding("static_end_card", tail[0], tail[1], 3 if share > 0.2 else 2,
                         f"{length:.1f}s static end card ({share:.0%} of the video)",
                         f"Last shot from {tail[0]:.1f}s has almost no movement (motion "
                         f"{sig.shots[-1].motion}).",
                         "Cut the end card to 1-1.5s, or move the call to action into the middle of the video as a text overlay."))

    if quiet_start(sig):
        f.append(Finding("quiet_start", 0, 0.5, 1, "Sound starts quiet",
                         f"First 0.5s is {sig.audio_db[0]} dB vs about "
                         f"{sorted(sig.audio_db[2:])[len(sig.audio_db[2:]) // 2]} dB for the rest.",
                         "Start the audio at full level; no fade-in."))
    if not sig.has_audio:
        f.append(Finding("no_audio", 0, sig.duration, 2, "No audio track", "The file has no audio stream.",
                         "Add a voiceover or a trending sound."))

    if sig.duration > 35 and (act or 0) > 2:
        f.append(Finding("long", 0, sig.duration, 1, f"Long for a hook that starts late ({sig.duration:.0f}s)",
                         "Long reels need a strong first second.", "Trim to the essential steps."))
    return f


def predict_retention(sig: Signals, params: dict, step: float = 0.2) -> list[tuple[float, float]]:
    """Survival curve: share of starting viewers still watching at each time."""
    act = action_start(sig)
    tail = static_tail(sig)
    qs = quiet_start(sig)
    s, t, curve = 1.0, 0.0, [(0.0, 1.0)]
    while t < sig.duration - 1e-6:
        h = params["base_hazard"] * (1 + params["early_boost"] * math.exp(-t / params["early_tau"]))
        if act is None or t < act:
            h *= params["no_action_mult"]
        if any(0 <= t - fl < 0.4 for fl in sig.flashes):
            h *= params["flash_mult"]
        if tail and t >= tail[0]:
            h *= params["static_tail_mult"]
        if qs and t < 1.0:
            h *= params["quiet_audio_mult"]
        s *= math.exp(-h * step)
        t = round(t + step, 3)
        curve.append((t, round(s, 4)))
    return curve


def at(curve: list[tuple[float, float]], t: float) -> float:
    best = curve[0][1]
    for ct, cs in curve:
        if ct <= t + 1e-6:
            best = cs
    return best


def biggest_drop_window(curve: list[tuple[float, float]], width: float = 1.0) -> tuple[float, float]:
    """1-second window where the curve loses the most viewers."""
    best, best_t = -1.0, 0.0
    for t, _ in curve:
        if t + width > curve[-1][0] + 1e-6:
            break
        loss = at(curve, t) - at(curve, t + width)
        if loss > best:
            best, best_t = loss, t
    return best_t, best_t + width


def findings_dicts(fs: list[Finding]) -> list[dict]:
    return [asdict(x) for x in sorted(fs, key=lambda x: (-x.severity, x.start))]
