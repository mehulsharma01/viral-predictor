"""Step 2b: Claude watches the frames and answers a fixed checklist.

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile). Skipped with --no-ai.
The answers are structured (pydantic), each with timestamps, so they can be
scored and checked later rather than read as free text.
"""
from __future__ import annotations

import base64
import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

MODEL = "claude-opus-5-5"


class Issue(BaseModel):
    start: float = Field(description="seconds")
    end: float = Field(description="seconds")
    severity: Literal[1, 2, 3]
    problem: str = Field(description="what a scrolling viewer experiences, one sentence")
    fix: str = Field(description="concrete edit instruction")


class Judgement(BaseModel):
    first_frame: str = Field(description="what is literally visible at 0.0s")
    hook_text: str = Field(description="on-screen text in the first 3 s, verbatim, or empty")
    promise: str = Field(description="what the viewer is promised, in their words")
    promise_shown_at: Optional[float] = Field(description="first second where the promise is SHOWN (not just written); null if never")
    payoff_at: Optional[float] = Field(description="second where the surprising/useful moment happens; null if none")
    payoff: str
    movement_in_first_second: bool = Field(description="does something visibly change/move in 0-1 s (not a cut, an action)")
    curiosity_gap: bool = Field(description="does the hook make the viewer need to see what happens next")
    text_readable_in_1s: bool = Field(description="hook text short and large enough to read in one second on a phone")
    share_trigger: Optional[str] = Field(description="why someone would send this to a friend; null if no clear reason")
    save_trigger: Optional[str] = Field(description="why someone would save it for later; null if no clear reason")
    cta_text: Optional[str]
    cta_at: Optional[float]
    original_footage: Literal["yes", "no", "unclear"] = Field(description="'no' if creator credits, watermarks or other creators' clips are visible")
    originality_evidence: str
    hook_score: int = Field(ge=0, le=10)
    value_score: int = Field(ge=0, le=10, description="how useful/surprising the content is once watched")
    issues: list[Issue]
    better_hooks: list[str] = Field(description="3 alternative opening concepts: first frame + text, each under 20 words")
    recut_plan: list[str] = Field(description="ordered edit steps using this footage only")


SYSTEM = """You are a short-form video editor who has studied thousands of Instagram Reels retention graphs.
You judge a reel the way a scrolling viewer experiences it: sound often on, thumb ready to swipe, deciding in under 1.5 seconds.
Be concrete and literal. Every claim must point at a timestamp from the frames you are given.
Do not praise. Do not guess at numbers you cannot see. If something is not visible in the frames, say so."""


def _img(path: str) -> dict:
    data = base64.standard_b64encode(open(path, "rb").read()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def judge(frames: list[tuple[float, str]], timeline: dict, context: str = "") -> Judgement:
    import anthropic

    content: list[dict] = [{"type": "text", "text":
        "Frames from one vertical reel, in order. Dense (4 per second) for the first 3 seconds, then 1 per second."}]
    for ts, path in frames:
        content.append({"type": "text", "text": f"t={ts:.2f}s"})
        content.append(_img(path))
    content.append({"type": "text", "text":
        "Measured timeline (from code, trust these numbers for cuts, motion and audio):\n"
        + json.dumps(timeline, indent=1)
        + (f"\n\nCreator context: {context}" if context else "")
        + "\n\nFill in the checklist. List issues in time order, only real ones, max 6."})

    client = anthropic.Anthropic()
    resp = client.beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": content}],
        output_format=Judgement,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"Claude did not return a judgement (stop_reason={resp.stop_reason}).")
    return resp.parsed_output
