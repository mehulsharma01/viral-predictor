"""Step 1: measure the video. Plain signal processing, no AI.

Everything here is reproducible: the same file always gives the same numbers.
Needs ffmpeg and ffprobe on PATH.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

FPS = 5            # analysis frame rate (one frame every 0.2 s)
W, H = 72, 128     # tiny frames are enough for motion / brightness
CUT_MOTION = 40.0  # mean abs pixel change that counts as a hard cut


@dataclass
class Shot:
    start: float
    end: float
    motion: float      # median within-shot motion (how much is moving)
    brightness: float  # mean luminance 0-255

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


@dataclass
class Signals:
    path: str
    duration: float
    width: int
    height: int
    has_audio: bool
    times: list[float] = field(default_factory=list)
    motion: list[float] = field(default_factory=list)
    brightness: list[float] = field(default_factory=list)
    contrast: list[float] = field(default_factory=list)
    cuts: list[float] = field(default_factory=list)
    flashes: list[float] = field(default_factory=list)   # fade-to-white / washed-out frames
    shots: list[Shot] = field(default_factory=list)
    audio_times: list[float] = field(default_factory=list)
    audio_db: list[float] = field(default_factory=list)  # RMS dB per 0.5 s window
    transcript: list[dict] = field(default_factory=list)  # [{start,end,text}]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shots"] = [asdict(s) | {"duration": s.duration} for s in self.shots]
        return d


def _need(tool: str) -> None:
    if not shutil.which(tool):
        raise SystemExit(f"{tool} not found. Install ffmpeg first (https://ffmpeg.org).")


def probe(path: str) -> dict:
    _need("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {
        "duration": float(info["format"]["duration"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def _frames(path: str) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={FPS},scale={W}:{H}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.float32)


def _audio_db(path: str, window: float = 0.5) -> tuple[list[float], list[float]]:
    sr = 8000
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "s16le", "-"], capture_output=True).stdout
    if not raw:
        return [], []
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    n = int(sr * window)
    times, dbs = [], []
    for i in range(0, len(a) - n + 1, n):
        rms = float(np.sqrt(np.mean(a[i:i + n] ** 2)) + 1e-9)
        times.append(round(i / sr, 2))
        dbs.append(round(float(20 * np.log10(rms)), 1))
    return times, dbs


def _transcribe(path: str) -> list[dict]:
    """Optional: timed transcript with faster-whisper if it is installed."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        return []
    model = WhisperModel("small", compute_type="int8")
    segments, _ = model.transcribe(path, vad_filter=True)
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments]


def extract(path: str, transcribe: bool = True) -> Signals:
    _need("ffmpeg")
    meta = probe(path)
    fr = _frames(path)
    gray = fr.mean(axis=3)
    n = len(gray)
    times = [round(i / FPS, 2) for i in range(n)]
    motion = [0.0] + [float(np.abs(gray[i] - gray[i - 1]).mean()) for i in range(1, n)]
    bright = [float(g.mean()) for g in gray]
    contrast = [float(g.std()) for g in gray]

    # Cuts: motion spikes. Adjacent spike frames are one transition.
    cut_idx: list[int] = []
    for i, m in enumerate(motion):
        if m >= CUT_MOTION and (not cut_idx or i - cut_idx[-1] > 1):
            cut_idx.append(i)

    # Flashes: a frame much brighter and flatter than its neighbours
    # (fade-through-white). They read as "the video glitched" and cost viewers.
    flashes = []
    for i in range(1, n - 1):
        nb = (bright[i - 1] + bright[i + 1]) / 2
        returns = bright[i] - bright[i + 1] > 30  # a flash comes back down; a cut to a bright card does not
        if returns and bright[i] - nb > 40 and contrast[i] < 0.75 * min(contrast[i - 1], contrast[i + 1]):
            flashes.append(times[i])

    # Shots between cuts; motion inside a shot excludes the cut frames themselves.
    bounds = [0] + cut_idx + [n]
    shots: list[Shot] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 1:
            continue
        inner = [motion[j] for j in range(a + 1, b) if motion[j] < CUT_MOTION] or [0.0]
        shots.append(Shot(start=round(a / FPS, 2), end=round(min(b / FPS, meta["duration"]), 2),
                          motion=round(float(np.median(inner)), 1),
                          brightness=round(float(np.mean(bright[a:b])), 1)))
    # merge micro-shots (flash frames) into the following shot
    merged: list[Shot] = []
    for s in shots:
        if merged and s.duration < 0.3:
            merged[-1].end = s.end
        else:
            merged.append(s)

    at, adb = _audio_db(path) if meta["has_audio"] else ([], [])
    return Signals(
        path=str(Path(path)), duration=round(meta["duration"], 2), width=meta["width"],
        height=meta["height"], has_audio=meta["has_audio"], times=times,
        motion=[round(m, 1) for m in motion], brightness=[round(b, 1) for b in bright],
        contrast=[round(c, 1) for c in contrast], cuts=[times[i] for i in cut_idx],
        flashes=flashes, shots=merged, audio_times=at, audio_db=adb,
        transcript=_transcribe(path) if transcribe else [],
    )


def keyframes(path: str, out_dir: str, hook_fps: float = 4, rest_fps: float = 1,
              hook_seconds: float = 3.0, width: int = 360) -> list[tuple[float, str]]:
    """JPEG frames for the AI judge: dense in the hook, sparse after."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dur = probe(path)["duration"]
    stamps = [round(i / hook_fps, 2) for i in range(int(hook_seconds * hook_fps))]
    t = hook_seconds
    while t < dur - 0.05:
        stamps.append(round(t, 2))
        t += 1 / rest_fps
    stamps.append(round(max(dur - 0.1, 0), 2))
    frames = []
    for ts in stamps:
        f = out / f"t{ts:06.2f}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(ts), "-i", path,
                        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(f)],
                       check=True)
        frames.append((ts, str(f)))
    return frames
