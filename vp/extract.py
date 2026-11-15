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


def _frames(path: str) -> tuple[np.ndarray, float]:
    """Every decoded frame, tiny and grey, with the stream frame rate."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=avg_frame_rate", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    num, den = (out.split("/") + ["1"])[:2]
    rate = float(num) / float(den or 1) if float(den or 1) else 30.0
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32), rate


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
    dense, rate = _frames(path)
    sig = analyse_frames([i / rate for i in range(len(dense))], dense, meta["duration"])
    at, adb = _audio_db(path) if meta["has_audio"] else ([], [])
    return Signals(
        path=str(Path(path)), duration=round(meta["duration"], 2), width=meta["width"],
        height=meta["height"], has_audio=meta["has_audio"], audio_times=at, audio_db=adb,
        transcript=_transcribe(path) if transcribe else [], **sig,
    )


def analyse_frames(t: list[float], g: np.ndarray, duration: float) -> dict:
    """Cuts, flashes and shots from every frame; motion on a 0.2 s grid.

    The browser page (reel-check) runs the same steps in JavaScript; keep them in sync.
    """
    n = len(g)
    lum = g.reshape(n, -1).mean(1)
    std = g.reshape(n, -1).std(1)
    d1 = np.zeros(n)
    d1[1:] = np.abs(g[1:] - g[:-1]).reshape(n - 1, -1).mean(1)

    cuts: list[float] = []
    flashes: list[float] = []
    for i in range(1, n):
        if d1[i] < CUT_MOTION or (cuts and t[i] - cuts[-1] < 0.3):
            continue
        cuts.append(round(t[i], 2))
        # Flash: jumps to a bright, flat frame and fades back down within 0.6 s.
        if lum[i] - lum[i - 1] > 40 and std[i] < 0.75 * std[i - 1]:
            j = i + 1
            while j < n and t[j] - t[i] <= 0.6:
                if lum[j] < lum[i] - 30:
                    flashes.append(round(t[i], 2))
                    break
                j += 1

    # 0.2 s grid of motion (nearest decoded frame to each grid time).
    times = [round(k / FPS, 2) for k in range(int(round(duration * FPS)))]
    idx = np.searchsorted(np.array(t), np.array(times) - 1e-6).clip(0, n - 1)
    motion = [0.0] + [float(np.abs(g[idx[k]] - g[idx[k - 1]]).mean()) for k in range(1, len(times))]

    def clean(k: int) -> bool:  # grid step k doesn't straddle a cut or a flash fade
        a, b = times[k - 1], times[k]
        return not any(a < c <= b + 1e-6 for c in cuts) and not any(f <= a < f + 0.5 for f in flashes)

    bounds = [0.0] + cuts + [duration]
    shots: list[Shot] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        inner = [motion[k] for k in range(1, len(times)) if a <= times[k - 1] and times[k] <= b + 1e-6 and clean(k)]
        lum_in = [lum[i] for i in range(n) if a <= t[i] < b] or [0.0]
        shot = Shot(start=round(a, 2), end=round(b, 2), motion=round(float(np.median(inner)) if inner else 0.0, 1),
                    brightness=round(float(np.mean(lum_in)), 1))
        if shots and shot.duration < 0.3:
            shots[-1].end = shot.end
        else:
            shots.append(shot)
    return {"times": times, "motion": [round(m, 1) for m in motion],
            "brightness": [round(float(lum[i]), 1) for i in idx],
            "contrast": [round(float(std[i]), 1) for i in idx],
            "cuts": cuts, "flashes": flashes, "shots": shots}


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
