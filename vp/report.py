"""Self-contained HTML report (one file, images embedded)."""
from __future__ import annotations

import base64
import html
from pathlib import Path

SEV = {3: ("High", "#d9480f"), 2: ("Medium", "#e8a400"), 1: ("Low", "#868e96")}


def _chart(report: dict, w: int = 640, h: int = 220) -> str:
    curve = report["predicted"]["curve"]
    dur = report["measured"]["duration"]
    pad_l, pad_b, pad_t = 40, 26, 10
    iw, ih = w - pad_l - 10, h - pad_b - pad_t
    x = lambda t: pad_l + iw * t / dur
    y = lambda s: pad_t + ih * (1 - s)
    bands = "".join(
        f'<rect x="{x(f["start"]):.1f}" y="{pad_t}" width="{max(x(min(f["end"], dur)) - x(f["start"]), 2):.1f}" '
        f'height="{ih}" fill="{SEV[f["severity"]][1]}" opacity="0.12"/>'
        for f in report["findings"] if f["severity"] >= 2)
    pts = " ".join(f"{x(t):.1f},{y(s):.1f}" for t, s in curve)
    grid = "".join(f'<line x1="{pad_l}" x2="{w - 10}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="g"/>'
                   f'<text x="{pad_l - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>'
                   for v in (0, .5, 1))
    ticks = "".join(f'<text x="{x(t):.1f}" y="{h - 8}" text-anchor="middle">{t}s</text>'
                    for t in range(0, int(dur) + 1, 3))
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Predicted retention">{bands}{grid}{ticks}'
            f'<polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="2.5"/></svg>')


def _thumb(path: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'data:image/jpeg;base64,{data}'


def write_html(report: dict, out: Path) -> None:
    e = html.escape
    p, s, m = report["predicted"], report["scores"], report["measured"]
    ai = report.get("ai")
    hook_frames = [(t, f) for t, f in report["frames"] if t < 3][:12]
    strip = "".join(f'<figure><img src="{_thumb(f)}" alt="frame at {t}s"><figcaption>{t:.2f}s</figcaption></figure>'
                    for t, f in hook_frames)
    rows = "".join(
        f'<tr><td><span class="sev" style="background:{SEV[f["severity"]][1]}">{SEV[f["severity"]][0]}</span></td>'
        f'<td class="n">{f["start"]:.1f}–{f["end"]:.1f}s</td><td><b>{e(f["title"])}</b>'
        f'<div class="mut">{e(f["evidence"])}</div></td><td>{e(f["fix"])}</td><td class="mut">{f["source"]}</td></tr>'
        for f in report["findings"])
    tiles = "".join(f'<div class="tile"><div class="mut">{k}</div><div class="big">{v}</div></div>'
                    for k, v in s.items())
    ai_html = ""
    if ai:
        li = lambda xs: "".join(f"<li>{e(x)}</li>" for x in xs)
        ai_html = f"""<h2>What Claude saw</h2>
<dl><dt>First frame</dt><dd>{e(ai['first_frame'])}</dd>
<dt>Hook text</dt><dd>{e(ai['hook_text'] or '—')}</dd>
<dt>Promise</dt><dd>{e(ai['promise'])} (shown at {ai['promise_shown_at'] if ai['promise_shown_at'] is not None else 'never'}s)</dd>
<dt>Payoff</dt><dd>{e(ai['payoff'])} (at {ai['payoff_at'] if ai['payoff_at'] is not None else '—'}s)</dd>
<dt>Share reason</dt><dd>{e(ai['share_trigger'] or 'none found')}</dd>
<dt>Save reason</dt><dd>{e(ai['save_trigger'] or 'none found')}</dd>
<dt>Original footage</dt><dd>{ai['original_footage']}: {e(ai['originality_evidence'])}</dd></dl>
<h3>Better openings</h3><ol>{li(ai['better_hooks'])}</ol>
<h3>Re-cut plan</h3><ol>{li(ai['recut_plan'])}</ol>"""
    elif report.get("ai_error"):
        ai_html = f'<p class="mut">Claude checklist skipped: {e(report["ai_error"])}</p>'
    notes = "".join(f"<p class='note'>{e(n)}</p>" for n in report["notes"])
    cal = report["calibration"]
    conf = ("Calibrated on {} of your posted reels.".format(cal["reels_logged"]) if cal["calibrated"] else
            f"Not calibrated yet ({cal['reels_logged']} reels logged). Numbers come from starting rules; "
            "log each posted reel's real Insights to make them yours.")

    out.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Reel Check</title>
<style>
:root{{--bg:#fff;--fg:#131A12;--mut:#5c6658;--panel:#F4F7F1;--line:#dfe5da;--accent:#3C6410}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111510;--fg:#eef2ea;--mut:#9aa594;--panel:#1b2119;--line:#2c3429;--accent:#A3E635}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px}}
h1{{margin:0 0 4px}} .mut{{color:var(--mut);font-size:13px}} .big{{font-size:28px;font-weight:700}}
.verdict{{display:inline-block;background:var(--panel);border-left:4px solid var(--accent);padding:8px 12px;margin:12px 0;font-weight:600}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px}}
.tile{{background:var(--panel);padding:10px 12px;border-radius:8px;text-transform:capitalize}}
.strip{{display:flex;gap:6px;overflow-x:auto;padding-bottom:6px}} figure{{margin:0}} .strip img{{height:150px;border-radius:6px;display:block}}
figcaption{{font-size:12px;color:var(--mut);text-align:center}}
svg{{width:100%;height:auto;font-size:11px;fill:var(--mut)}} .g{{stroke:var(--line)}}
table{{width:100%;border-collapse:collapse}} td{{border-top:1px solid var(--line);padding:8px 6px;vertical-align:top}}
.n{{white-space:nowrap;font-variant-numeric:tabular-nums}} .sev{{color:#fff;border-radius:4px;padding:1px 6px;font-size:12px}}
.wrap{{overflow-x:auto}} .note{{background:#fff4e0;color:#5a3d00;padding:8px 12px;border-radius:6px}}
dt{{font-weight:600}} dd{{margin:0 0 8px}}
</style></head><body><main>
<h1>Reel Check: {e(report['name'])}</h1>
<div class="mut">{m['duration']}s · {len(m['cuts'])} cuts · demo starts at {m['action_start'] if m['action_start'] is not None else '—'}s</div>
<div class="verdict">{e(report['verdict'])}</div>{notes}
<div class="tiles">{tiles}
<div class="tile"><div class="mut">predicted skip rate</div><div class="big">{p['skip_rate']}%</div></div></div>
<p class="mut">{e(conf)}</p>
<h2>First 3 seconds</h2><div class="strip">{strip}</div>
<h2>Predicted retention</h2><p class="mut">Shaded: where the problems below sit. Compare with the real graph in Insights after posting.</p>{_chart(report)}
<h2>Why, with timestamps</h2><div class="wrap"><table>{rows}</table></div>
{ai_html}
</main></body></html>""")
