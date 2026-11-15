#!/usr/bin/env python3
"""Viral Predictor: check a reel before posting, learn from it after posting.

  python predict.py analyze VIDEO.mp4 [--no-ai] [--context "beauty hacks, Dubai"]
  python predict.py compare A.mp4 B.mp4 [--no-ai]
  python predict.py log reports/NAME/report.json --skip-rate 77.6 --drop 0-2.5 --views 1000 --shares 0
  python predict.py calibrate
"""
from __future__ import annotations

import argparse
import json
import sys


def _summary(r: dict) -> str:
    p, s = r["predicted"], r["scores"]
    lines = [f"\n{r['name']}  ({r['measured']['duration']}s)",
             f"Verdict: {r['verdict']}",
             "Scores: " + "  ".join(f"{k} {v}" for k, v in s.items()),
             f"Predicted skip rate: {p['skip_rate']}%  ({p['skip_rate_source']})",
             f"Predicted still watching: 1s {p['retention_1s']}%, 3s {p['retention_3s']}%, "
             f"end of content {p['retention_end_of_content']}%", "", "Why:"]
    for f in r["findings"]:
        tag = {3: "HIGH", 2: "MED ", 1: "low "}[f["severity"]]
        lines.append(f"  [{tag}] {f['start']:.1f}-{f['end']:.1f}s  {f['title']}")
        lines.append(f"         fix: {f['fix']}")
    for n in r["notes"]:
        lines.append(f"\nNote: {n}")
    if r.get("ai_error"):
        lines.append(f"\nClaude checklist skipped ({r['ai_error'][:120]})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="score one video")
    a.add_argument("video")
    a.add_argument("--no-ai", action="store_true", help="measured rules only, no Claude call")
    a.add_argument("--context", default="", help="who the account is for, niche, market")
    a.add_argument("--out", help="report folder (default reports/<name>)")
    a.add_argument("--no-transcript", action="store_true")

    c = sub.add_parser("compare", help="which of two versions to post")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--no-ai", action="store_true")
    c.add_argument("--context", default="")

    lg = sub.add_parser("log", help="record real Insights after posting")
    lg.add_argument("report")
    lg.add_argument("--skip-rate", type=float, required=True, help="Insights > Overview > Skip rate, e.g. 77.6")
    lg.add_argument("--drop", help="seconds where the retention graph falls hardest, e.g. 0-2.5")
    for k in ("views", "likes", "comments", "shares", "saves", "reposts"):
        lg.add_argument(f"--{k}", type=int)
    lg.add_argument("--non-followers", type=float, dest="non_followers_pct")

    sub.add_parser("calibrate", help="fit on logged reels and print accuracy")

    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        from vp.analyze import analyze
        r = analyze(args.video, use_ai=not args.no_ai, context=args.context, out_dir=args.out,
                    transcribe=not args.no_transcript)
        print(_summary(r))
        print(f"\nReport: reports/{r['name']}/report.html" if not args.out else f"\nReport: {args.out}/report.html")
    elif args.cmd == "compare":
        from vp.analyze import analyze
        ra = analyze(args.a, use_ai=not args.no_ai, context=args.context)
        rb = analyze(args.b, use_ai=not args.no_ai, context=args.context)
        for r in (ra, rb):
            print(_summary(r))
        win = ra if ra["scores"]["overall"] >= rb["scores"]["overall"] else rb
        lose = rb if win is ra else ra
        print(f"\nPost: {win['name']}  (overall {win['scores']['overall']} vs {lose['scores']['overall']}, "
              f"predicted skip {win['predicted']['skip_rate']}% vs {lose['predicted']['skip_rate']}%)")
    elif args.cmd == "log":
        from vp.calibrate import log
        metrics = {k: getattr(args, k) for k in ("views", "likes", "comments", "shares", "saves", "reposts",
                                                 "non_followers_pct") if getattr(args, k) is not None}
        row = log(args.report, args.skip_rate, args.drop, **metrics)
        print(f"Logged {row['name']}: predicted skip {row['predicted_skip']}%, real {row['skip_rate']}%")
    elif args.cmd == "calibrate":
        from vp.calibrate import calibrate
        print(json.dumps(calibrate(), indent=2))


if __name__ == "__main__":
    sys.exit(main())
