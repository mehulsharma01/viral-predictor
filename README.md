# Viral Predictor

Checks a reel **before** you post it and tells you **why** viewers will leave, with timestamps. After you
post, you log the real Instagram Insights, and it learns from your own results.

It does not promise "this will go viral". Nobody can say that honestly. It predicts what can be checked:

| Prediction | Checked against |
|---|---|
| Skip rate | Insights › Overview › Skip rate |
| Where viewers drop (the reasons) | The retention graph in Insights |
| Which of two versions is better | The real numbers of both |

The target is **80% on those checks**, measured on your own posted reels (`python predict.py calibrate`).

## How it works

1. **Measure** (`vp/extract.py`, plain code with no AI, so the same file always gives the same result): cuts, motion per shot, fade-to-white
   frames, when the first sustained action starts, a static end card, audio level per 0.5s, and an
   optional transcript (`pip install faster-whisper`).
2. **Rules** (`vp/rules.py`): turns the measurements into findings with time ranges, and a predicted
   retention curve. The weights start as priors.
3. **Claude checklist** (`vp/judge.py`): Claude looks at the frames (4 per second for the first 3s, then
   1 per second) and answers a fixed checklist: what's in the first frame, hook text, whether the promise is
   shown or only written, when the payoff happens, whether there's a share or save reason, whether the footage
   is original, better openings and a re-cut plan.
4. **Score and report**: hook, hold, value, share, overall, a verdict and an HTML report.
5. **Learn** (`vp/calibrate.py`): every posted reel is logged with the prediction made *before* posting.
   From 8 reels on, a model is fitted on your data and tested leave-one-out.

## Use

```bash
pip install -r requirements.txt          # also needs ffmpeg installed
export ANTHROPIC_API_KEY=...             # for the Claude checklist (skip with --no-ai)

python predict.py analyze reel.mp4 --context "Glimsera, beauty hacks"
python predict.py compare version_a.mp4 version_b.mp4

# after posting, from Insights:
python predict.py log reports/reel/report.json --skip-rate 77.6 --drop 0-2.5 \
    --views 1000 --likes 7 --comments 0 --shares 0 --saves 3 --reposts 2 --non-followers 56.6
python predict.py calibrate

streamlit run app.py                     # same thing in a browser
```

`--drop` is the stretch of the retention graph where the line falls hardest (in seconds).

## What's in `data/`

`outcomes.csv` is the learning memory: one row per posted reel. Commit it. `model.json` appears once 8
reels are logged.

The first row is the "grey eyeshadow" reel (77.6% skip, 1k views). The priors were set after looking at
that reel, so its near-perfect prediction **is not evidence**. Only reels logged from now on count.

## Old version

The original text-only simulator (asks GPT 20 times "would you like this?") is in
`legacy/text_simulator.py` (needs `openai scipy statsmodels`, no longer in requirements). It never looked at the video, so it could not see timing problems.

## Browser version

`web/reel-check.html` runs the same measurement in the browser (it reads every frame while the video plays)
and asks Claude through the claude.ai artifact runtime, so no API key is needed there. Keep
`analyseFrames()` in it in step with `analyse_frames()` in `vp/extract.py`.
