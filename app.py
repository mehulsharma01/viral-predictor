"""Streamlit front end: upload a reel, get the report.  Run: streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from vp.analyze import analyze

st.set_page_config(page_title="Viral Predictor", layout="wide")
st.title("Viral Predictor")
st.caption("Check a reel before you post it. Every reason comes with a timestamp you can check against Insights.")

video = st.file_uploader("Reel (mp4/mov)", type=["mp4", "mov", "m4v"])
context = st.text_input("Account context (optional)", placeholder="Beauty hacks for women in the UAE")
use_ai = st.checkbox("Claude checklist (needs ANTHROPIC_API_KEY)", value=True)

if video and st.button("Analyze"):
    tmp = Path(tempfile.mkdtemp()) / video.name
    tmp.write_bytes(video.getbuffer())
    with st.spinner("Measuring cuts, motion and audio" + (", then asking Claude" if use_ai else "")):
        r = analyze(str(tmp), use_ai=use_ai, context=context)
    components.html(Path(f"reports/{r['name']}/report.html").read_text(), height=1600, scrolling=True)
