#!/usr/bin/env python3
"""YuE2 Streamlit UI — dropdowns for language, CoT mode, key."""

import os
import time
import streamlit as st
from pathlib import Path
from yue2 import YuE2Pipeline

OUTPUT_DIR = "/mnt/c/Users/Kagur/Music/YuE2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@st.cache_resource
def load_pipeline():
    with st.spinner("Loading YuE2 pipeline... (this may take a minute)"):
        pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda")
    return pipe

def read_abc(artifact_dir):
    score_path = Path(artifact_dir) / "score.abc"
    if score_path.exists():
        return score_path.read_text(encoding="utf-8")
    for p in Path(artifact_dir).rglob("*.abc"):
        return p.read_text(encoding="utf-8")
    return "No ABC score found."

pipe = load_pipeline()

st.title("🎵 YuE2 Music Generator")
st.markdown("Generate full songs with vocals. [YuE2 model](https://huggingface.co/m-a-p/YuE2-3B)")

LANGUAGES = ["English", "Chinese", "Russian", "Japanese", "Korean", "Spanish", "French", "German"]
COT_MODES = ["full", "melody", "off"]
KEYS = ["C major", "G major", "D major", "A major", "E major", "B major",
        "F major", "A minor", "E minor", "B minor", "G minor", "D minor",
        "C# minor", "F# minor", "D# minor", "Bb major", "Eb major"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    style = st.text_area("Style prompt", 
        value="Cyber metal, powerful male vocals, heavy distorted guitars, driving drums, heavy bass, defiant mood",
        height=100)
    lyrics = st.text_area("Lyrics",
        value="[Verse 1]\nOn the knife's edge, walking slow and sure\nRust turns to silver, the air turns pure\n\n[Chorus]\nFoggy Albion! Through the dust we roam",
        height=300)
    abc_score = st.text_area("ABC Score (optional)",
        value="",
        height=150,
        help="Leave empty for auto-planning.")
    
    with st.expander("Song settings", expanded=True):
        col_lang, col_cot, col_key = st.columns(3)
        with col_lang:
            language = st.selectbox("Language", LANGUAGES)
        with col_cot:
            cot = st.selectbox("CoT mode", COT_MODES)
        with col_key:
            key = st.selectbox("Key", KEYS)
        
        col_bpm, col_dur, col_seed = st.columns(3)
        with col_bpm:
            bpm = st.text_input("BPM", value="120")
        with col_dur:
            duration = st.text_input("Duration", value="3:30")
        with col_seed:
            seed = st.text_input("Seed", value="42")
        
        cfg_scale = st.text_input("CFG scale", value="1.2")
    
    generate = st.button("🎵 Generate", type="primary", use_container_width=True)

with col2:
    st.subheader("Output")
    if generate:
        if not style.strip() or not lyrics.strip():
            st.error("Style and lyrics must not be empty.")
        else:
            with st.spinner("Generating song... This may take 1–3 minutes"):
                try:
                    seed_val = int(seed)
                    cfg_val = float(cfg_scale)
                    
                    enhanced_style = style.strip()
                    enhanced_style += f", {language} language"
                    if bpm.strip():
                        enhanced_style += f", {bpm.strip()} bpm"
                    enhanced_style += f", key of {key}"
                    if duration.strip():
                        enhanced_style += f", duration {duration.strip()}"
                    
                    timestamp = int(time.time())
                    artifact_dir = os.path.join(OUTPUT_DIR, f"song_{timestamp}_seed{seed_val}")
                    os.makedirs(artifact_dir, exist_ok=True)
                    
                    if abc_score.strip():
                        song = pipe(style=enhanced_style, lyrics=lyrics, abc=abc_score, cot=cot, seed=seed_val, cfg_scale=cfg_val)
                    else:
                        song = pipe(style=enhanced_style, lyrics=lyrics, cot=cot, seed=seed_val, cfg_scale=cfg_val)
                    
                    audio_path = os.path.join(artifact_dir, "song.flac")
                    song.save(audio_path)
                    song.save_artifacts(artifact_dir)
                    
                    abc_text = read_abc(artifact_dir)
                    
                    st.success(f"Done! Saved to {artifact_dir}")
                    st.audio(audio_path, format="audio/flac")
                    
                    with open(audio_path, "rb") as f:
                        st.download_button("Download song.flac", f, file_name="song.flac", mime="audio/flac")
                    
                    with st.expander("ABC Score"):
                        st.text_area("Generated ABC", abc_text, height=300)
                    with st.expander("Generation Info"):
                        st.text(f"CoT: {cot}\nSeed: {seed_val}\nCFG: {cfg_val}\nEnhanced style: {enhanced_style}\nArtifacts: {artifact_dir}")
                
                except Exception as e:
                    st.error(f"Generation error: {e}")
    else:
        st.info("Fill in the inputs and click Generate.")
