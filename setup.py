import os

files = {}

files["app.py"] = '''#!/usr/bin/env python3
"""YuE2 Gradio UI — extended web interface with ABC score and artifacts."""

import argparse
import os
import time
import gradio as gr
from pathlib import Path
from yue2 import YuE2Pipeline


def load_pipeline(model_repo, device):
    print(f"Loading YuE2 pipeline from '{model_repo}' on '{device}'...")
    pipe = YuE2Pipeline.from_pretrained(model_repo, device=device)
    print("Pipeline loaded!")
    return pipe


def read_abc(artifact_dir):
    """Read score.abc from artifacts if available."""
    score_path = Path(artifact_dir) / "score.abc"
    if score_path.exists():
        return score_path.read_text(encoding="utf-8")
    for p in Path(artifact_dir).rglob("*.abc"):
        return p.read_text(encoding="utf-8")
    return "No ABC score found."


def generate_song(pipe, style, lyrics, cot, seed, cfg_scale):
    if not style.strip() or not lyrics.strip():
        return None, "Error: style and lyrics must not be empty.", "", "", "", "", "", ""

    try:
        seed = int(seed)
        timestamp = int(time.time())
        artifact_dir = f"outputs/song_{timestamp}_seed{seed}"
        os.makedirs(artifact_dir, exist_ok=True)

        song = pipe(style=style, lyrics=lyrics, cot=cot, seed=seed, cfg_scale=cfg_scale)

        audio_path = os.path.join(artifact_dir, "song.flac")
        song.save(audio_path)
        song.save_artifacts(artifact_dir)

        abc_text = read_abc(artifact_dir)
        info = f"CoT: {cot}\\nSeed: {seed}\\nCFG: {cfg_scale}\\nArtifacts: {os.path.abspath(artifact_dir)}"
        status = f"Done! Audio + ABC saved to {artifact_dir}"

        return audio_path, status, abc_text, lyrics, style, info, artifact_dir, abc_text
    except Exception as e:
        return None, f"Generation error: {e}", "", "", "", "", "", ""


def build_interface(pipe):
    with gr.Blocks(title="YuE2 Music Generator") as demo:
        gr.Markdown("# YuE2 Music Generator")
        gr.Markdown("Generate complete songs with vocals + see the symbolic ABC score. Powered by [YuE2](https://huggingface.co/m-a-p/YuE2-3B).")

        with gr.Row():
            with gr.Column(scale=1):
                style = gr.Textbox(label="Style prompt", lines=3,
                    placeholder="Cyber metal, powerful male vocals, heavy guitars, driving drums, heavy bass, defiant mood")
                lyrics = gr.Textbox(label="Lyrics", lines=18,
                    placeholder="[Verse 1]\\nOn the knife's edge, walking slow and sure\\nRust turns to silver, the air turns pure\\n\\n[Chorus]\\nFoggy Albion! Through the dust we roam")
                with gr.Row():
                    cot = gr.Dropdown(["full", "melody", "off"], value="full", label="CoT mode")
                    seed = gr.Number(value=42, label="Seed", precision=0)
                    cfg_scale = gr.Slider(1.0, 2.0, value=1.2, step=0.1, label="CFG scale")
                generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=2):
                audio = gr.Audio(label="Generated song", type="filepath", show_download_button=True)
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("ABC Score"):
                        abc_out = gr.Textbox(label="Symbolic score (ABC notation)", lines=25, interactive=False, show_copy_button=True)
                        abc_download = gr.File(label="Download ABC", visible=False)
                    with gr.TabItem("Lyrics"):
                        lyrics_out = gr.Textbox(label="Input lyrics", lines=25, interactive=False, show_copy_button=True)
                    with gr.TabItem("Style"):
                        style_out = gr.Textbox(label="Input style", lines=10, interactive=False, show_copy_button=True)
                    with gr.TabItem("Info"):
                        info_out = gr.Textbox(label="Generation info", lines=10, interactive=False)
                        artifacts_out = gr.Textbox(label="Artifacts folder", lines=2, interactive=False)

        generate_btn.click(
            generate_song,
            inputs=[style, lyrics, cot, seed, cfg_scale],
            outputs=[audio, status, abc_out, lyrics_out, style_out, info_out, artifacts_out, abc_download],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="YuE2 Gradio UI")
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    pipe = load_pipeline(args.model, args.device)
    demo = build_interface(pipe)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
'''

files["requirements.txt"] = "gradio>=4.0.0\n"

files["README.md"] = '''# YuE2 Gradio UI

A lightweight web interface for **YuE2** — the open music generation model that
**rivals Suno v5**. Turn lyrics and a style prompt into a complete song with
vocals and accompaniment, then inspect the symbolic ABC score and download
the generated artifacts.

## About YuE2

YuE2 is developed by **M-A-P (Multimodal Art Projection)**:

- **Main repository:** [github.com/m-a-p/YuE](https://github.com/m-a-p/YuE)
- **Model weights:** [huggingface.co/m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)
- **Paper:** [arXiv:2503.08638](https://arxiv.org/abs/2503.08638)

This project is **only a UI wrapper** — it does not include the model itself.

## Features

- Generate full songs from lyrics + style prompt
- Listen to the result in the browser
- Inspect the generated **ABC symbolic score**
- View input lyrics, style, and generation info
- Auto-save audio + artifacts to `outputs/`

## Prerequisites

Install YuE2 first, following the official instructions in the
[YuE2 model card](https://huggingface.co/m-a-p/YuE2-3B):

- **Linux** (or WSL2 on Windows)
- **Python 3.10+**
- **NVIDIA GPU with BF16 support** (16GB+ VRAM recommended)

## Installation

    git clone https://github.com/YOUR_USERNAME/yue2-gradio-ui.git
    cd yue2-gradio-ui
    pip install -r requirements.txt

## Usage

    python app.py

Then open **http://localhost:7860** in your browser.

### Options

    python app.py --port 7860        # custom port
    python app.py --share            # public Gradio link
    python app.py --model m-a-p/YuE2-3B

## Artifacts

Each generation creates a folder under `outputs/` containing:

- `song.flac` — generated audio
- `score.abc` — symbolic score
- `tokens/`, `latents/`, `settings.json` — from YuE2

## License

- **This UI code:** MIT
- **YuE2 model weights:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial)

## Credits

All credit for the model goes to the [YuE](https://github.com/m-a-p/YuE) team at M-A-P.
'''

files[".gitignore"] = '''# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# Output files
*.flac
*.wav
*.mp3
outputs/

# Misc
.DS_Store
'''

files["LICENSE"] = '''MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

for name, content in files.items():
    with open(name, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created {name}")

print("All files created!")