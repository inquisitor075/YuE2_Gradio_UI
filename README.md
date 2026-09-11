# YuE2 Gradio UI

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

    git clone https://github.com/inquisitor075/yue2-gradio-ui.git
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
