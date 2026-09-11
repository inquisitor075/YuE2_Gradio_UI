# 🎵 YuE2 Music Generator UI

A simple web interface for **YuE2-3B** — the open-source music generation model that rivals Suno v5. Generate full songs with vocals and accompaniment, control language, key, BPM, and more, right from your browser.

## 🔗 Links

- **My repository:** https://github.com/inquisitor075/YuE2_Gradio_UI
- **Official YuE2-3B model:** https://huggingface.co/m-a-p/YuE2-3B
- **Main YuE repo:** https://github.com/m-a-p/YuE
- **Paper:** https://arxiv.org/abs/2503.08638

## ⚙️ Requirements

- **Windows 10/11 with WSL2**
- **Ubuntu in WSL2**
- **Python 3.11**
- **NVIDIA GPU with BF16 support** (16GB+ VRAM recommended)
- **CUDA 12.8** (for RTX 40/50 series Blackwell)

## 🚀 Installation

### 1. Install YuE2 inside WSL2

Follow the official guide: https://huggingface.co/m-a-p/YuE2-3B

Short version:

python -m pip install huggingface-hub==0.36.2
hf download m-a-p/YuE2-3B yue2_infer-0.1.5-py3-none-any.whl --local-dir .
python -m pip install ./yue2_infer-0.1.5-py3-none-any.whl
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

### 2. Clone this UI

git clone https://github.com/inquisitor075/YuE2_Gradio_UI.git
cd YuE2_Gradio_UI
pip install -r requirements.txt

### 3. Create output folder on Windows

mkdir -p /mnt/c/Users/$USER/Music/YuE2

Or create C:\Users\YourName\Music\YuE2 in Windows Explorer.

## ▶️ Usage

### Run manually

streamlit run app.py

Then open the displayed URL in your browser, usually:
http://localhost:8501

### Run with a Windows shortcut

Create start.bat with:

@echo off
wsl -d Ubuntu -e bash -lc "cd /home/YOUR_WSL_USER/yue2_project/YuE2_Gradio_UI && source /home/YOUR_WSL_USER/yue2_project/venv/bin/activate && streamlit run app.py"
pause

Double-click it to launch.

## 🎛️ Controls

| Field | Description |
|-------|-------------|
| **Style prompt** | Genre, mood, instruments, vocal style |
| **Lyrics** | Song text, with optional [Verse] / [Chorus] markers |
| **Language** | English, Chinese, Russian, Japanese, etc. |
| **CoT mode** | full (melody + chords), melody, or off |
| **Key** | Musical key and scale |
| **BPM** | Tempo hint |
| **Duration** | Length hint, e.g. 3:30 |
| **Seed** | Reproducibility |
| **CFG scale** | Text guidance strength (1.0–2.0) |
| **ABC Score** | Optional symbolic score for precise melody/chord control |

## 📁 Output

All generated files are saved directly to your Windows disk:

C:\Users\YourName\Music\YuE2\song_TIMESTAMP_seedSEED\

Contains:
- song.flac — generated audio
- score.abc — symbolic score
- tokens/, latents/, settings.json — YuE2 artifacts

## 📝 Notes

- YuE2 model weights are **CC BY-NC 4.0** (non-commercial use only).
- This UI code is MIT licensed.

## 🙏 Credits

All credit for the YuE2 model goes to the [M-A-P](https://github.com/m-a-p) team.
