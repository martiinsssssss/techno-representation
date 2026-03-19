from pathlib import Path

# Dataset and output locations used across notebook and scripts
DATA_DIR = "data/music_radar"
OUTPUT_DIR = Path("data/features_musicradar")
FIGURES_DIR = Path("figures_musicradar")

# Consistent palette for class labels
LABEL_COLORS = {
    "Bass": "#00f5d4",
    "Claps": "#ff4d6d",
    "Cymbals": "#ffd60a",
    "FX n Noise": "#9b5de5",
    "Hats": "#f15bb5",
    "Hi Perc": "#00bbf9",
    "Kicks": "#ff006e",
    "Lo Perc": "#adb5bd",
    "Snares": "#fb5607",
    "Synth1": "#3a86ff",
    "Synth2": "#8338ec",
}

BG_COLOR = "#0b0b0f"
AX_COLOR = "#12131a"
GRID_COLOR = "#2a2d36"
TEXT_COLOR = "#f5f5f5"
SPINE_COLOR = "#3a3d46"
WAVE_COLOR = "#FF2D2D"

NOTEBOOK_OUTPUT_CSS = """
<style>
.jp-OutputArea-output, .jupyter-widgets-output-area, .output, .output_area {
    background: #000000 !important;
    color: #ffffff !important;
}

.widget-box, .jupyter-widgets {
    background: #000000 !important;
    color: #ffffff !important;
}

div.output_subarea {
    background: #000000 !important;
}

table {
    background: #000000 !important;
    color: #ffffff !important;
}

th, td {
    border-color: #444444 !important;
}
</style>
"""
