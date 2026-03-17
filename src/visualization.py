from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import plotly.express as px

# Dark / techno palette
LABEL_COLORS = {
    "Bass": "#00f5d4",       # neon aqua
    "Claps": "#ff4d6d",      # hot pink-red
    "Cymbals": "#ffd60a",    # acid yellow
    "FX n Noise": "#9b5de5", # purple
    "Hats": "#f15bb5",       # pink
    "Hi Perc": "#00bbf9",    # electric blue
    "Kicks": "#ff006e",      # neon magenta
    "Lo Perc": "#adb5bd",    # cold grey
    "Snares": "#fb5607",     # orange
    "Synth1": "#3a86ff",     # vivid blue
    "Synth2": "#8338ec",     # violet
}

BG_COLOR = "#0b0b0f"
AX_COLOR = "#12131a"
GRID_COLOR = "#2a2d36"
TEXT_COLOR = "#f5f5f5"
EDGE_COLOR = "#ffffff"


def save_scatter_plot(df: pd.DataFrame, x: str, y: str, color: str, title: str, outpath: str):
    fig, ax = plt.subplots(figsize=(8.8, 6.6), facecolor=BG_COLOR)
    ax.set_facecolor(AX_COLOR)

    for label, subdf in df.groupby(color):
        ax.scatter(
            subdf[x],
            subdf[y],
            label=label,
            alpha=0.9,
            s=90,
            color=LABEL_COLORS.get(label, "#999999"),
            edgecolors=EDGE_COLOR,
            linewidths=0.7,
        )

    ax.set_xlabel(x.replace("_", " "), fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel(y.replace("_", " "), fontsize=11, color=TEXT_COLOR)
    ax.set_title(title, fontsize=15, pad=12, color=TEXT_COLOR)

    ax.grid(True, alpha=0.35, linewidth=0.6, color=GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    for spine in ax.spines.values():
        spine.set_color("#3a3d46")

    legend = ax.legend(
        frameon=True,
        fontsize=10,
        title="Category",
        title_fontsize=10,
        loc="best",
    )
    legend.get_frame().set_facecolor(AX_COLOR)
    legend.get_frame().set_edgecolor("#3a3d46")
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    legend.get_title().set_color(TEXT_COLOR)

    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()


def interactive_scatter(df: pd.DataFrame, x: str, y: str, color: str, hover_data=None, title: str = ""):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_map=LABEL_COLORS,
        hover_data=hover_data or ["filename", "label"],
        title=title,
    )

    fig.update_traces(
        marker=dict(
            size=11,
            line=dict(width=1.0, color="white"),
            opacity=0.92,
        )
    )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=AX_COLOR,
        font=dict(color=TEXT_COLOR, size=12),
        title_font=dict(size=18, color=TEXT_COLOR),
        legend_title_text="Category",
        xaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            color=TEXT_COLOR,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            color=TEXT_COLOR,
        ),
    )
    return fig


def save_spectrogram(filepath: str, outpath: str):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG_COLOR)
    ax.set_facecolor(AX_COLOR)

    img = librosa.display.specshow(
        D, sr=sr, x_axis="time", y_axis="log", cmap="magma", ax=ax
    )
    cbar = plt.colorbar(img, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)

    ax.set_title(Path(filepath).name, fontsize=12, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    for spine in ax.spines.values():
        spine.set_color("#3a3d46")

    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()