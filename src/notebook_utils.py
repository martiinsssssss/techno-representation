from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import Audio, HTML, display

from src.constants import (
    AX_COLOR,
    BG_COLOR,
    GRID_COLOR,
    LABEL_COLORS,
    NOTEBOOK_OUTPUT_CSS,
    SPINE_COLOR,
    TEXT_COLOR,
    WAVE_COLOR,
)


def apply_notebook_output_theme() -> None:
    display(HTML(NOTEBOOK_OUTPUT_CSS))


def apply_dark_matplotlib_theme() -> None:
    plt.rcParams["figure.facecolor"] = BG_COLOR
    plt.rcParams["axes.facecolor"] = AX_COLOR
    plt.rcParams["savefig.facecolor"] = BG_COLOR
    plt.rcParams["axes.edgecolor"] = SPINE_COLOR
    plt.rcParams["axes.labelcolor"] = TEXT_COLOR
    plt.rcParams["xtick.color"] = TEXT_COLOR
    plt.rcParams["ytick.color"] = TEXT_COLOR
    plt.rcParams["text.color"] = TEXT_COLOR
    plt.rcParams["axes.titlecolor"] = TEXT_COLOR
    plt.rcParams["grid.color"] = GRID_COLOR
    plt.rcParams["grid.alpha"] = 0.35
    plt.rcParams["axes.grid"] = True
    plt.rcParams["font.size"] = 11


def make_interactive_plot(plot_df: pd.DataFrame, title: str = "Interactive plot"):
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="label",
        color_discrete_map=LABEL_COLORS,
        hover_data=["filename", "label"],
        category_orders={"label": sorted(plot_df["label"].unique())},
        title=title,
    )

    fig.update_traces(
        marker=dict(
            size=11,
            line=dict(width=1, color="white"),
            opacity=0.92,
        )
    )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=AX_COLOR,
        font=dict(color=TEXT_COLOR, size=12),
        title_font=dict(size=18, color=TEXT_COLOR),
        legend_title_text="Category",
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, color=TEXT_COLOR),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, color=TEXT_COLOR),
        height=700,
        width=1100,
    )

    return fig


def show_audio_details(filename: str, feat_df: pd.DataFrame) -> None:
    row = feat_df[feat_df["filename"] == filename].iloc[0]
    y, sr = librosa.load(row["filepath"], sr=None, mono=True)

    display(
        HTML(
            f"""
    <div style="
        background:{BG_COLOR};
        color:{TEXT_COLOR};
        padding:18px;
        border-radius:16px;
        border:1px solid {SPINE_COLOR};
        margin:12px 0;
        box-shadow:0 4px 12px rgba(0,0,0,0.35);
    ">
        <div style="font-size:32px; font-weight:800; margin-bottom:6px;">
            {row['filename']}
        </div>
        <div style="font-size:16px; color:#dddddd;">
            <b>Category:</b> {row['label']}
        </div>
    </div>
    """
        )
    )
    display(Audio(row["filepath"]))

    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(AX_COLOR)
    librosa.display.waveshow(y, sr=sr, ax=ax, color=WAVE_COLOR, alpha=0.95)
    ax.set_title("Waveform", color=TEXT_COLOR, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel("Time", color=TEXT_COLOR, fontsize=12)
    ax.grid(color=GRID_COLOR, alpha=0.35)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    d_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(AX_COLOR)

    img = librosa.display.specshow(
        d_db,
        sr=sr,
        x_axis="time",
        y_axis="log",
        cmap="magma",
        ax=ax,
    )

    cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR)
    cbar.outline.set_edgecolor(SPINE_COLOR)

    ax.set_title("Spectrogram", color=TEXT_COLOR, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel("Time", color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("Hz", color=TEXT_COLOR, fontsize=12)
    ax.grid(color=GRID_COLOR, alpha=0.2)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    preview_cols = [
        c
        for c in [
            "rms_mean",
            "zcr_mean",
            "spectral_centroid_mean",
            "spectral_bandwidth_mean",
            "spectral_rolloff_mean",
            "onset_strength_mean",
            "encodec_num_codebooks",
            "encodec_total_frames",
            "clap_dim",
        ]
        if c in row.index
    ]

    preview_df = pd.DataFrame(
        {
            "feature": preview_cols,
            "value": [row[c] for c in preview_cols],
        }
    )
    preview_df["value"] = pd.to_numeric(preview_df["value"], errors="coerce").round(4)

    display(
        preview_df.style.hide(axis="index")
        .set_properties(
            **{
                "background-color": BG_COLOR,
                "color": TEXT_COLOR,
                "border-color": SPINE_COLOR,
                "font-size": "12pt",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [("border-collapse", "collapse"), ("width", "420px")],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#111111"),
                        ("color", TEXT_COLOR),
                        ("font-weight", "bold"),
                        ("border", f"1px solid {SPINE_COLOR}"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [("border", f"1px solid {SPINE_COLOR}")],
                },
            ]
        )
    )


def save_dark_plot(plot_df: pd.DataFrame, title: str, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.6), facecolor=BG_COLOR)
    ax.set_facecolor(AX_COLOR)

    for label, subdf in plot_df.groupby("label"):
        ax.scatter(
            subdf["x"],
            subdf["y"],
            label=label,
            alpha=0.9,
            s=90,
            color=LABEL_COLORS.get(label, "#999999"),
            edgecolors="white",
            linewidths=0.7,
        )

    ax.set_title(title, fontsize=15, pad=12, color=TEXT_COLOR)
    ax.set_xlabel("UMAP 1", color=TEXT_COLOR)
    ax.set_ylabel("UMAP 2", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, alpha=0.35, color=GRID_COLOR)

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)

    leg = ax.legend(title="Category", fontsize=9, title_fontsize=10, frameon=True)
    leg.get_frame().set_facecolor(AX_COLOR)
    leg.get_frame().set_edgecolor(SPINE_COLOR)
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)
    leg.get_title().set_color(TEXT_COLOR)

    outpath = str(Path(outpath))
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()
