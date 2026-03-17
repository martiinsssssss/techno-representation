from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

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

FEATURE_DIR = Path("data/features_musicradar")

st.set_page_config(page_title="Techno Audio Representations Explorer", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}
    h1, h2, h3, h4, h5, h6, p, div, label, span {{
        color: {TEXT_COLOR};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    features = pd.read_csv(FEATURE_DIR / "features.csv")
    classical_umap = pd.read_csv(FEATURE_DIR / "classical_umap.csv")
    encodec_umap = pd.read_csv(FEATURE_DIR / "encodec_umap.csv")
    clap_umap = pd.read_csv(FEATURE_DIR / "clap_umap.csv")
    return features, classical_umap, encodec_umap, clap_umap


def make_spectrogram(filepath):
    y, sr = sf.read(filepath)
    if y.ndim == 2:
        y = y.mean(axis=1)

    y = y.astype("float32")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3), facecolor=BG_COLOR)
    ax.set_facecolor(AX_COLOR)

    img = librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="log",
        ax=ax,
        cmap="magma",
    )

    ax.set_title(Path(filepath).name, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    for spine in ax.spines.values():
        spine.set_color("#3a3d46")

    cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)

    plt.tight_layout()
    return fig


st.title("Techno Audio Representations Explorer")

features, classical_umap, encodec_umap, clap_umap = load_data()

audio_names = features["filename"].tolist()
selected_name = st.selectbox("Choose an audio file", audio_names)

selected_row = features[features["filename"] == selected_name].iloc[0]
filepath = selected_row["filepath"]

st.subheader("Audio")
st.audio(filepath)

st.subheader("Metadata")
st.write({
    "filename": selected_row["filename"],
    "label": selected_row["label"],
})

st.subheader("Spectrogram")
fig = make_spectrogram(filepath)
st.pyplot(fig)

st.subheader("2D Representation Spaces")
view = st.radio("Select representation", ["Classical UMAP", "Encodec UMAP", "CLAP UMAP"])

if view == "Classical UMAP":
    plot_df = classical_umap.copy()
    x_col, y_col = "classical_umap_x", "classical_umap_y"
elif view == "Encodec UMAP":
    plot_df = encodec_umap.copy()
    x_col, y_col = "encodec_umap_x", "encodec_umap_y"
else:
    plot_df = clap_umap.copy()
    x_col, y_col = "clap_umap_x", "clap_umap_y"

plot_df["selected"] = plot_df["filename"] == selected_name

fig_scatter = px.scatter(
    plot_df,
    x=x_col,
    y=y_col,
    color="label",
    color_discrete_map=LABEL_COLORS,
    symbol="selected",
    hover_data=["filename", "label"],
    title=view,
)

fig_scatter.update_traces(
    marker=dict(size=11, line=dict(width=1, color="white"), opacity=0.92)
)

fig_scatter.update_layout(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=AX_COLOR,
    font=dict(color=TEXT_COLOR, size=12),
    title_font=dict(size=18, color=TEXT_COLOR),
    legend_title_text="Category",
    xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, color=TEXT_COLOR),
    yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, color=TEXT_COLOR),
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("A few feature values")
feature_preview = {
    k: selected_row[k]
    for k in selected_row.index
    if k in [
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
}
st.write(feature_preview)