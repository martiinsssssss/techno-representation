from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

FEATURE_DIR = Path("data/features")


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

    D = librosa.amplitude_to_db(abs(librosa.stft(y.astype("float32"))), ref=max)
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax)
    ax.set_title(Path(filepath).name)
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
    "filepath": selected_row["filepath"],
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
    symbol="selected",
    hover_data=["filename", "label"],
    title=view,
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