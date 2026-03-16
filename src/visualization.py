from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import plotly.express as px


def save_scatter_plot(df: pd.DataFrame, x: str, y: str, color: str, title: str, outpath: str):
    plt.figure(figsize=(8, 6))
    for label, subdf in df.groupby(color):
        plt.scatter(subdf[x], subdf[y], label=label, alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def interactive_scatter(df: pd.DataFrame, x: str, y: str, color: str, hover_data=None, title: str = ""):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        hover_data=hover_data or ["filename", "label"],
        title=title,
    )
    return fig


def save_spectrogram(filepath: str, outpath: str):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    D = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=max)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(Path(filepath).name)
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()