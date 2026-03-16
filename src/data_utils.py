from pathlib import Path
import pandas as pd

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


def scan_audio_dataset(data_dir: str) -> pd.DataFrame:
    """
    Expects:
    data/raw/
        kick/
        hat/
        bass/
        synth/
        texture/
    """
    data_dir = Path(data_dir)
    rows = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for audio_path in sorted(class_dir.rglob("*")):
            if audio_path.suffix.lower() in AUDIO_EXTENSIONS:
                rows.append(
                    {
                        "filepath": str(audio_path.resolve()),
                        "filename": audio_path.name,
                        "label": label,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No audio files found in {data_dir}")
    return df