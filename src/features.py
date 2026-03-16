import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import soundfile as sf
import torch

from encodec import EncodecModel
from encodec.utils import convert_audio
from transformers import ClapModel, ClapProcessor


TARGET_SR = 48000
MAX_DURATION_SEC = 4.0


def load_audio_mono(filepath: str, sr: int = TARGET_SR, max_duration_sec: float = MAX_DURATION_SEC):
    y, orig_sr = sf.read(filepath)

    if y.ndim == 2:
        y = np.mean(y, axis=1)

    max_samples = int(max_duration_sec * orig_sr)
    if len(y) > max_samples:
        y = y[:max_samples]

    if orig_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=sr)

    return y.astype(np.float32), sr


def summarize_feature_matrix(x: np.ndarray, prefix: str):
    feats = {}
    for i in range(x.shape[0]):
        vals = x[i]
        feats[f"{prefix}_{i}_mean"] = float(np.mean(vals))
        feats[f"{prefix}_{i}_std"] = float(np.std(vals))
        feats[f"{prefix}_{i}_min"] = float(np.min(vals))
        feats[f"{prefix}_{i}_max"] = float(np.max(vals))
    return feats


def extract_classical_features(filepath: str):
    y, sr = load_audio_mono(filepath, sr=24000)

    feats = {}

    feats["duration_sec"] = len(y) / sr
    feats["rms_mean"] = float(np.mean(librosa.feature.rms(y=y)))
    feats["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    feats["spectral_centroid_mean"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats["spectral_bandwidth_mean"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    feats["spectral_rolloff_mean"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    feats["onset_strength_mean"] = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feats.update(summarize_feature_matrix(mfcc, "mfcc"))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.update(summarize_feature_matrix(chroma, "chroma"))

    return feats


class EncodecFeatureExtractor:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, filepath: str):
        y, sr = load_audio_mono(filepath, sr=24000)

        wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        wav = convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
        wav = wav.to(self.device)

        encoded_frames = self.model.encode(wav)
        all_codes = []

        for frame in encoded_frames:
            codes = frame[0]
            codes = codes.squeeze(0).detach().cpu().numpy()
            all_codes.append(codes)

        codes = np.concatenate(all_codes, axis=-1)

        feats = {}
        feats["encodec_num_codebooks"] = int(codes.shape[0])
        feats["encodec_total_frames"] = int(codes.shape[1])

        for q in range(codes.shape[0]):
            q_vals = codes[q].astype(np.float32)
            feats[f"encodec_q{q}_mean"] = float(np.mean(q_vals))
            feats[f"encodec_q{q}_std"] = float(np.std(q_vals))
            feats[f"encodec_q{q}_min"] = float(np.min(q_vals))
            feats[f"encodec_q{q}_max"] = float(np.max(q_vals))

            hist, _ = np.histogram(q_vals, bins=16, range=(0, 1024), density=True)
            for b, v in enumerate(hist):
                feats[f"encodec_q{q}_hist_{b}"] = float(v)

        return feats


class CLAPFeatureExtractor:
    def __init__(self, device: str = None, model_name: str = "laion/clap-htsat-unfused"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, filepath: str):
        y, sr = load_audio_mono(filepath, sr=48000)

        inputs = self.processor(
            audios=[y],
            sampling_rate=sr,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        audio_embeds = self.model.get_audio_features(**inputs)
        emb = audio_embeds.squeeze(0).detach().cpu().numpy().astype(np.float32)

        feats = {}
        feats["clap_dim"] = int(emb.shape[0])

        for i, v in enumerate(emb):
            feats[f"clap_{i}"] = float(v)

        return feats


def extract_all_features(
    filepath: str,
    encodec_extractor: EncodecFeatureExtractor,
    clap_extractor: CLAPFeatureExtractor,
):
    feats = {}
    feats.update(extract_classical_features(filepath))
    feats.update(encodec_extractor.extract(filepath))
    feats.update(clap_extractor.extract(filepath))
    return feats