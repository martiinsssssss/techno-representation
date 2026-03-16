import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap


METADATA_COLUMNS = ["filepath", "filename", "label"]


def split_feature_sets(df: pd.DataFrame):
    classical_cols = [
        c for c in df.columns
        if c not in METADATA_COLUMNS
        and not c.startswith("encodec_")
        and not c.startswith("clap_")
    ]

    encodec_cols = [c for c in df.columns if c.startswith("encodec_")]
    clap_cols = [c for c in df.columns if c.startswith("clap_") and c != "clap_dim"]

    return classical_cols, encodec_cols, clap_cols


def prepare_matrix(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def compute_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)
    return Z, pca


def compute_umap(X, n_components=2):
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=42,
        n_neighbors=min(10, max(2, X.shape[0] - 1)),
        min_dist=0.1,
    )
    Z = reducer.fit_transform(X)
    return Z, reducer


def compute_tsne(X, n_components=2):
    perplexity = min(10, max(2, X.shape[0] // 3))
    reducer = TSNE(
        n_components=n_components,
        random_state=42,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    Z = reducer.fit_transform(X)
    return Z, reducer


def add_embedding_columns(df: pd.DataFrame, Z, prefix: str):
    out = df.copy()
    out[f"{prefix}_x"] = Z[:, 0]
    out[f"{prefix}_y"] = Z[:, 1]
    return out


def compute_silhouette(X, labels):
    if len(set(labels)) < 2:
        return np.nan
    if len(labels) < 3:
        return np.nan
    return float(silhouette_score(X, labels))