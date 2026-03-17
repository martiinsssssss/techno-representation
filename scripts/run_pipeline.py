import argparse
from pathlib import Path
import pandas as pd

from src.data_utils import scan_audio_dataset
from src.features import EncodecFeatureExtractor, CLAPFeatureExtractor, extract_all_features
from src.analysis import (
    split_feature_sets,
    prepare_matrix,
    compute_pca,
    compute_umap,
    add_embedding_columns,
    compute_silhouette,
)
from src.visualization import save_scatter_plot


def process_representation(feat_df, feature_cols, rep_name, output_dir):
    X, _ = prepare_matrix(feat_df, feature_cols)

    Z_pca, _ = compute_pca(X)
    Z_umap, _ = compute_umap(X)

    pca_df = add_embedding_columns(feat_df, Z_pca, f"{rep_name}_pca")
    umap_df = add_embedding_columns(feat_df, Z_umap, f"{rep_name}_umap")

    pca_df.to_csv(output_dir / f"{rep_name}_pca.csv", index=False)
    umap_df.to_csv(output_dir / f"{rep_name}_umap.csv", index=False)

    save_scatter_plot(
        pca_df,
        f"{rep_name}_pca_x",
        f"{rep_name}_pca_y",
        "label",
        f"{rep_name.capitalize()} - PCA",
        f"figures/{rep_name}_pca.png"
    )

    save_scatter_plot(
        umap_df,
        f"{rep_name}_umap_x",
        f"{rep_name}_umap_y",
        "label",
        f"{rep_name.capitalize()} - UMAP",
        f"figures/{rep_name}_umap.png"
    )

    sil = compute_silhouette(X, feat_df["label"].values)
    return sil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/features")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning dataset...")
    df = scan_audio_dataset(args.data_dir)
    print(f"Found {len(df)} files")

    encodec_extractor = EncodecFeatureExtractor()
    clap_extractor = CLAPFeatureExtractor()

    rows = []
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] Processing {row['filename']}")
        feats = extract_all_features(
            row["filepath"],
            encodec_extractor=encodec_extractor,
            clap_extractor=clap_extractor,
        )
        rows.append({**row.to_dict(), **feats})

    feat_df = pd.DataFrame(rows)
    feat_csv = output_dir / "features.csv"
    feat_df.to_csv(feat_csv, index=False)
    print(f"Saved features to {feat_csv}")

    classical_cols, encodec_cols, clap_cols = split_feature_sets(feat_df)

    classical_sil = process_representation(feat_df, classical_cols, "classical", output_dir)
    encodec_sil = process_representation(feat_df, encodec_cols, "encodec", output_dir)
    clap_sil = process_representation(feat_df, clap_cols, "clap", output_dir)

    scores = pd.DataFrame(
        [
            {"feature_set": "classical", "silhouette_score": classical_sil},
            {"feature_set": "encodec", "silhouette_score": encodec_sil},
            {"feature_set": "clap", "silhouette_score": clap_sil},
        ]
    )
    scores.to_csv(output_dir / "silhouette_scores.csv", index=False)

    print("\nSilhouette scores:")
    print(scores)
    print("\nDone.")


if __name__ == "__main__":
    main()