## Mini Report (Auto-generated)

- Date: 2026-03-19 18:16
- Audio files: 399
- Categories: 11

### Silhouette Ranking

|   rank | feature_set   |   silhouette_score |
|-------:|:--------------|-------------------:|
|      1 | clap          |          0.300551  |
|      2 | classical     |          0.0946691 |
|      3 | encodec       |          0.0285665 |

### Supervised Benchmark (CV)

| feature_set   |   cv_accuracy_mean |   cv_accuracy_std |   cv_f1_macro_mean |   cv_f1_macro_std |
|:--------------|-------------------:|------------------:|-------------------:|------------------:|
| clap          |             0.9824 |            0.0129 |             0.984  |            0.0121 |
| classical     |             0.9623 |            0.0213 |             0.9666 |            0.0195 |
| encodec       |             0.7969 |            0.0221 |             0.8078 |            0.0096 |

### Key Observation

Best unsupervised separation: **clap** (silhouette = **0.301**).
Best supervised representation: **clap** (macro-F1 = **0.984**).