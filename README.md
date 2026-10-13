# Shopee Product Matching

A multimodal product matching project based on the Kaggle [Shopee — Price Match Guarantee](https://www.kaggle.com/competitions/shopee-product-matching) competition.

The goal is to identify duplicate product listings using image and text features, with a focus on retrieval systems, metric learning, and reproducible ML experiments.

## Approach

The current baseline combines two independent retrieval branches:

- **Image:** Pretrained EfficientNet-B0 embeddings
- **Text:** TF-IDF with word-level n-grams
- **Retrieval:** Cosine distance and threshold filtering
- **Fusion:** Union of image and text matches

## Results

| Evaluation | Mean F1 |
|---|---:|
| Initial Validation | 0.72832 |
| 5-Fold Group Cross-Validation | **0.73710 ± 0.00744** |

*Results are based on local validation with different gallery configurations and are not directly comparable.*

## Notebooks

- [Hash Baseline](notebooks/baseline.ipynb)
- [EfficientNet-B0 Embeddings](notebooks/efficientnet_b0_embeddings.ipynb)
- [Multimodal Baseline & Cross-Validation](notebooks/multimodal_baseline.ipynb)

---

**Status:** Work in progress — baseline and cross-validation completed.
