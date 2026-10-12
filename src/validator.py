import numpy as np

def calculate_f1(target, preds):
    """
Calculate the F1 score for a single query.

Args:
    target: Iterable of ground-truth posting IDs.
    preds: Iterable of predicted posting IDs.

Returns:
    float: F1 score between 0.0 and 1.0.
"""
    target_matches = set(target)
    pred_matches = set(preds)

    common = target_matches & pred_matches
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_matches)
    recall = len(common) / len(target_matches)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def evaluate_predictions(val_df, preds):
    """
Evaluate retrieval predictions using mean F1 score.

Computes the F1 score for each validation query and returns
the average score across all validation samples.

Args:
    val_df: Validation DataFrame containing:
        - posting_id: Unique identifier of each query.
        - target_matches: List of ground-truth posting IDs.
    preds: Dictionary mapping query IDs to sets of predicted
        posting IDs, produced by multimodal_fusion().

Returns:
    float: Mean F1 score across all validation queries.
"""
    f1_scores = []
    for row in val_df.itertuples(index=False):
        id = row.posting_id
        target = row.target_matches
        f1 = calculate_f1(target, preds[id])
        f1_scores.append(f1)

    return np.mean(f1_scores)