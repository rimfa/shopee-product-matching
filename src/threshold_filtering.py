import numpy as np

def threshold_filtering(neighbours, distances, gallery_ids, query_ids, threshold):
    """
Filter retrieved neighbors by cosine distance threshold.

Args:
    neighbours: Gallery indices of neighbors for each query.
    distances: Corresponding cosine distances.
    gallery_ids: Posting IDs of all gallery items.
    query_ids: Posting IDs of query items.
    threshold: Maximum allowed cosine distance.

Returns:
    Dictionary mapping each query ID to an array of matched posting IDs.

Raises:
    ValueError: If input arrays have inconsistent lengths.
"""
    if not (len(query_ids) == len(neighbours) == len(distances)):
        raise ValueError(
            "query_ids, neighbours и distances должны иметь одинаковую длину"
        )
    filtered_matches = {}
    for id, n, d in zip(query_ids, neighbours, distances):
        mask = d <= threshold
        correct_imgs = n[mask]
        filtered_matches[id] = gallery_ids[correct_imgs]
    return filtered_matches