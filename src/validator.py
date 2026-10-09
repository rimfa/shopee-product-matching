import numpy as np
from sklearn.neighbors import NearestNeighbors

def calculate_f1(target, preds):
    target_matches = set(target.split())
    pred_matches = set(preds)

    common = target_matches & pred_matches
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_matches)
    recall = len(common) / len(target_matches)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def multimodal(query_df, query_hash, query_tfidf, gallery_hash, gallery_tfidf, gallery_df):
    res = []

    gallery_post_ids = gallery_df['posting_id'].values
    max_bits = 9

    hash_knn = NearestNeighbors(metric='hamming')
    hash_knn.fit(gallery_hash)
    hash_dist, hash_idxs = hash_knn.radius_neighbors(query_hash, radius=max_bits/64)

    tfidf_knn = NearestNeighbors(metric='cosine')
    tfidf_knn.fit(gallery_tfidf)
    tfidf_dist, tfidf_idxs = tfidf_knn.kneighbors(query_tfidf, n_neighbors=50)

    thresholds = np.arange(0.1, 0.4, 0.03)

    for bit in range(4, max_bits+1):
        for t in thresholds:
            f1_scores = []

            for i in range(len(query_df)):
                h_ind = hash_idxs[i][hash_dist[i] <= bit/64]
                img_neighb = gallery_post_ids[h_ind]

                tfidf_ind = tfidf_idxs[i][tfidf_dist[i] <= t]
                txt_neighb = gallery_post_ids[tfidf_ind]

                preds = set(txt_neighb) | set(img_neighb)
                f1_scores.append(calculate_f1(query_df['target_matches'].iloc[i], preds))

            mean_f1 = np.mean(f1_scores)
            res.append([bit, round(t, 4), round(mean_f1, 4)])

    return res
