from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy

def image_retrieval(train_emb, val_emb, radius):
  """
Find nearest neighbors using cosine distance.

Uses train and validation embeddings as the gallery
and validation embeddings as queries.

Args:
    train_emb: Training embeddings and posting IDs.
    val_emb: Validation embeddings and posting IDs.
    radius: Maximum cosine distance for candidate retrieval.

Returns:
    Dictionary containing query_ids, gallery_ids,
    neighbor indices, and corresponding distances.
"""
  emb_gallery = np.vstack((train_emb['embeddings'], val_emb['embeddings']))
  id_gallery = np.concatenate((train_emb['posting_ids'], val_emb['posting_ids']))

  emb_query = val_emb['embeddings']
  ind_query = val_emb['posting_ids']

  image_knn = NearestNeighbors(metric='cosine')
  image_knn.fit(emb_gallery)

  distances, indices = image_knn.radius_neighbors(emb_query, radius=radius)

  return {'query_ids':ind_query,
          'gallery_ids': id_gallery,
          'indices': indices,
          'distances': distances
          }


def text_retrieval(train_emb, val_emb, radius):

  emb_gallery = scipy.sparse.vstack((train_emb['embeddings'], val_emb['embeddings']))
  id_gallery = np.concatenate((train_emb['posting_ids'], val_emb['posting_ids']))

  emb_query = val_emb['embeddings']
  ind_query = val_emb['posting_ids']

  text_knn = NearestNeighbors(metric='cosine')
  text_knn.fit(emb_gallery)

  distances, indices = text_knn.radius_neighbors(emb_query, radius=radius)

  return {'query_ids':ind_query,
          'gallery_ids': id_gallery,
          'indices': indices,
          'distances': distances
          }