from tqdm import tqdm
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def get_embeddings(model, dataloader, device): #later rename to get_img_embeddings
  embeddings = []
  model.eval()

  with torch.no_grad():
    for images in tqdm(dataloader, desc='getting embeddings'):
      images = images.to(device)
      emb = model(images).detach().cpu().numpy()

      embeddings.append(emb)

  embeddings = np.vstack(embeddings)
  return np.array(embeddings)

def clean_text(string):
    string = re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9]', ' ', string)
    string = re.sub(r'\s+', ' ', string)
    return string.strip()


def get_tfidf(model, val_df, train_df):
    train_corpus = train_df['title'].fillna('').apply(clean_text)
    val_corpus = val_df['title'].fillna('').apply(clean_text)

    train_tfidf = model.fit_transform(train_corpus)
    val_tfidf = model.transform(val_corpus)
    train_emb = {
        'embeddings': train_tfidf,
        'posting_ids': train_df['posting_id'].to_numpy()
    }
    val_emb = {
        'embeddings': val_tfidf,
        'posting_ids': val_df['posting_id'].to_numpy()
    }
    return train_emb, val_emb


