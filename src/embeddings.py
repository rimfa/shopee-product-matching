from tqdm import tqdm
import numpy as np
import torch
def get_embeddings(model, dataloader, device):

  embeddings = []
  model.eval()

  with torch.no_grad():
    for images in tqdm(dataloader, desc='getting embeddings'):
      images = images.to(device)
      emb = model(images).detach().cpu().numpy()

      embeddings.append(emb)

  embeddings = np.vstack(embeddings)
  return np.array(embeddings)