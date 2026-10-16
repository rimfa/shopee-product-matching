
import numpy as np
import pandas as pd


def load_data(csv_path):
    """Load the original dataset."""
    return pd.read_csv(csv_path)


def load_embeddings(npz_path):
    """Load embeddings and posting IDs from an NPZ file."""
    with np.load(npz_path, allow_pickle=True) as data:
        return {
            'posting_ids': data['posting_ids'],
            'embeddings': data['embeddings']
        }


def align_dataframe(df, posting_ids):
    """Align dataframe rows with the order of posting IDs."""
    return (
        df.set_index('posting_id', drop=False)
        .loc[posting_ids]
        .reset_index(drop=True)
    )


def align_embeddings(data, posting_ids):
    """Align embeddings with the selected posting ID order."""
    ids = pd.Index(data['posting_ids'])

    if not ids.is_unique:
        raise ValueError('Duplicate posting IDs in embeddings.')

    indices = ids.get_indexer(posting_ids)

    if (indices == -1).any() or len(ids) != len(posting_ids):
        raise ValueError('Embedding posting IDs do not match the selected split.')

    return {
        'posting_ids': np.asarray(posting_ids),
        'embeddings': data['embeddings'][indices]
    }


def add_target_matches(df):
    """Create ground-truth matches from label_group."""
    df = df.copy()

    group_matches = df.groupby('label_group')['posting_id'].agg(list)
    df['target_matches'] = df['label_group'].map(group_matches)

    return df


def load_project_data(
    csv_path,
    image_train_path=None,
    image_val_path=None,
    text_train_path=None,
    text_val_path=None,
    load_image=True,
    load_text=False
):
    """Load dataset, splits and selected embedding modalities."""

    if not load_image and not load_text:
        raise ValueError('Select at least one modality.')

    full_df = load_data(csv_path)

    image = None
    text = None

    if load_image:
        if image_train_path is None or image_val_path is None:
            raise ValueError('Image embedding paths are required.')

        image = {
            'train': load_embeddings(image_train_path),
            'val': load_embeddings(image_val_path)
        }

    if load_text:
        if text_train_path is None or text_val_path is None:
            raise ValueError('Text embedding paths are required.')

        text = {
            'train': load_embeddings(text_train_path),
            'val': load_embeddings(text_val_path)
        }

    # Use the selected modality to restore the original split
    reference = image if image is not None else text

    train_ids = reference['train']['posting_ids']
    val_ids = reference['val']['posting_ids']

    if set(train_ids) & set(val_ids):
        raise ValueError('Train and validation posting IDs overlap.')

    train_df = align_dataframe(full_df, train_ids)
    val_df = align_dataframe(full_df, val_ids)

    if set(train_df['label_group']) & set(val_df['label_group']):
        raise ValueError('Train and validation label groups overlap.')

    val_df = add_target_matches(val_df)

    # Ensure that all embeddings follow the same row order
    if image is not None:
        image['train'] = align_embeddings(image['train'], train_ids)
        image['val'] = align_embeddings(image['val'], val_ids)

    if text is not None:
        text['train'] = align_embeddings(text['train'], train_ids)
        text['val'] = align_embeddings(text['val'], val_ids)

    return {
        'full_df': full_df,
        'train_df': train_df,
        'val_df': val_df,
        'image': image,
        'text': text
    }
