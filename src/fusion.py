
def multimodal_fusion(image_matches, text_matches):
    """
Combine image and text retrieval results using set union.

Args:
    image_matches: Dictionary mapping query IDs to image-based matches.
    text_matches: Dictionary mapping query IDs to text-based matches.

Returns:
    Dictionary mapping each query ID to a set of unique
    posting IDs found by either modality.

Raises:
    ValueError: If query IDs differ between the two modalities.
"""

    if not (set(image_matches.keys()) == set(text_matches.keys())):
        raise ValueError (
            "keys image_matches в text_matches должны совпадать" 
        )
    preds = {}
    for (id, img) in (image_matches.items()):

        set_img = set(img)
        set_txt = set(text_matches[id])
        preds[id] = set_img.union(set_txt)


    return preds