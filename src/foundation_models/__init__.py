from .clip import CLIP

def create_foundation_model(fm_name):
    """
    Create the foundation model given a foundation model name.
    It has the following methods attached to it:
        - fm.embed_img(img)
        - fm.embed_txt(prompts)
    Some foundation models may not have the embed_text method.

    Possible foundation model names:
        - 'clip': CLIP
        - 'dino': DINO
        - 'pixels': Pixels
    """
    if fm_name=='clip':
        fm = CLIP()
    else:
        raise ValueError(f"Unknown foundation model name: {fm_name}")
    return fm