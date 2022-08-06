
import numpy as np

def norm(img):
    norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    return norm

