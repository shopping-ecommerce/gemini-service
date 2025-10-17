import numpy as np

def cosine_similarity(vec1, vec2) -> float:
    v1 = np.asarray(vec1, dtype=float)
    v2 = np.asarray(vec2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)
