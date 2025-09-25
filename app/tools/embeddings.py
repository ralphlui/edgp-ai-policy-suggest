from typing import List

def embed_text(text: str, dim: int = 1536) -> List[float]:
    """
    Placeholder embedding function. Replace with your model call.
    Keep the dimension consistent with your index.
    """
    import math
    return [math.sin(i + len(text)) * 0.01 for i in range(dim)]
