import pandas as pd
import numpy as np
from umap import UMAP

def get_cluster_visualization_data(embeddings, labels, texts):
    """
    Generate cluster visualization data using UMAP for dimensionality reduction.
    
    Args:
        embeddings: List or array of embeddings (can be 1D or 2D)
        labels: Cluster labels for each embedding (NumPy array, list, or None)
        texts: Text associated with each embedding (list or None)
    
    Returns:
        DataFrame with UMAP coordinates, cluster labels, and texts
    """
    # Handle None input or empty embeddings
    if embeddings is None:
        return pd.DataFrame(columns=["x", "y", "cluster", "text"])
    
    embeddings = np.array(embeddings)
    
    # Check for empty arrays (0 points)
    if embeddings.size == 0:
        return pd.DataFrame(columns=["x", "y", "cluster", "text"])
    
    # Convert 1D array to 2D (single point)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    n_points = len(embeddings)
    
    # Set UMAP parameters
    n_neighbors = min(15, n_points - 1) if n_points > 1 else 1
    
    # Handle single point case
    if n_points == 1:
        umap_result = np.array([[0, 0]])
    else:
        umap = UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=n_neighbors,
            min_dist=0.1
        )
        umap_result = umap.fit_transform(embeddings)
    
    # Create DataFrame with results
    df = pd.DataFrame(umap_result, columns=["x", "y"])
    
    # FIXED: Use explicit None check instead of truthy check
    df["cluster"] = [str(l) for l in labels] if labels is not None else ["0"] * n_points
    df["text"] = texts if texts is not None else [""] * n_points
    
    # Filter out noise points (cluster = -1)
    if "-1" in df["cluster"].values:
        df = df[df["cluster"] != "-1"]
    
    return df

from pymilvus import Collection
import numpy as np

def get_all_embeddings_and_texts(collection_name):
    collection = Collection(collection_name)
    collection.load()
    # 批量取出所有数据
    iterator = collection.query_iterator(
        batch_size=100, expr="id > 0", output_fields=["id", "embedding", "content"]
    )
    ids, embeddings, texts = [], [], []
    while True:
        batch = iterator.next()
        if len(batch) == 0:
            break
        for data in batch:
            ids.append(data["id"])
            embeddings.append(data["embedding"])
            texts.append(data["content"])
    return ids, np.array(embeddings), texts