import argparse
import os
from typing import Tuple, List

import numpy as np
from datasets import load_dataset


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize each row vector to unit L2 norm. Handles zero vectors safely.
    """
    norms = np.linalg.norm(matrix, ord=2, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between rows of a and rows of b.
    a: (N, D), b: (M, D) -> (N, M)
    """
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return a_norm @ b_norm.T


def topk_pairs(sim: np.ndarray, k: int, largest: bool = True, exclude_same_index: bool = True) -> List[Tuple[int, int, float]]:
    """
    Return top-k pairs across the full similarity matrix.
    If exclude_same_index is True and matrix is square, skip pairs where i == j.
    """
    n_rows, n_cols = sim.shape
    pairs: List[Tuple[int, int, float]] = []

    for i in range(n_rows):
        row = sim[i]
        if exclude_same_index and n_rows == n_cols:
            # Avoid selecting identical index
            row = row.copy()
            row[i] = -np.inf if largest else np.inf

        # Argpartition for efficiency
        if largest:
            idx = np.argpartition(-row, range(min(k, n_cols)))[:k]
            idx = idx[np.argsort(-row[idx])]
        else:
            idx = np.argpartition(row, range(min(k, n_cols)))[:k]
            idx = idx[np.argsort(row[idx])]

        for j in idx:
            pairs.append((i, j, float(sim[i, j])))

    # Now we have up to N*k directional pairs; for square matrices we may want global top-k unique pairs
    # Sort globally
    pairs.sort(key=lambda t: t[2], reverse=largest)
    # Deduplicate mirror pairs by keeping (min(i,j), max(i,j)) for square case
    seen = set()
    unique: List[Tuple[int, int, float]] = []
    for i, j, s in pairs:
        key = (i, j) if n_rows != n_cols else (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        unique.append((i, j, s))
        if len(unique) >= k:
            break

    return unique


def extract_embeddings(dataset, field: str) -> np.ndarray:
    """
    Extract embeddings from a datasets.Dataset that stores Array2D shaped (1, D).
    Returns np.ndarray of shape (N, D).
    """
    # Each entry is a 2D array shaped (1, D)
    arr2d = dataset[field]
    # Convert to (N, D)
    matrix = np.stack([np.asarray(row).reshape(-1) for row in arr2d], axis=0)
    return matrix.astype(np.float32)


def save_csv(pairs: List[Tuple[int, int, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("i,j,similarity\n")
        for i, j, s in pairs:
            f.write(f"{i},{j},{s}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate most and least similar pairs in a HF dataset.")
    parser.add_argument("repo_id", type=str, help="HF dataset repo id or local path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load")
    parser.add_argument("--modality", type=str, default="image", choices=["image", "text", "cross"],
                        help="Similarity modality: image (image-image), text (text-text), cross (image-text)")
    parser.add_argument("--topk", type=int, default=50, help="Number of most similar pairs to output")
    parser.add_argument("--bottomk", type=int, default=50, help="Number of least similar pairs to output")
    parser.add_argument("--outdir", type=str, default="artifacts/similarity", help="Output directory for CSVs")
    args = parser.parse_args()

    ds = load_dataset(args.repo_id, split=args.split)

    if args.modality == "image":
        emb = extract_embeddings(ds, "image_embedding")
        sim = cosine_similarity_matrix(emb, emb)
        most = topk_pairs(sim, args.topk, largest=True, exclude_same_index=True)
        least = topk_pairs(sim, args.bottomk, largest=False, exclude_same_index=True)
        prefix = "image_image"
    elif args.modality == "text":
        emb = extract_embeddings(ds, "text_embedding")
        sim = cosine_similarity_matrix(emb, emb)
        most = topk_pairs(sim, args.topk, largest=True, exclude_same_index=True)
        least = topk_pairs(sim, args.bottomk, largest=False, exclude_same_index=True)
        prefix = "text_text"
    else:
        img = extract_embeddings(ds, "image_embedding")
        txt = extract_embeddings(ds, "text_embedding")
        sim = cosine_similarity_matrix(img, txt)
        most = topk_pairs(sim, args.topk, largest=True, exclude_same_index=False)
        least = topk_pairs(sim, args.bottomk, largest=False, exclude_same_index=False)
        prefix = "image_text"

    most_path = os.path.join(args.outdir, f"{prefix}_most_similar.csv")
    least_path = os.path.join(args.outdir, f"{prefix}_least_similar.csv")
    save_csv(most, most_path)
    save_csv(least, least_path)

    print(f"Saved most similar pairs -> {most_path}")
    print(f"Saved least similar pairs -> {least_path}")


if __name__ == "__main__":
    main()


