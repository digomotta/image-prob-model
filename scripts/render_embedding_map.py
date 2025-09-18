import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw


def extract_embeddings(dataset, field: str) -> np.ndarray:
    """
    Extract embeddings from a datasets.Dataset that stores Array2D shaped (1, D).
    Returns np.ndarray of shape (N, D) with dtype float32.
    """
    arr2d = dataset[field]
    matrix = np.stack([np.asarray(row).reshape(-1) for row in arr2d], axis=0)
    return matrix.astype(np.float32)


def compute_2d_projection(
    embeddings: np.ndarray,
    method: str = "pca",
    random_state: int = 42,
    perplexity: float = 30.0,
) -> np.ndarray:
    """
    Project high-dimensional embeddings (N, D) to 2D using PCA or t-SNE.
    PCA uses NumPy SVD to avoid external dependencies.
    t-SNE requires scikit-learn if selected; falls back to PCA if unavailable.
    Returns array of shape (N, 2).
    """
    method = method.lower()
    if method == "pca":
        x = embeddings.astype(np.float64, copy=False)
        mean = x.mean(axis=0, keepdims=True)
        x_centered = x - mean
        # Compute top-2 principal components via SVD
        # x_centered = U S Vt; projection = x_centered @ Vt.T[:, :2]
        _, _, v_t = np.linalg.svd(x_centered, full_matrices=False)
        components = v_t[:2].T  # (D, 2)
        coords = x_centered @ components  # (N, 2)
        return coords.astype(np.float32)

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE  # type: ignore
        except Exception:
            # Fallback to PCA if TSNE is not available
            return compute_2d_projection(embeddings, method="pca")

        tsne = TSNE(
            n_components=2,
            perplexity=float(perplexity),
            init="pca",
            learning_rate="auto",
            random_state=int(random_state),
            n_iter=1000,
            verbose=0,
        )
        coords = tsne.fit_transform(embeddings)
        return coords.astype(np.float32)

    # Default fallback
    return compute_2d_projection(embeddings, method="pca")


def normalize_to_canvas(
    coords: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    padding: int = 8,
) -> np.ndarray:
    """
    Linearly map 2D coordinates to canvas pixel coordinates with padding.
    Returns integer pixel coordinates of shape (N, 2).
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (N, 2)")

    x = coords[:, 0]
    y = coords[:, 1]

    # Avoid zero ranges
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)

    w = canvas_width - 2 * padding
    h = canvas_height - 2 * padding

    x_pix = (padding + x_norm * w).astype(np.int32)
    # Invert Y so larger values appear lower on the canvas
    y_pix = (padding + (1.0 - y_norm) * h).astype(np.int32)

    return np.stack([x_pix, y_pix], axis=1)


def load_image_from_example(example) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Attempt to load a PIL image and return (image, path_hint).
    Handles datasets.Image objects, file paths, and dicts with 'path'.
    """
    img = example.get("image")
    if img is None:
        return None, None

    # PIL Image object
    if isinstance(img, Image.Image):
        path_hint = getattr(img, "filename", None)
        return img, path_hint

    # HF datasets Image returns dict-like with 'path'
    if isinstance(img, dict) and "path" in img:
        path = img["path"]
        try:
            pil = Image.open(path).convert("RGB")
            return pil, path
        except Exception:
            return None, path

    # String path
    if isinstance(img, str):
        try:
            pil = Image.open(img).convert("RGB")
            return pil, img
        except Exception:
            return None, img

    return None, None


def paste_thumbnails(
    images: List[Optional[Image.Image]],
    coords_px: np.ndarray,
    canvas_size: Tuple[int, int],
    thumb_size: int,
    add_border: bool = True,
    border_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Paste thumbnails onto a blank white canvas at the given pixel coordinates.
    Coordinates refer to the top-left corner of each thumbnail.
    """
    canvas_w, canvas_h = canvas_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, (img, (x, y)) in enumerate(zip(images, coords_px)):
        if img is None:
            continue
        copy = img.copy()
        copy.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)

        # Shift to center the thumbnail around the point instead of top-left
        x0 = int(x - copy.width // 2)
        y0 = int(y - copy.height // 2)

        # Clamp to canvas bounds
        x0 = max(0, min(x0, canvas_w - copy.width))
        y0 = max(0, min(y0, canvas_h - copy.height))

        if add_border:
            draw.rectangle(
                [x0 - 1, y0 - 1, x0 + copy.width, y0 + copy.height],
                outline=border_color,
                width=1,
            )

        canvas.paste(copy, (x0, y0))

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a 2D image embedding map from a Hugging Face dataset.",
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HF dataset repo id (e.g., user/dataset) or local dataset path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load",
    )
    parser.add_argument(
        "--embedding-field",
        type=str,
        default="image_embedding",
        help="Dataset field name that stores embeddings as Array2D (1, D)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pca",
        choices=["pca", "tsne"],
        help="Projection method to 2D (default: pca)",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=4096,
        help="Square canvas size in pixels (default: 4096)",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=64,
        help="Thumbnail max size in pixels (default: 64)",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (only used when --method tsne)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="artifacts/embedding_map",
        help="Directory to save the map image and coordinates CSV",
    )
    args = parser.parse_args()

    ds = load_dataset(args.repo_id, split=args.split)

    emb = extract_embeddings(ds, args.embedding_field)
    coords2d = compute_2d_projection(
        emb,
        method=args.method,
        perplexity=args.perplexity,
    )

    coords_px = normalize_to_canvas(
        coords2d, canvas_width=args.canvas_size, canvas_height=args.canvas_size, padding=16
    )

    images: List[Optional[Image.Image]] = []
    path_hints: List[Optional[str]] = []
    for i in range(len(ds)):
        pil, path_hint = load_image_from_example(ds[i])
        images.append(pil)
        path_hints.append(path_hint)

    canvas = paste_thumbnails(
        images=images,
        coords_px=coords_px,
        canvas_size=(args.canvas_size, args.canvas_size),
        thumb_size=args.thumb_size,
    )

    os.makedirs(args.outdir, exist_ok=True)
    map_path = os.path.join(args.outdir, f"map_{args.method}.png")
    csv_path = os.path.join(args.outdir, f"coords_{args.method}.csv")

    canvas.save(map_path)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,x,y,image_path\n")
        for idx, (xy, p) in enumerate(zip(coords_px, path_hints)):
            f.write(f"{idx},{int(xy[0])},{int(xy[1])},{p or ''}\n")

    print(f"Saved 2D map -> {map_path}")
    print(f"Saved coordinates -> {csv_path}")


if __name__ == "__main__":
    main()


