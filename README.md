## Image Embeddings and Similarity Toolkit

Utilities to:
- Encode images with a vision model (DINOv3 via `transformers`)
- Optionally generate image descriptions with a small VLM (SmolVLM)
- Export enriched datasets to the Hugging Face Hub
- Render a 2D thumbnail map of image embeddings
- Compute most/least similar pairs using cosine similarity

### Environment
- Tested with: Python 3.12, CUDA 11.8, PyTorch 2.5.1, torchvision 0.20.1
- Conda env name used here: `image-model`

Create the environment (CUDA 11.8):
```bash
conda create -n image-model -y python=3.12
conda install -n image-model -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda run -n image-model pip install --no-cache-dir "transformers<5" timm accelerate datasets scikit-learn ipykernel
conda run -n image-model python -m ipykernel install --user --name image-model --display-name "Python (image-model)"
```

Verify CUDA works:
```bash
conda run -n image-model python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### Notebook workflow
File: `image-encoding.ipynb`
1) In Jupyter, select kernel: "Python (image-model)"
2) Run the cells to:
   - Load dataset (`rod-motta/wpp-images` by default)
   - Compute image embeddings with DINOv3
   - (Optional) Generate descriptions with SmolVLM
   - (Optional) Export enriched dataset to the Hub

Note: If you use OpenAI embeddings in the notebook, provide the key via environment variable and never commit it:
```bash
export OPENAI_API_KEY=your_key_here
```

### Render a 2D map of image embeddings
Script: `scripts/render_embedding_map.py`

Project embeddings to 2D (PCA default; t-SNE optional) and render a thumbnail canvas.
```bash
conda run -n image-model python scripts/render_embedding_map.py rod-motta/wpp-images-enriched \
  --split train \
  --method pca \
  --canvas-size 4096 \
  --thumb-size 64 \
  --outdir artifacts/embedding_map
```
Outputs:
- `artifacts/embedding_map/map_pca.png`
- `artifacts/embedding_map/coords_pca.csv`

Tips:
- Make images bigger: increase `--thumb-size` (e.g., 128)
- Less crowding: increase `--canvas-size` (e.g., 6000)
- Use t-SNE: `--method tsne --perplexity 30` (requires scikit-learn)

### Similarity reports
Script: `scripts/evaluate_similarity.py`

Compute most/least similar pairs for image-image, text-text, or image-text.
```bash
# Image-image
conda run -n image-model python scripts/evaluate_similarity.py rod-motta/wpp-images-enriched \
  --split train --modality image --topk 50 --bottomk 50 --outdir artifacts/similarity

# Text-text
conda run -n image-model python scripts/evaluate_similarity.py rod-motta/wpp-images-enriched \
  --split train --modality text --topk 50 --bottomk 50 --outdir artifacts/similarity

# Image-text (cross)
conda run -n image-model python scripts/evaluate_similarity.py rod-motta/wpp-images-enriched \
  --split train --modality cross --topk 50 --bottomk 50 --outdir artifacts/similarity
```
Outputs are CSV files under `artifacts/similarity/`.

### Repo hygiene
- `artifacts/`, dataset exports, caches, and secrets are gitignored via `.gitignore`
- Do not hardcode keys in notebooks or scripts; use environment variables

### License
TBD


