Install dependencies, then run the ingester with:

```bash
uv sync
uv run python main.py
```

Images are downloaded into `alkaram/images/`.
Background-removed WebP derivatives are written to `alkaram/images-processed/`.
Products are stored in `alkaram/products.sqlite3`.
OpenCLIP embeddings are generated per product image and stored in a `sqlite-vec` virtual table inside that database.
Product text embeddings are stored separately for optional reranking.
Re-imports skip unchanged products by comparing a stored content hash and existing image embedding counts.
The first OpenCLIP run may download model weights.
Background removal is done locally on macOS before image embeddings are recalculated.

Rebuild processed images and embeddings from already-downloaded files with:

```bash
uv run python main.py --reembed-existing
```

Use that command after changing background-removal or embedding logic.

Run the tiny demo UI with:

```bash
uv run uvicorn alkaram.webui:app --reload --port 8069
```

Then open `http://127.0.0.1:8069`, upload a screenshot, and inspect the grouped product matches.
