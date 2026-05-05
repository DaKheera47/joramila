Install dependencies, then run the ingester with:

```bash
uv sync
uv run python main.py
```

Images are downloaded into `alkaram/images/`.
Products are stored in `alkaram/products.sqlite3`.
OpenCLIP embeddings are generated per product image and stored in a `sqlite-vec` virtual table inside that database.
Product text embeddings are stored separately for optional reranking.
The first OpenCLIP run may download model weights.
