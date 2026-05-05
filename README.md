Install dependencies, then run the ingester with:

```bash
uv sync
uv run python main.py
```

Images are downloaded into `alkaram/images/`.
Products are stored in `alkaram/products.sqlite3`.
Embeddings are stored in a `sqlite-vec` virtual table inside that database.
