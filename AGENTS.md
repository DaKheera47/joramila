# JoraMila Agent Notes

## Purpose

This repo ingests Alkaram product data into a local SQLite catalog for image-first retrieval.

Core flow:
- scrape product URLs from the sitemap
- download product images to `alkaram/images/`
- derive background-removed WebP images in `alkaram/images-processed/`
- generate OpenCLIP embeddings per product image
- store product metadata in SQLite
- store image embeddings in `sqlite-vec`
- expose a tiny FastAPI demo UI for screenshot-to-product search

## Key Files

- `main.py`
  Runs ingestion from the CLI.
- `alkaram/scraper.py`
  Sitemap fetch, product parsing, image download, progress logging, and ingestion orchestration.
- `alkaram/image_processing.py`
  macOS background removal and WebP conversion pipeline.
- `alkaram/remove_background.swift`
  Vision-based foreground extraction helper compiled locally on macOS.
- `alkaram/database.py`
  SQLite schema, upserts, and grouped image search.
- `alkaram/embeddings.py`
  OpenCLIP embedding generation.
- `alkaram/webui.py`
  Minimal demo UI on FastAPI.
- `alkaram/products.sqlite3`
  Local catalog database.

## Data Model

`products` stores catalog metadata.

`product_images` stores one row per downloaded product image.
- includes both original local image path and processed local image path

`product_image_embeddings` is the primary vector index.
- one embedding per product image
- this is the main retrieval path
- query image -> nearest product images -> grouped back to products
- embeddings should be built from processed background-removed images, not raw originals, unless explicitly overridden

`text_embedding` on `products` is secondary and should only help reranking or future hybrid search.

Do not collapse product images back into a single averaged product embedding unless explicitly requested.

## Runbook

Install deps:

```bash
uv sync
```

Ingest products:

```bash
uv run python main.py --limit 10 --workers 4
```

Rebuild processed images and embeddings from existing local files:

```bash
uv run python main.py --reembed-existing
```

Use that command after changing preprocessing or embedding behavior.
`--workers` should be used to parallelize the preprocessing side of this rebuild flow.

Run the demo UI:

```bash
uv run uvicorn alkaram.webui:app --reload --port 8069
```

## Working Rules

- Keep the retrieval model image-first.
- Prefer updating existing product rows instead of creating duplicate products.
- Image files may be reused if they already exist locally.
- Re-imports should skip unchanged products when the stored content hash and image embedding counts still match.
- Query-time images should go through the same preprocessing path as indexed product images.
- Database writes should remain deterministic and idempotent where practical.
- Keep the web UI intentionally small unless the user asks for a richer product.

## Change Discipline

When making future changes, update this `AGENTS.md` in the same task if the change affects:
- architecture
- data model
- retrieval logic
- run commands
- ports
- storage locations
- key dependencies
- operational assumptions

Do not let `AGENTS.md` drift behind the codebase.
