from __future__ import annotations

from html import escape
from pathlib import Path
import secrets

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .database import DATABASE_PATH, ProductDatabase
from .embeddings import ProductEmbedder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALKARAM_DIR = PROJECT_ROOT / "alkaram"
UPLOAD_DIR = ALKARAM_DIR / "uploads"
MAX_RESULTS = 12


app = FastAPI(title="JoraMila Demo")
app.mount("/files", StaticFiles(directory=PROJECT_ROOT), name="files")
_EMBEDDER: ProductEmbedder | None = None


def _page(body: str) -> HTMLResponse:
	return HTMLResponse(
		f"""
		<!doctype html>
		<html lang="en">
		<head>
			<meta charset="utf-8" />
			<meta name="viewport" content="width=device-width, initial-scale=1" />
			<title>JoraMila Demo</title>
			<style>
				:root {{
					color-scheme: light;
					font-family: Georgia, serif;
					background: #f5f0e8;
					color: #1d1b18;
				}}
				* {{ box-sizing: border-box; }}
				body {{
					margin: 0;
					padding: 24px;
					background:
						radial-gradient(circle at top left, #fff8ef 0, transparent 30rem),
						linear-gradient(180deg, #f9f3ea 0%, #f0e7da 100%);
				}}
				main {{
					max-width: 1100px;
					margin: 0 auto;
				}}
				h1 {{
					margin: 0 0 8px;
					font-size: 2.2rem;
				}}
				p {{
					margin: 0 0 16px;
				}}
				form {{
					display: flex;
					gap: 12px;
					flex-wrap: wrap;
					padding: 16px;
					border: 1px solid #d8c9b4;
					background: rgba(255, 252, 247, 0.92);
				}}
				input[type=file] {{
					flex: 1 1 260px;
				}}
				button {{
					border: 0;
					background: #1d1b18;
					color: #fff8ef;
					padding: 10px 16px;
					cursor: pointer;
				}}
				.grid {{
					display: grid;
					grid-template-columns: 280px 1fr;
					gap: 20px;
					margin-top: 20px;
				}}
				.query-card, .result-card {{
					background: rgba(255, 252, 247, 0.94);
					border: 1px solid #d8c9b4;
					padding: 14px;
				}}
				.query-card img, .result-card img {{
					width: 100%;
					display: block;
					aspect-ratio: 4 / 5;
					object-fit: cover;
					background: #ebe2d5;
				}}
				.results {{
					display: grid;
					grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
					gap: 16px;
				}}
				.meta {{
					font-size: 0.92rem;
					color: #5e564c;
				}}
				a {{
					color: inherit;
				}}
				.empty {{
					margin-top: 20px;
					padding: 16px;
					border: 1px dashed #b9ab96;
					background: rgba(255, 252, 247, 0.8);
				}}
				@media (max-width: 860px) {{
					.grid {{
						grid-template-columns: 1fr;
					}}
				}}
			</style>
		</head>
		<body>
			<main>{body}</main>
		</body>
		</html>
		"""
	)


def _home_body(message: str = "") -> str:
	return f"""
		<h1>JoraMila Demo</h1>
		<p>Upload a screenshot. It searches against stored product image embeddings and groups results by product.</p>
		<form action="/search" method="post" enctype="multipart/form-data">
			<input type="file" name="file" accept="image/*" required />
			<button type="submit">Search</button>
		</form>
		{f'<div class="empty">{escape(message)}</div>' if message else ''}
	"""


def _save_upload(file: UploadFile) -> Path:
	extension = Path(file.filename or "upload").suffix.lower() or ".png"
	UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
	filename = f"{secrets.token_hex(8)}{extension}"
	destination = UPLOAD_DIR / filename
	with destination.open("wb") as handle:
		handle.write(file.file.read())
	return destination


def _public_path(path: str | Path) -> str:
	return "/files/" + str(path).replace("\\", "/")


def _get_embedder() -> ProductEmbedder:
	global _EMBEDDER
	if _EMBEDDER is None:
		_EMBEDDER = ProductEmbedder(project_root=PROJECT_ROOT)
	return _EMBEDDER


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
	if not DATABASE_PATH.exists():
		return _page(_home_body("No database yet. Run `uv run python main.py --limit 10` first."))
	return _page(_home_body())


@app.post("/search", response_class=HTMLResponse)
def search(file: UploadFile = File(...)) -> HTMLResponse:
	if not DATABASE_PATH.exists():
		raise HTTPException(status_code=400, detail="Database not found. Run ingest first.")

	upload_path = _save_upload(file)
	embedder = _get_embedder()
	database = ProductDatabase(db_path=DATABASE_PATH, embedding_dimensions=embedder.dimensions)
	try:
		embedding = embedder.embed_image_path(upload_path)
		results = database.search_similar_products_by_image(embedding, limit=MAX_RESULTS)
	finally:
		database.close()

	upload_src = _public_path(upload_path.relative_to(PROJECT_ROOT))
	if not results:
		return _page(
			f"""
			{_home_body()}
			<div class="grid">
				<div class="query-card">
					<img src="{escape(upload_src)}" alt="Uploaded query image" />
				</div>
				<div class="empty">No results.</div>
			</div>
			"""
		)

	result_cards = []
	for result in results:
		match = result["matches"][0]
		image_index = max(0, match["sortOrder"] - 1)
		local_images = result["localImagesUrl"]
		image_src = _public_path(local_images[image_index]) if local_images else ""
		price = f"{result['currency']} {result['price']:.2f}" if result["price"] is not None else "No price"
		result_cards.append(
			f"""
			<article class="result-card">
				<img src="{escape(image_src)}" alt="{escape(result['title'])}" />
				<h3>{escape(result['title'])}</h3>
				<p class="meta">{escape(result['category'] or 'unknown')} · {escape(result['stitchedStatus'] or 'unknown')}</p>
				<p class="meta">{escape(price)} · distance {match['distance']:.4f}</p>
				<p><a href="{escape(result['productUrl'])}" target="_blank" rel="noreferrer">Open product</a></p>
			</article>
			"""
		)

	return _page(
		f"""
		{_home_body()}
		<div class="grid">
			<div class="query-card">
				<img src="{escape(upload_src)}" alt="Uploaded query image" />
				<p class="meta" style="margin-top: 12px;">Uploaded screenshot</p>
			</div>
			<div class="results">
				{''.join(result_cards)}
			</div>
		</div>
		"""
	)
