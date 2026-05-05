from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import Product


DATABASE_PATH = Path(__file__).resolve().parent / "products.sqlite3"


class ProductDatabase:
	def __init__(self, db_path: Path = DATABASE_PATH, embedding_dimensions: int = 512) -> None:
		self.db_path = Path(db_path)
		self.embedding_dimensions = embedding_dimensions
		self.connection = self._connect()
		self._initialize()

	def close(self) -> None:
		self.connection.close()

	def get_product_import_states(self) -> dict[int, dict[str, Any]]:
		rows = self.connection.execute(
			"""
			SELECT
				p.id,
				p.content_hash,
				(SELECT COUNT(*) FROM product_images AS pi WHERE pi.product_id = p.id) AS image_count,
				(SELECT COUNT(*) FROM product_image_embeddings AS pie WHERE pie.product_id = p.id) AS embedding_count
			FROM products AS p
			"""
		).fetchall()
		return {
			row[0]: {
				"content_hash": row[1],
				"image_count": row[2],
				"embedding_count": row[3],
			}
			for row in rows
		}

	def search_similar_products_by_image(
		self,
		embedding: list[float],
		limit: int = 25,
		per_product_limit: int = 4,
	) -> list[dict]:
		rows = self.connection.execute(
			"""
			SELECT
				pie.image_id,
				pie.product_id,
				pie.sort_order,
				pie.distance,
				p.title,
				p.brand,
				p.seller,
				p.product_url,
				p.local_image_urls,
				p.price,
				p.currency,
				p.category,
				p.stitched_status
			FROM product_image_embeddings AS pie
			JOIN products AS p ON p.id = pie.product_id
			WHERE pie.embedding MATCH ?
			  AND k = ?
			ORDER BY pie.distance
			""",
			(json.dumps(embedding), limit),
		).fetchall()

		grouped: dict[int, dict] = {}
		for row in rows:
			product_id = row[1]
			product = grouped.setdefault(
				product_id,
				{
					"productId": product_id,
					"title": row[4],
					"brand": row[5],
					"seller": row[6],
					"productUrl": row[7],
					"localImagesUrl": json.loads(row[8]),
					"price": row[9],
					"currency": row[10],
					"category": row[11],
					"stitchedStatus": row[12],
					"matches": [],
				},
			)
			if len(product["matches"]) >= per_product_limit:
				continue
			product["matches"].append(
				{
					"imageId": row[0],
					"sortOrder": row[2],
					"distance": row[3],
				}
			)

		return sorted(
			grouped.values(),
			key=lambda product: min(match["distance"] for match in product["matches"]),
		)

	def upsert_product(self, product: Product) -> None:
		self.connection.execute(
			"""
			INSERT INTO products (
				id,
				title,
				brand,
				seller,
				product_url,
				image_urls,
				local_image_urls,
				price,
				currency,
				category,
				stitched_status,
				text_embedding,
				content_hash
			)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(id) DO UPDATE SET
				title = excluded.title,
				brand = excluded.brand,
				seller = excluded.seller,
				product_url = excluded.product_url,
				image_urls = excluded.image_urls,
				local_image_urls = excluded.local_image_urls,
				price = excluded.price,
				currency = excluded.currency,
				category = excluded.category,
				stitched_status = excluded.stitched_status,
				text_embedding = excluded.text_embedding,
				content_hash = excluded.content_hash,
				updated_at = CURRENT_TIMESTAMP
			""",
			(
				product.id,
				product.title,
				product.brand,
				product.seller,
				product.product_url,
				json.dumps(product.image_urls, ensure_ascii=False),
				json.dumps(product.local_image_urls, ensure_ascii=False),
				product.price,
				product.currency,
				product.category,
				product.stitched_status,
				json.dumps(product.text_embedding),
				product.content_hash,
			),
		)

		self.connection.execute("DELETE FROM product_images WHERE product_id = ?", (product.id,))
		self.connection.execute(
			"DELETE FROM product_image_embeddings WHERE product_id = ?",
			(product.id,),
		)

		for image in product.images:
			self.connection.execute(
				"""
				INSERT INTO product_images (
					id,
					product_id,
					image_url,
					local_image_url,
					sort_order
				)
				VALUES (?, ?, ?, ?, ?)
				""",
				(
					image.id,
					image.product_id,
					image.image_url,
					image.local_image_url,
					image.sort_order,
				),
			)
			if image.embedding:
				self.connection.execute(
					"""
					INSERT INTO product_image_embeddings (
						image_id,
						product_id,
						sort_order,
						embedding
					)
					VALUES (?, ?, ?, ?)
					""",
					(
						image.id,
						image.product_id,
						image.sort_order,
						json.dumps(image.embedding),
					),
				)

		self.connection.commit()

	def _connect(self) -> sqlite3.Connection:
		try:
			import sqlite_vec
		except ImportError as exc:
			raise RuntimeError("sqlite-vec is required. Run `uv sync` to install project dependencies.") from exc

		self.db_path.parent.mkdir(parents=True, exist_ok=True)
		connection = sqlite3.connect(self.db_path)
		connection.execute("PRAGMA journal_mode = WAL")
		connection.execute("PRAGMA synchronous = NORMAL")
		connection.enable_load_extension(True)
		sqlite_vec.load(connection)
		connection.enable_load_extension(False)
		return connection

	def _initialize(self) -> None:
		self.connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS products (
				id INTEGER PRIMARY KEY,
				title TEXT NOT NULL,
				brand TEXT NOT NULL,
				seller TEXT NOT NULL,
				product_url TEXT NOT NULL UNIQUE,
				image_urls TEXT NOT NULL,
				local_image_urls TEXT NOT NULL,
				price REAL,
				currency TEXT,
				category TEXT,
				stitched_status TEXT,
				text_embedding TEXT NOT NULL,
				content_hash TEXT NOT NULL DEFAULT '',
				updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
			)
			"""
		)
		self._ensure_products_columns()
		self.connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS product_images (
				id TEXT PRIMARY KEY,
				product_id INTEGER NOT NULL,
				image_url TEXT NOT NULL,
				local_image_url TEXT NOT NULL,
				sort_order INTEGER NOT NULL,
				FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
			)
			"""
		)
		self._ensure_image_embeddings_table()
		self.connection.execute(
			"CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)"
		)
		self.connection.execute(
			"CREATE INDEX IF NOT EXISTS idx_products_stitched_status ON products(stitched_status)"
		)
		self.connection.execute(
			"CREATE INDEX IF NOT EXISTS idx_product_images_product_id ON product_images(product_id)"
		)
		self.connection.commit()

	def _ensure_image_embeddings_table(self) -> None:
		row = self.connection.execute(
			"SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'product_image_embeddings'"
		).fetchone()
		expected_signature = f"FLOAT[{self.embedding_dimensions}]"
		if row and row[0] and expected_signature in row[0]:
			return
		if row:
			self.connection.execute("DROP TABLE product_image_embeddings")
		self.connection.execute(
			f"""
			CREATE VIRTUAL TABLE product_image_embeddings
			USING vec0(
				image_id TEXT PRIMARY KEY,
				product_id INTEGER,
				sort_order INTEGER,
				embedding FLOAT[{self.embedding_dimensions}]
			)
			"""
		)

	def _ensure_products_columns(self) -> None:
		columns = {
			row[1]
			for row in self.connection.execute("PRAGMA table_info(products)").fetchall()
		}
		if "content_hash" not in columns:
			self.connection.execute(
				"ALTER TABLE products ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
			)
