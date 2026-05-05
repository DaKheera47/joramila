from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .embeddings import EMBEDDING_DIMENSIONS
from .models import Product


DATABASE_PATH = Path(__file__).resolve().parent / "products.sqlite3"


class ProductDatabase:
	def __init__(self, db_path: Path = DATABASE_PATH, embedding_dimensions: int = EMBEDDING_DIMENSIONS) -> None:
		self.db_path = Path(db_path)
		self.embedding_dimensions = embedding_dimensions
		self.connection = self._connect()
		self._initialize()

	def close(self) -> None:
		self.connection.close()

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
				stitched_status
			)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
			),
		)
		self.connection.execute("DELETE FROM product_embeddings WHERE product_id = ?", (product.id,))
		self.connection.execute(
			"INSERT INTO product_embeddings (product_id, embedding) VALUES (?, ?)",
			(product.id, json.dumps(product.embedding)),
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
				updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
			)
			"""
		)
		self.connection.execute(
			f"""
			CREATE VIRTUAL TABLE IF NOT EXISTS product_embeddings
			USING vec0(
				product_id INTEGER PRIMARY KEY,
				embedding FLOAT[{self.embedding_dimensions}]
			)
			"""
		)
		self.connection.execute(
			"CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)"
		)
		self.connection.execute(
			"CREATE INDEX IF NOT EXISTS idx_products_stitched_status ON products(stitched_status)"
		)
		self.connection.commit()
