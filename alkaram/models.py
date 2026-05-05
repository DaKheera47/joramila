from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProductImage:
	id: str
	product_id: int
	image_url: str
	local_image_url: str
	sort_order: int
	processed_image_url: str = ""
	embedding: list[float] = field(default_factory=list)


@dataclass
class Product:
	id: int
	title: str
	brand: str
	seller: str
	product_url: str
	image_urls: list[str] = field(default_factory=list)
	local_image_urls: list[str] = field(default_factory=list)
	processed_image_urls: list[str] = field(default_factory=list)
	images: list[ProductImage] = field(default_factory=list)
	price: Optional[float] = None
	currency: Optional[str] = None
	category: Optional[str] = None
	stitched_status: Optional[str] = None
	text_embedding: list[float] = field(default_factory=list)
	content_hash: str = ""
	skip_db_write: bool = False
	restored_files: int = 0
