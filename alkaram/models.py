from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Product:
	id: int
	title: str
	brand: str
	seller: str
	product_url: str
	image_urls: list[str] = field(default_factory=list)
	local_image_urls: list[str] = field(default_factory=list)
	price: Optional[float] = None
	currency: Optional[str] = None
	category: Optional[str] = None
	stitched_status: Optional[str] = None
	embedding: list[float] = field(default_factory=list)
