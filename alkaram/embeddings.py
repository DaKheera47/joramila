from __future__ import annotations

import hashlib
import math
import re

from .models import Product


EMBEDDING_DIMENSIONS = 256


class ProductEmbedder:
	def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
		self.dimensions = dimensions

	def embed_product(self, product: Product) -> list[float]:
		return self.embed_text(
			" | ".join(
				filter(
					None,
					[
						product.title,
						product.brand,
						product.seller,
						product.category,
						product.stitched_status,
					],
				)
			)
		)

	def embed_text(self, text: str) -> list[float]:
		normalized = re.sub(r"\s+", " ", text.strip().lower())
		if not normalized:
			return [0.0] * self.dimensions

		vector: list[float] = []
		for index in range(self.dimensions):
			digest = hashlib.blake2b(f"{index}:{normalized}".encode("utf-8"), digest_size=8).digest()
			raw_value = int.from_bytes(digest, "big")
			scaled = (raw_value / ((1 << 64) - 1)) * 2.0 - 1.0
			vector.append(scaled)

		norm = math.sqrt(sum(value * value for value in vector)) or 1.0
		return [value / norm for value in vector]
