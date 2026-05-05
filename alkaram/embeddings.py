from __future__ import annotations

from pathlib import Path
from typing import Optional

import open_clip
from PIL import Image
import torch

from .models import Product


OPENCLIP_MODEL_NAME = "ViT-B-32"
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"
MAX_PRODUCT_IMAGES_FOR_EMBEDDING = 4


class ProductEmbedder:
	def __init__(
		self,
		model_name: str = OPENCLIP_MODEL_NAME,
		pretrained: str = OPENCLIP_PRETRAINED,
		project_root: Optional[Path] = None,
		max_images: int = MAX_PRODUCT_IMAGES_FOR_EMBEDDING,
	) -> None:
		self.model_name = model_name
		self.pretrained = pretrained
		self.project_root = Path(project_root).resolve() if project_root else None
		self.max_images = max_images
		self.device = self._detect_device()

		self.model, _, self.preprocess = open_clip.create_model_and_transforms(
			self.model_name,
			pretrained=self.pretrained,
			device=self.device,
		)
		self.model.eval()
		self.tokenizer = open_clip.get_tokenizer(self.model_name)
		self.dimensions = self._infer_dimensions()

	def embed_product_text(self, product: Product) -> list[float]:
		text = " | ".join(
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
		return self.embed_text(text)

	def embed_product_images(self, product: Product) -> list[list[float]]:
		embeddings: list[list[float]] = []
		for image in product.images[: self.max_images]:
			image_path = image.processed_image_url or image.local_image_url
			try:
				embeddings.append(self.embed_image_path(image_path))
			except Exception:
				continue
		return embeddings

	def embed_text(self, text: str) -> list[float]:
		normalized = text.strip()
		if not normalized:
			return [0.0] * self.dimensions

		tokens = self.tokenizer([normalized]).to(self.device)
		with torch.no_grad():
			features = self.model.encode_text(tokens)
		return self._tensor_to_unit_list(features[0])

	def embed_image_path(self, image_path: str | Path) -> list[float]:
		path = Path(image_path)
		if not path.is_absolute():
			if self.project_root is None:
				raise ValueError("project_root is required for relative image paths")
			path = self.project_root / path

		with Image.open(path) as image:
			tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)

		with torch.no_grad():
			features = self.model.encode_image(tensor)
		return self._tensor_to_unit_list(features[0])

	def _infer_dimensions(self) -> int:
		projection = getattr(self.model, "text_projection", None)
		if projection is not None:
			shape = getattr(projection, "shape", None)
			if shape:
				return int(shape[-1])

		with torch.no_grad():
			sample = self.model.encode_text(self.tokenizer(["test"]).to(self.device))
		return int(sample.shape[-1])

	@staticmethod
	def _detect_device() -> str:
		if torch.cuda.is_available():
			return "cuda"
		if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
			return "mps"
		return "cpu"

	def _tensor_to_unit_list(self, tensor: torch.Tensor) -> list[float]:
		vector = tensor.detach().float().cpu().tolist()
		return self._normalize(vector)

	@staticmethod
	def _normalize(vector: list[float]) -> list[float]:
		norm = sum(value * value for value in vector) ** 0.5 or 1.0
		return [value / norm for value in vector]
