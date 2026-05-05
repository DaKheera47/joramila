from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from curl_cffi import requests

from .database import DATABASE_PATH, ProductDatabase
from .embeddings import ProductEmbedder
from .models import Product, ProductImage


BASE_URL = "https://us.alkaramstudio.com/"
SITEMAP_URL = "https://us.alkaramstudio.com/sitemap_products_1.xml?from=8053676605722&to=10218512646426"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_OUTPUT_DIR = Path(__file__).resolve().parent / "images"


class AlkaramScraper:
	def __init__(self, base_url: str = BASE_URL, brand: str = "Alkaram Studio") -> None:
		self.base_url = base_url.rstrip("/") + "/"
		self.brand = brand
		self.embedder = ProductEmbedder(project_root=PROJECT_ROOT)
		self._thread_local = threading.local()
		self._base_headers = {
			"User-Agent": (
				"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
				"(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
			),
			"Accept": (
				"text/html,application/xhtml+xml,application/xml;q=0.9,"
				"image/avif,image/webp,*/*;q=0.8"
			),
			"Accept-Language": "en-US,en;q=0.9",
			"Referer": self.base_url,
		}

	def ingest_products(
		self,
		db_path: Path = DATABASE_PATH,
		output_dir: Path = IMAGE_OUTPUT_DIR,
		max_workers: int = 8,
		limit: Optional[int] = None,
	) -> int:
		product_urls = self.fetch_sitemap_product_links()
		if limit is not None:
			product_urls = product_urls[: max(0, limit)]
		if not product_urls:
			return 0

		worker_count = max(1, min(max_workers, len(product_urls)))
		processed = 0
		saved = 0
		skipped = 0
		start_time = time.monotonic()
		total = len(product_urls)
		database = ProductDatabase(db_path=db_path, embedding_dimensions=self.embedder.dimensions)
		existing_states = database.get_product_import_states()

		try:
			with ThreadPoolExecutor(max_workers=worker_count) as executor:
				futures = {
					executor.submit(self.build_product, product_url, output_dir, existing_states): product_url
					for product_url in product_urls
				}
				for completed, future in enumerate(as_completed(futures), start=1):
					product_url = futures[future]
					elapsed = max(time.monotonic() - start_time, 0.001)
					avg_seconds = elapsed / completed
					remaining = total - completed
					eta_seconds = avg_seconds * remaining
					per_second = completed / elapsed

					try:
						product = future.result()
						if product.skip_db_write:
							skipped += 1
							status = (
								f"skipped unchanged id={product.id} restored_files={product.restored_files} "
								f"category={product.category or 'unknown'} "
								f"stitched={product.stitched_status or 'unknown'}"
							)
						else:
							product.text_embedding = self.embedder.embed_product_text(product)
							image_embeddings = self.embedder.embed_product_images(product)
							for image, embedding in zip(product.images, image_embeddings):
								image.embedding = embedding
							database.upsert_product(product)
							saved += 1
							status = (
								f"saved id={product.id} image_embeddings={self._embedded_image_count(product)} "
								f"category={product.category or 'unknown'} "
								f"stitched={product.stitched_status or 'unknown'}"
							)
						processed += 1
					except Exception as exc:
						status = f"failed: {exc.__class__.__name__}"

					print(
						(
							f"[{completed}/{total}] left={remaining} "
							f"eta={self._format_duration(eta_seconds)} "
							f"avg={self._format_duration(avg_seconds)} "
							f"rate={per_second:.2f}/s "
							f"{status} {product_url}"
						)
					)
		finally:
			database.close()

		print(f"Finished: processed={processed} saved={saved} skipped={skipped}")
		return processed

	def build_product(
		self,
		product_url: str,
		output_dir: Path = IMAGE_OUTPUT_DIR,
		existing_states: Optional[dict[int, dict[str, Any]]] = None,
	) -> Product:
		soup = self.soup(product_url)

		try:
			product_data = self.fetch_product_json(product_url)
		except Exception:
			product_data = {}

		title = (
			self._text_value(product_data.get("title"))
			or self._first_text(
				soup,
				["h1", "[data-product-title]", ".product__title", ".product-title", "meta[property='og:title']"],
			)
			or product_url.rsplit("/", 1)[-1].replace("-", " ").strip()
		)
		handle = self._text_value(product_data.get("handle")) or product_url.rstrip("/").rsplit("/", 1)[-1]
		seller = self._text_value(product_data.get("vendor")) or self.brand
		price = self._extract_price(product_data, soup)
		currency = self._extract_currency(soup)
		product_id = self._extract_product_id(product_data, product_url)
		image_urls = self.extract_product_images(product_url, soup=soup, product_data=product_data)
		local_image_urls = self.plan_local_image_urls(handle, image_urls, output_dir=output_dir)
		category = self._derive_category(handle, title)
		stitched_status = self._derive_stitched_status(handle, title)
		images = [
			ProductImage(
				id=f"{product_id}:{index:02d}",
				product_id=product_id,
				image_url=image_url,
				local_image_url=local_image_url,
				sort_order=index,
			)
			for index, (image_url, local_image_url) in enumerate(zip(image_urls, local_image_urls), start=1)
		]

		product = Product(
			id=product_id,
			title=title.strip(),
			brand=self.brand,
			seller=seller,
			product_url=product_url,
			image_urls=image_urls,
			local_image_urls=local_image_urls,
			images=images,
			price=price,
			currency=currency,
			category=category,
			stitched_status=stitched_status,
		)
		product.content_hash = self._product_content_hash(product)

		existing_state = (existing_states or {}).get(product_id)
		expected_embedding_count = min(len(product.images), self.embedder.max_images)
		if self._should_skip_product(product, existing_state, expected_embedding_count):
			product.restored_files = self.ensure_local_images(handle, image_urls, output_dir=output_dir)
			product.skip_db_write = True
			return product

		product.restored_files = self.ensure_local_images(handle, image_urls, output_dir=output_dir)
		return product

	def fetch(self, url: str) -> str:
		response = self._session().get(url, impersonate="chrome124", timeout=30)
		response.raise_for_status()
		return response.text

	def soup(self, url: str) -> BeautifulSoup:
		return BeautifulSoup(self.fetch(url), "html.parser")

	def fetch_sitemap_product_links(self, sitemap_url: str = SITEMAP_URL) -> list[str]:
		response = self._session().get(sitemap_url, impersonate="chrome124", timeout=30)
		response.raise_for_status()

		root = ET.fromstring(response.text)
		links: list[str] = []
		seen: set[str] = set()
		for node in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
			loc = node.findtext("{http://www.sitemaps.org/schemas/sitemap/0.9}loc", default="")
			if not loc or "/products/" not in loc:
				continue
			product_url = loc.strip().rstrip("/")
			if product_url not in seen:
				seen.add(product_url)
				links.append(product_url)
		return links

	def fetch_product_json(self, product_url: str) -> dict[str, Any]:
		json_url = self._product_json_url(product_url)
		response = self._session().get(json_url, impersonate="chrome124", timeout=30)
		response.raise_for_status()
		return response.json()

	def extract_product_images(
		self,
		product_url: str,
		soup: Optional[BeautifulSoup] = None,
		product_data: Optional[dict[str, Any]] = None,
	) -> list[str]:
		if soup is None:
			soup = self.soup(product_url)
		if product_data is None:
			try:
				product_data = self.fetch_product_json(product_url)
			except Exception:
				product_data = {}

		images: list[str] = []
		seen: set[str] = set()

		def add(raw_url: Optional[str]) -> None:
			normalized = self._normalize_image_url(raw_url, product_url)
			if not normalized or normalized in seen:
				return
			seen.add(normalized)
			images.append(normalized)

		for raw_url in product_data.get("images", []) if isinstance(product_data, dict) else []:
			add(raw_url)

		featured_image = product_data.get("featured_image") if isinstance(product_data, dict) else None
		if isinstance(featured_image, str):
			add(featured_image)

		for selector in [
			"meta[property='og:image']",
			"meta[property='og:image:secure_url']",
			"meta[name='twitter:image']",
		]:
			add(self._first_attr(soup, [selector], "content"))

		for image_node in soup.select(
			"product-gallery img, .product-gallery img, .product-gallery__image-list img, .product-gallery__thumbnail img"
		):
			add(image_node.get("src"))
			add(self._best_srcset_url(image_node.get("srcset"), product_url))

		return images

	def plan_local_image_urls(self, handle: str, image_urls: list[str], output_dir: Path = IMAGE_OUTPUT_DIR) -> list[str]:
		local_paths: list[str] = []
		for index, image_url in enumerate(image_urls, start=1):
			destination = self._image_destination(handle, index, image_url, output_dir)
			try:
				relative_path = destination.relative_to(PROJECT_ROOT)
			except ValueError:
				relative_path = destination
			local_paths.append(relative_path.as_posix())
		return local_paths

	def ensure_local_images(self, handle: str, image_urls: list[str], output_dir: Path = IMAGE_OUTPUT_DIR) -> int:
		output_dir.mkdir(parents=True, exist_ok=True)

		restored_files = 0
		for index, image_url in enumerate(image_urls, start=1):
			destination = self._image_destination(handle, index, image_url, output_dir)
			if not destination.exists() or destination.stat().st_size <= 0:
				response = self._session().get(image_url, impersonate="chrome124", timeout=60)
				response.raise_for_status()
				destination.write_bytes(response.content)
				restored_files += 1

		return restored_files

	def scrape(self, limit: int = 20, max_workers: int = 8) -> list[Product]:
		if limit <= 0:
			return []

		product_urls = self.fetch_sitemap_product_links()[:limit]
		worker_count = max(1, min(max_workers, len(product_urls)))
		products_by_index: dict[int, Product] = {}
		with ThreadPoolExecutor(max_workers=worker_count) as executor:
			futures = {
				executor.submit(self.build_product, product_url, IMAGE_OUTPUT_DIR): index
				for index, product_url in enumerate(product_urls)
			}
			for future in as_completed(futures):
				index = futures[future]
				try:
					product = future.result()
					product.text_embedding = self.embedder.embed_product_text(product)
					image_embeddings = self.embedder.embed_product_images(product)
					for image, embedding in zip(product.images, image_embeddings):
						image.embedding = embedding
					products_by_index[index] = product
				except Exception:
					continue
		return [products_by_index[index] for index in sorted(products_by_index)]

	def scrape_json(self, limit: int = 20, max_workers: int = 8) -> str:
		return json.dumps(
			[
				{
					"id": product.id,
					"title": product.title,
					"brand": product.brand,
					"seller": product.seller,
					"productUrl": product.product_url,
					"imagesUrl": product.image_urls,
					"localImagesUrl": product.local_image_urls,
					"price": product.price,
					"currency": product.currency,
					"category": product.category,
					"stitchedStatus": product.stitched_status,
					"textEmbedding": product.text_embedding,
					"images": [
						{
							"id": image.id,
							"imageUrl": image.image_url,
							"localImageUrl": image.local_image_url,
							"sortOrder": image.sort_order,
							"embedding": image.embedding,
						}
						for image in product.images
					],
				}
				for product in self.scrape(limit=limit, max_workers=max_workers)
			],
			indent=2,
			ensure_ascii=False,
		)

	def _session(self) -> requests.Session:
		session = getattr(self._thread_local, "session", None)
		if session is None:
			session = requests.Session()
			session.headers.update(self._base_headers)
			self._thread_local.session = session
		return session

	@staticmethod
	def _first_text(soup: BeautifulSoup, selectors: Iterable[str]) -> Optional[str]:
		for selector in selectors:
			node = soup.select_one(selector)
			if not node:
				continue
			if node.name == "meta":
				content = node.get("content")
				if content:
					return content.strip()
			text = node.get_text(" ", strip=True)
			if text:
				return re.sub(r"\s+", " ", text).strip()
		return None

	@staticmethod
	def _first_attr(soup: BeautifulSoup, selectors: Iterable[str], attr: str) -> Optional[str]:
		for selector in selectors:
			node = soup.select_one(selector)
			if not node:
				continue
			value = node.get(attr)
			if value:
				return value.strip()
		return None

	@staticmethod
	def _normalize_image_url(raw_url: Optional[str], base_url: str) -> Optional[str]:
		if not raw_url:
			return None
		parsed = urlparse(urljoin(base_url, raw_url.strip()))
		query_items = [
			(key, value)
			for key, value in parse_qsl(parsed.query, keep_blank_values=True)
			if key.lower() not in {"width", "w"}
		]
		return urlunparse(parsed._replace(query=urlencode(query_items, doseq=True), fragment=""))

	@staticmethod
	def _product_json_url(product_url: str) -> str:
		parsed = urlparse(product_url)
		path = parsed.path.rstrip("/")
		if not path.endswith(".js"):
			path = f"{path}.js"
		return urlunparse(parsed._replace(path=path, query="", fragment=""))

	@staticmethod
	def _best_srcset_url(srcset: Optional[str], base_url: str) -> Optional[str]:
		if not srcset:
			return None
		best_url: Optional[str] = None
		best_width = -1
		for candidate in srcset.split(","):
			parts = candidate.strip().split()
			if not parts:
				continue
			url = parts[0]
			width = 0
			if len(parts) > 1 and parts[1].endswith("w"):
				try:
					width = int(parts[1][:-1])
				except ValueError:
					width = 0
			if width >= best_width:
				best_width = width
				best_url = url
		return AlkaramScraper._normalize_image_url(best_url, base_url)

	@staticmethod
	def _image_destination(handle: str, index: int, image_url: str, output_dir: Path) -> Path:
		slug = AlkaramScraper._slugify(handle)
		extension = AlkaramScraper._image_extension(image_url)
		return output_dir / f"{slug}-{index:02d}{extension}"

	@staticmethod
	def _slugify(value: str) -> str:
		slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
		return slug or "product"

	@staticmethod
	def _image_extension(image_url: str) -> str:
		extension = Path(urlparse(image_url).path).suffix
		if extension:
			return extension.lower()
		return ".jpg"

	@staticmethod
	def _format_duration(seconds: float) -> str:
		seconds = max(0, int(round(seconds)))
		hours, remainder = divmod(seconds, 3600)
		minutes, secs = divmod(remainder, 60)
		if hours:
			return f"{hours:02d}:{minutes:02d}:{secs:02d}"
		return f"{minutes:02d}:{secs:02d}"

	@staticmethod
	def _text_value(value: Any) -> Optional[str]:
		if isinstance(value, str):
			stripped = value.strip()
			return stripped or None
		return None

	@staticmethod
	def _extract_product_id(product_data: dict[str, Any], product_url: str) -> int:
		raw_id = product_data.get("id")
		if isinstance(raw_id, int):
			return raw_id
		if isinstance(raw_id, str) and raw_id.isdigit():
			return int(raw_id)
		digest = hashlib.blake2b(product_url.encode("utf-8"), digest_size=8).digest()
		return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF

	@staticmethod
	def _extract_currency(soup: BeautifulSoup) -> str:
		for selector in [
			"meta[property='product:price:currency']",
			"meta[property='og:price:currency']",
			"meta[itemprop='priceCurrency']",
		]:
			node = soup.select_one(selector)
			if node and node.get("content"):
				return node["content"].strip().upper()
		return "USD"

	@staticmethod
	def _extract_price(product_data: dict[str, Any], soup: BeautifulSoup) -> Optional[float]:
		raw_price = product_data.get("price")
		if isinstance(raw_price, (int, float)):
			return round(float(raw_price) / 100.0, 2)
		if isinstance(raw_price, str):
			try:
				numeric = float(raw_price)
			except ValueError:
				numeric = None
			if numeric is not None:
				return round(numeric / 100.0, 2) if raw_price.isdigit() else round(numeric, 2)

		meta_price = soup.select_one("meta[property='product:price:amount']")
		if meta_price and meta_price.get("content"):
			try:
				return round(float(meta_price["content"].strip()), 2)
			except ValueError:
				return None
		return None

	@staticmethod
	def _product_content_hash(product: Product) -> str:
		payload = {
			"id": product.id,
			"title": product.title,
			"brand": product.brand,
			"seller": product.seller,
			"product_url": product.product_url,
			"image_urls": product.image_urls,
			"local_image_urls": product.local_image_urls,
			"price": product.price,
			"currency": product.currency,
			"category": product.category,
			"stitched_status": product.stitched_status,
		}
		encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
		return hashlib.blake2b(encoded, digest_size=16).hexdigest()

	@staticmethod
	def _should_skip_product(
		product: Product,
		existing_state: Optional[dict[str, Any]],
		expected_embedding_count: int,
	) -> bool:
		if not existing_state:
			return False
		return (
			existing_state.get("content_hash") == product.content_hash
			and existing_state.get("image_count") == len(product.images)
			and existing_state.get("embedding_count") == expected_embedding_count
		)

	@staticmethod
	def _embedded_image_count(product: Product) -> int:
		return sum(1 for image in product.images if image.embedding)

	@staticmethod
	def _derive_category(handle: str, title: str) -> str:
		filename = f"{handle} {title}".lower().replace("-", " ")
		category_rules = [
			("bridal-ish", ("bridal", "wedding", "lehenga", "gharara", "baraat")),
			("luxury pret", ("luxury pret", "luxury", "pret")),
			("formal", ("formal", "velvet", "organza", "chiffon", "silk", "festive")),
			("lawn", ("lawn", "cambric", "khaddar")),
			("casual", ("casual", "kurti", "tunic", "top", "daily wear")),
		]
		for category, keywords in category_rules:
			if any(keyword in filename for keyword in keywords):
				return category
		return "casual"

	@staticmethod
	def _derive_stitched_status(handle: str, title: str) -> str:
		filename = f"{handle} {title}".lower().replace("-", " ")
		if "made to order" in filename or "made-to-order" in filename or "mto" in filename:
			return "made-to-order"
		if "unstitched" in filename:
			return "unstitched"
		if re.search(r"\b[1-4]\s*pc\b", filename) or re.search(r"\b[1-4]\s*piece\b", filename):
			return "stitched"
		if any(keyword in filename for keyword in ("stitched", "rtw", "kurti", "ready to wear")):
			return "stitched"
		return "unstitched"


if __name__ == "__main__":
	scraper = AlkaramScraper()
	count = scraper.ingest_products()
	print(f"Ingested {count} products into {DATABASE_PATH}")
