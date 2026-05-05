from __future__ import annotations

import json
import mimetypes
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from curl_cffi import requests


BASE_URL = "https://us.alkaramstudio.com/"
SITEMAP_URL = "https://us.alkaramstudio.com/sitemap_products_1.xml?from=8053676605722&to=10218512646426"
IMAGE_OUTPUT_DIR = Path(__file__).resolve().parent / "images"


@dataclass
class Product:
	title: str
	url: str
	price: Optional[str] = None
	image: Optional[str] = None
	images: list[str] = field(default_factory=list)


class AlkaramScraper:
	def __init__(self, base_url: str = BASE_URL) -> None:
		self.base_url = base_url.rstrip("/") + "/"
		self.session = requests.Session()
		self.session.headers.update(
			{
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
		)

	def fetch(self, url: str) -> str:
		response = self.session.get(url, impersonate="chrome124", timeout=30)
		response.raise_for_status()
		return response.text

	def soup(self, url: str) -> BeautifulSoup:
		return BeautifulSoup(self.fetch(url), "html.parser")

	def homepage(self) -> BeautifulSoup:
		return self.soup(self.base_url)

	def fetch_sitemap_product_links(self, sitemap_url: str = SITEMAP_URL) -> list[str]:
		response = self.session.get(sitemap_url, impersonate="chrome124", timeout=30)
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

	def fetch_product_json(self, product_url: str) -> dict:
		json_url = self._product_json_url(product_url)
		response = self.session.get(json_url, impersonate="chrome124", timeout=30)
		response.raise_for_status()
		return response.json()

	def extract_category_links(self) -> list[str]:
		soup = self.homepage()
		links: list[str] = []
		seen: set[str] = set()

		for a in soup.select("a[href]"):
			href = (a.get("href") or "").strip()
			if not href:
				continue
			absolute = urljoin(self.base_url, href)
			if self.base_url not in absolute:
				continue
			if any(skip in absolute for skip in ("/account", "/cart", "/checkout", "/search")):
				continue
			if absolute not in seen:
				seen.add(absolute)
				links.append(absolute)
		return links

	def extract_product_links(self, category_url: str) -> list[str]:
		soup = self.soup(category_url)
		links: list[str] = []
		seen: set[str] = set()

		for a in soup.select("a[href]"):
			href = (a.get("href") or "").strip()
			if not href:
				continue
			absolute = urljoin(category_url, href).split("?")[0].rstrip("/")
			if "/products/" in absolute or "/product/" in absolute:
				if absolute not in seen:
					seen.add(absolute)
					links.append(absolute)
		return links

	def parse_product(self, product_url: str) -> Product:
		soup = self.soup(product_url)

		title = self._first_text(
			soup,
			["h1", "[data-product-title]", ".product__title", ".product-title", "meta[property='og:title']"],
		) or product_url.rsplit("/", 1)[-1].replace("-", " ").strip()

		price = self._first_text(
			soup,
			[
				"[class*='price']",
				"[data-product-price]",
				"meta[property='product:price:amount']",
			],
		)
		meta_price = soup.select_one("meta[property='product:price:amount']")
		if meta_price and meta_price.get("content"):
			price = meta_price.get("content")

		image = self._first_attr(
			soup,
			["meta[property='og:image']", "[data-product-image]", ".product__media img", "img"],
			"content",
		) or self._first_attr(soup, [".product__media img", "img"], "src")

		images = self.extract_product_images(product_url, soup=soup)
		if not image and images:
			image = images[0]

		return Product(title=title.strip(), url=product_url, price=price, image=image, images=images)

	def extract_product_images(self, product_url: str, soup: Optional[BeautifulSoup] = None) -> list[str]:
		if soup is None:
			soup = self.soup(product_url)
		images: list[str] = []
		seen: set[str] = set()

		def add(raw_url: Optional[str]) -> None:
			normalized = self._normalize_image_url(raw_url, product_url)
			if not normalized or normalized in seen:
				return
			seen.add(normalized)
			images.append(normalized)

		try:
			product_data = self.fetch_product_json(product_url)
		except Exception:
			product_data = {}

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

	def download_product_images(self, product_url: str, output_dir: Path = IMAGE_OUTPUT_DIR) -> list[Path]:
		product = self.parse_product(product_url)
		output_dir.mkdir(parents=True, exist_ok=True)

		downloaded: list[Path] = []
		slug = self._slugify(product.url.rstrip("/").rsplit("/", 1)[-1]) or self._slugify(product.title)

		for index, image_url in enumerate(product.images, start=1):
			extension = self._image_extension(image_url)
			destination = output_dir / f"{slug}-{index:02d}{extension}"
			if destination.exists() and destination.stat().st_size > 0:
				downloaded.append(destination)
				continue

			response = self.session.get(image_url, impersonate="chrome124", timeout=60)
			response.raise_for_status()
			if not extension:
				destination = destination.with_suffix(
					self._content_type_extension(response.headers.get("Content-Type", ""))
				)
			destination.write_bytes(response.content)
			downloaded.append(destination)

		return downloaded

	def download_all_images(self, output_dir: Path = IMAGE_OUTPUT_DIR) -> list[Path]:
		downloaded: list[Path] = []
		for product_url in self.fetch_sitemap_product_links():
			try:
				downloaded.extend(self.download_product_images(product_url, output_dir=output_dir))
			except Exception:
				continue
		return downloaded

	def scrape(self, limit: int = 20) -> list[Product]:
		products: list[Product] = []
		for product_url in self.fetch_sitemap_product_links():
			try:
				products.append(self.parse_product(product_url))
			except Exception:
				continue
			if len(products) >= limit:
				return products
		return products

	def scrape_json(self, limit: int = 20) -> str:
		return json.dumps([asdict(p) for p in self.scrape(limit=limit)], indent=2, ensure_ascii=False)

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
		if path.endswith(".js"):
			json_path = path
		else:
			json_path = f"{path}.js"
		return urlunparse(parsed._replace(path=json_path, query="", fragment=""))

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
	def _content_type_extension(content_type: str) -> str:
		mime_type = content_type.split(";", 1)[0].strip().lower()
		if not mime_type:
			return ".jpg"
		extension = mimetypes.guess_extension(mime_type)
		return extension or ".jpg"


if __name__ == "__main__":
	scraper = AlkaramScraper()
	paths = scraper.download_all_images()
	print(f"Downloaded {len(paths)} images to {IMAGE_OUTPUT_DIR}")
