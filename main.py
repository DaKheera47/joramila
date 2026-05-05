from __future__ import annotations

import argparse

from alkaram.scraper import AlkaramScraper, IMAGE_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download Alkaram product images.")
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Maximum number of products to process. Omit to process the full sitemap.",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=8,
		help="Number of concurrent workers to use.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	scraper = AlkaramScraper()
	print(
		f"Starting download: limit={args.limit if args.limit is not None else 'all'} "
		f"workers={args.workers}"
	)
	downloaded = scraper.download_all_images(output_dir=IMAGE_OUTPUT_DIR, max_workers=args.workers, limit=args.limit)
	print(f"Downloaded {len(downloaded)} images to {IMAGE_OUTPUT_DIR}")


if __name__ == "__main__":
	main()
