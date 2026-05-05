from __future__ import annotations

import argparse
from pathlib import Path

from alkaram.database import DATABASE_PATH
from alkaram.scraper import AlkaramScraper, IMAGE_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Ingest Alkaram products into SQLite.")
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
	parser.add_argument(
		"--db",
		type=Path,
		default=DATABASE_PATH,
		help="SQLite database path.",
	)
	parser.add_argument(
		"--reembed-existing",
		action="store_true",
		help="Rebuild processed images and embeddings from already-downloaded local files in the database.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	scraper = AlkaramScraper()
	if args.reembed_existing:
		print(
			f"Starting local re-embed: limit={args.limit if args.limit is not None else 'all'} "
			f"db={args.db}"
		)
		processed = scraper.reembed_existing_products(
			db_path=args.db,
			limit=args.limit,
			max_workers=args.workers,
		)
		print(f"Re-embedded {processed} products in {args.db}")
	else:
		print(
			f"Starting ingest: limit={args.limit if args.limit is not None else 'all'} "
			f"workers={args.workers} db={args.db}"
		)
		processed = scraper.ingest_products(
			db_path=args.db,
			output_dir=IMAGE_OUTPUT_DIR,
			max_workers=args.workers,
			limit=args.limit,
		)
		print(f"Processed {processed} products against {args.db}")


if __name__ == "__main__":
	main()
