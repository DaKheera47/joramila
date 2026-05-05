from pathlib import Path

from alkaram.scraper import AlkaramScraper, IMAGE_OUTPUT_DIR


def main() -> None:
    scraper = AlkaramScraper()
    downloaded = scraper.download_all_images()
    print(f"Downloaded {len(downloaded)} images to {Path(IMAGE_OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
