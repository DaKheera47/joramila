from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_IMAGE_OUTPUT_DIR = Path(__file__).resolve().parent / "images-processed"
REMOVE_BACKGROUND_SWIFT = Path(__file__).resolve().parent / "remove_background.swift"
REMOVE_BACKGROUND_BINARY = Path("/private/tmp/joramila-remove-background")


class ProductImagePreprocessor:
	def __init__(
		self,
		project_root: Path = PROJECT_ROOT,
		output_dir: Path = PROCESSED_IMAGE_OUTPUT_DIR,
		swift_script: Path = REMOVE_BACKGROUND_SWIFT,
		binary_path: Path = REMOVE_BACKGROUND_BINARY,
	) -> None:
		self.project_root = Path(project_root).resolve()
		self.output_dir = Path(output_dir)
		self.swift_script = Path(swift_script)
		self.binary_path = Path(binary_path)

	def preprocess_relative_path(self, image_path: str | Path) -> str:
		source_path = self._absolute_path(image_path)
		destination = self._planned_destination(source_path)
		self._ensure_processed_image(source_path, destination)
		return self._relative_to_project_root(destination)

	def planned_relative_path(self, image_path: str | Path) -> str:
		source_path = self._absolute_path(image_path)
		destination = self._planned_destination(source_path)
		return self._relative_to_project_root(destination)

	def preprocess_many(self, image_paths: list[str | Path]) -> list[str]:
		return [self.preprocess_relative_path(path) for path in image_paths]

	def _planned_destination(self, source_path: Path) -> Path:
		return self.output_dir / f"{source_path.stem}.webp"

	def _relative_to_project_root(self, path: Path) -> str:
		try:
			return path.relative_to(self.project_root).as_posix()
		except ValueError:
			return path.as_posix()

	def _ensure_processed_image(self, source_path: Path, destination: Path) -> None:
		destination.parent.mkdir(parents=True, exist_ok=True)
		required_mtime = max(source_path.stat().st_mtime, self.swift_script.stat().st_mtime)
		if destination.exists() and destination.stat().st_mtime >= required_mtime:
			return

		self._ensure_background_remover_binary()
		with tempfile.TemporaryDirectory(prefix="joramila-bg-") as tmp_dir:
			mask_png = Path(tmp_dir) / f"{source_path.stem}.png"
			self._run_background_removal(source_path, mask_png)
			with Image.open(mask_png) as image:
				image.convert("RGBA").save(destination, format="WEBP", quality=85, method=6)

	def _ensure_background_remover_binary(self) -> None:
		if self.binary_path.exists() and self.binary_path.stat().st_mtime >= self.swift_script.stat().st_mtime:
			return
		result = subprocess.run(
			[
				"swiftc",
				str(self.swift_script),
				"-o",
				str(self.binary_path),
			],
			check=False,
			capture_output=True,
			text=True,
		)
		if result.returncode != 0:
			raise RuntimeError(f"swiftc failed: {result.stderr.strip() or result.stdout.strip()}")

	def _run_background_removal(self, source_path: Path, destination: Path) -> None:
		result = subprocess.run(
			[
				str(self.binary_path),
				str(source_path),
				str(destination),
			],
			check=False,
			capture_output=True,
			text=True,
		)
		if result.returncode != 0:
			raise RuntimeError(
				f"background removal failed for {source_path.name}: {result.stderr.strip() or result.stdout.strip()}"
			)

	def _absolute_path(self, image_path: str | Path) -> Path:
		path = Path(image_path)
		if path.is_absolute():
			return path
		return self.project_root / path
