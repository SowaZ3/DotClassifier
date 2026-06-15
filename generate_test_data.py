from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
BW_DIR = SCRIPT_DIR / "test_images_bw"
COLOR_DIR = SCRIPT_DIR / "test_images_color"


class DotGenerator:
    """Generate synthetic dot patterns with four size classes."""

    def __init__(self, base_size: float = 5.0) -> None:
        self.base_size = float(base_size)
        self.size_classes = self._build_size_classes(self.base_size)

    @staticmethod
    def _build_size_classes(base_size: float) -> list[dict[str, object]]:
        dot_sizes = [
            base_size,
            base_size * 1.2,
            base_size * 1.2 * 1.2,
            base_size * 1.2 * 1.2 * 1.2,
        ]
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        ]
        return [
            {"id": class_id, "size": size, "color": color}
            for class_id, (size, color) in enumerate(zip(dot_sizes, colors), start=1)
        ]

    def generate_pattern(
        self,
        width: int = 800,
        height: int = 800,
        spacing: int = 15,
        distribution: str = "uniform",
        noise_level: float = 0.1,
    ) -> tuple[Image.Image, Image.Image, dict[int, int]]:
        """Generate a synthetic dot image and its color ground truth."""
        if width <= 0 or height <= 0:
            raise ValueError("Image width and height must be positive.")
        if spacing <= 0:
            raise ValueError("Spacing must be positive.")
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("Noise level must be between 0.0 and 1.0.")
        if distribution not in {"uniform", "random", "gradient", "mixed"}:
            raise ValueError(
                "Unsupported distribution. Use one of: uniform, random, gradient, mixed."
            )

        image_bw = Image.new("L", (width, height), 255)
        image_color = Image.new("RGB", (width, height), (255, 255, 255))

        draw_bw = ImageDraw.Draw(image_bw)
        draw_color = ImageDraw.Draw(image_color)

        stats = {1: 0, 2: 0, 3: 0, 4: 0}
        cols = width // spacing
        rows = height // spacing

        for row in range(rows):
            for col in range(cols):
                x = col * spacing + spacing // 2
                y = row * spacing + spacing // 2

                if noise_level > 0:
                    x += random.uniform(-spacing * noise_level, spacing * noise_level)
                    y += random.uniform(-spacing * noise_level, spacing * noise_level)

                if distribution == "uniform":
                    size_class = random.choice(self.size_classes)
                elif distribution == "random":
                    size_class = random.choices(self.size_classes, weights=[0.25] * 4)[0]
                elif distribution == "gradient":
                    progress = row / max(rows, 1)
                    if progress < 0.25:
                        size_class = self.size_classes[0]
                    elif progress < 0.5:
                        size_class = self.size_classes[1]
                    elif progress < 0.75:
                        size_class = self.size_classes[2]
                    else:
                        size_class = self.size_classes[3]
                else:
                    size_class = random.choices(
                        self.size_classes, weights=[0.15, 0.35, 0.35, 0.15]
                    )[0]

                radius = float(size_class["size"]) / 2.0
                draw_bw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)
                draw_color.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=size_class["color"],
                )
                stats[int(size_class["id"])] += 1

        return image_bw, image_color, stats

    def generate_from_image(
        self, image_path: str, base_size: float = 5.0, spacing: int = 10
    ) -> tuple[Image.Image, Image.Image, dict[int, int]]:
        """Generate a dot approximation from a grayscale source image."""
        if spacing <= 0:
            raise ValueError("Spacing must be positive.")

        source = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if source is None:
            raise ValueError(f"Cannot load image: {image_path}")

        max_dim = 800
        height, width = source.shape
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            source = cv2.resize(source, (int(width * scale), int(height * scale)))

        height, width = source.shape
        size_classes = self._build_size_classes(float(base_size))

        output_bw = Image.new("L", (width, height), 255)
        output_color = Image.new("RGB", (width, height), (255, 255, 255))
        draw_bw = ImageDraw.Draw(output_bw)
        draw_color = ImageDraw.Draw(output_color)

        stats = {1: 0, 2: 0, 3: 0, 4: 0}
        cols = width // spacing
        rows = height // spacing

        for row in range(rows):
            for col in range(cols):
                x = col * spacing + spacing // 2
                y = row * spacing + spacing // 2
                if x >= width or y >= height:
                    continue

                brightness = source[y, x]
                darkness = 1.0 - (brightness / 255.0)

                if darkness < 0.2:
                    size_class = size_classes[0]
                elif darkness < 0.4:
                    size_class = size_classes[1]
                elif darkness < 0.6:
                    size_class = size_classes[2]
                else:
                    size_class = size_classes[3]

                radius = (float(size_class["size"]) / 2.0) * (0.5 + darkness * 0.5)
                if radius <= 1.0:
                    continue

                draw_bw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)
                draw_color.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=size_class["color"],
                )
                stats[int(size_class["id"])] += 1

        return output_bw, output_color, stats


def save_image(image: Image.Image, output_path: str, dpi: tuple[int, int] = (300, 300)) -> None:
    """Save a PIL image to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, dpi=dpi)


def create_test_dataset() -> None:
    """Generate a synthetic dataset for repeatable evaluation."""
    for folder in (BW_DIR, COLOR_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    generator = DotGenerator()

    print("=" * 60)
    print("  BATCH GENERATION - DATASET CREATION")
    print("=" * 60)
    print()

    spacings = [10, 14, 18, 22, 26, 30]
    noises = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    repeats_per_config = 3

    total_images = len(spacings) * len(noises) * repeats_per_config
    current = 0

    for spacing in spacings:
        for noise in noises:
            for repeat_index in range(1, repeats_per_config + 1):
                current += 1
                print(
                    f"[{current}/{total_images}] Generating "
                    f"spacing={spacing}, noise={noise}, repeat={repeat_index}"
                )

                image_bw, image_color, stats = generator.generate_pattern(
                    width=800,
                    height=800,
                    spacing=spacing,
                    distribution="random",
                    noise_level=noise,
                )

                filename = (
                    f"{spacing}_{noise}_{stats.get(1, 0)}_{stats.get(2, 0)}_"
                    f"{stats.get(3, 0)}_{stats.get(4, 0)}.png"
                )

                save_image(image_bw, str(BW_DIR / filename))
                save_image(image_color, str(COLOR_DIR / filename))

    print()
    print("DONE")
    print(f"Generated {total_images} image pairs in {BW_DIR.name} and {COLOR_DIR.name}.")


if __name__ == "__main__":
    create_test_dataset()
