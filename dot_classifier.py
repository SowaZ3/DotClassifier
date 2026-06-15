from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_ground_truth_from_path(image_path: str | Path) -> dict[str, object]:
    """
    Parse metadata from a synthetic filename.

    Expected format:
    spacing_noise_class1_class2_class3_class4.png
    """
    stem = Path(image_path).stem
    parts = stem.split("_")
    if len(parts) != 6:
        raise ValueError(
            "Expected filename format: spacing_noise_class1_class2_class3_class4.png"
        )

    spacing_str, noise_str, c1_str, c2_str, c3_str, c4_str = parts
    return {
        "spacing": int(spacing_str),
        "noise": float(noise_str),
        "gt_by_size": {
            1: int(c1_str),
            2: int(c2_str),
            3: int(c3_str),
            4: int(c4_str),
        },
    }


def iter_dataset_images(images_dir: str | Path) -> list[Path]:
    """Return only valid dataset images and skip helper previews."""
    valid_images: list[Path] = []
    for path in sorted(Path(images_dir).glob("*.png")):
        try:
            parse_ground_truth_from_path(path)
        except ValueError:
            continue
        valid_images.append(path)
    return valid_images


def load_and_prepare_grayscale_image(image_path: str) -> tuple[np.ndarray, bool]:
    """
    Load an image and normalize it for analysis.

    Returns:
        prepared_gray: grayscale image in the analysis coordinate system
        photo_mode: True when the image looked like a phone/photo capture
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    photo_mode = max(image.shape[:2]) > 1000
    if not photo_mode:
        return image, False

    cropped = extract_dot_region_from_photo(image)
    upscaled = cv2.resize(cropped, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)
    return enhanced, True


def extract_dot_region_from_photo(gray_image: np.ndarray) -> np.ndarray:
    """Locate and crop the most likely dot field from a phone photo."""
    blackhat = cv2.morphologyEx(
        gray_image,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    _, thresholded = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded)

    small_components = np.zeros_like(thresholded)
    dot_points: list[tuple[float, float]] = []
    for index in range(1, num_labels):
        area = stats[index, cv2.CC_STAT_AREA]
        width = stats[index, cv2.CC_STAT_WIDTH]
        height = stats[index, cv2.CC_STAT_HEIGHT]
        if 5 <= area <= 120 and width <= 20 and height <= 20:
            small_components[labels == index] = 255
            center_x, center_y = centroids[index]
            dot_points.append((float(center_x), float(center_y)))

    if len(dot_points) < 32:
        return gray_image

    dilated = cv2.dilate(
        small_components,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        iterations=2,
    )
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray_image

    image_center_x = gray_image.shape[1] / 2.0
    image_center_y = gray_image.shape[0] / 2.0
    dot_points_array = np.array(dot_points, dtype=np.float32)
    best_crop_bounds: tuple[int, int, int, int] | None = None
    best_score = float("-inf")

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50_000 or area > 350_000:
            continue

        (center_x, center_y), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        short_side = max(1.0, min(rect_w, rect_h))
        long_side = max(rect_w, rect_h)
        aspect_ratio = long_side / short_side
        if aspect_ratio > 1.25:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        inside_mask = (
            (dot_points_array[:, 0] >= x)
            & (dot_points_array[:, 0] <= x + width)
            & (dot_points_array[:, 1] >= y)
            & (dot_points_array[:, 1] <= y + height)
        )
        inside_points = dot_points_array[inside_mask]
        if inside_points.shape[0] < 100:
            continue

        inside_x_min = int(np.floor(inside_points[:, 0].min()))
        inside_y_min = int(np.floor(inside_points[:, 1].min()))
        inside_x_max = int(np.ceil(inside_points[:, 0].max()))
        inside_y_max = int(np.ceil(inside_points[:, 1].max()))
        crop_width = max(1, inside_x_max - inside_x_min)
        crop_height = max(1, inside_y_max - inside_y_min)
        crop_aspect = max(crop_width, crop_height) / max(1.0, min(crop_width, crop_height))
        if crop_aspect > 1.15:
            continue

        score = (
            area
            + inside_points.shape[0] * 120.0
            - abs(crop_aspect - 1.0) * 450_000.0
            - abs(center_x - image_center_x) * 25.0
            - abs(center_y - image_center_y) * 40.0
        )
        if score > best_score:
            best_score = score
            best_crop_bounds = (inside_x_min, inside_y_min, inside_x_max, inside_y_max)

    if best_crop_bounds is None:
        return gray_image

    x1, y1, x2, y2 = best_crop_bounds
    padding = 28
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(gray_image.shape[1], x2 + padding)
    y2 = min(gray_image.shape[0], y2 + padding)
    return gray_image[y1:y2, x1:x2]


class DotClassifier:
    """
    Detect and classify four dot sizes from a grayscale image.

    The size progression uses +20% between classes:
    - Class 1 (base): 5.0 px
    - Class 2: 6.0 px
    - Class 3: 7.2 px
    - Class 4: 8.64 px
    """

    def __init__(self, base_size: float = 5.0) -> None:
        self.base_size = float(base_size)

        # Tuned preprocessing and contour filtering parameters.
        self.adaptive_block = 11
        self.adaptive_c = 3.17058136637993
        self.morph_kernel_size = 5
        self.morph_iterations = 1
        self.area_min = 8.0
        self.area_max = 1923.8263354175413
        self.circularity_min = 0.19394272743200627

        self.size_classes = self._generate_size_classes()
        sizes = [size_class["size"] for size_class in self.size_classes]
        self.b12 = (sizes[0] + sizes[1]) / 2.0
        self.b23 = (sizes[1] + sizes[2]) / 2.0
        self.b34 = (sizes[2] + sizes[3]) / 2.0

    def _generate_size_classes(self) -> list[dict[str, object]]:
        sizes = [
            self.base_size,
            self.base_size * 1.2,
            self.base_size * 1.2 * 1.2,
            self.base_size * 1.2 * 1.2 * 1.2,
        ]
        return [
            {
                "id": class_id,
                "size": size,
                "range": (round(size * 0.9, 2), round(size * 1.1, 2)),
            }
            for class_id, size in enumerate(sizes, start=1)
        ]

    def _cluster_diameters(
        self, diameters: list[float], n_clusters: int = 4, n_iter: int = 20
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Run simple 1D k-means on equivalent diameters."""
        if len(diameters) < n_clusters:
            return None, None

        values = np.array(diameters, dtype=np.float32)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            filtered_values = values[(values >= lower) & (values <= upper)]
            if filtered_values.shape[0] >= n_clusters:
                values_for_centers = filtered_values
            else:
                values_for_centers = values
        else:
            values_for_centers = values

        centers = np.array(
            [size_class["size"] for size_class in self.size_classes], dtype=np.float32
        )
        labels = np.zeros(values_for_centers.shape[0], dtype=np.int32)

        for _ in range(n_iter):
            distances = np.abs(values_for_centers[:, None] - centers[None, :])
            labels = distances.argmin(axis=1)
            for cluster_index in range(n_clusters):
                mask = labels == cluster_index
                if np.any(mask):
                    centers[cluster_index] = values_for_centers[mask].mean()

        order = np.argsort(centers)
        centers_sorted = centers[order]
        old_to_class = {old_index: new_index + 1 for new_index, old_index in enumerate(order)}
        full_distances = np.abs(values[:, None] - centers[None, :])
        full_labels = full_distances.argmin(axis=1)
        labels_mapped = np.array([old_to_class[label] for label in full_labels], dtype=np.int32)
        return centers_sorted, labels_mapped

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Apply blur, adaptive thresholding, and morphology."""
        prepared_image, photo_mode = load_and_prepare_grayscale_image(image_path)

        if photo_mode:
            blurred = cv2.GaussianBlur(prepared_image, (5, 5), 0)
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            return binary

        blurred = cv2.GaussianBlur(prepared_image, (5, 5), 1)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            int(self.adaptive_block),
            self.adaptive_c,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(self.morph_kernel_size), int(self.morph_kernel_size))
        )
        return cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, iterations=int(self.morph_iterations)
        )

    def detect_dots(self, binary_image: np.ndarray) -> list[dict[str, float | tuple[float, float]]]:
        """Detect contours that look like dots."""
        dots: list[dict[str, float | tuple[float, float]]] = []
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) >= self.area_min]
        median_area = float(np.median(areas)) if areas else None
        blob_factor = 3.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.area_min:
                continue

            if median_area is not None and area > blob_factor * median_area:
                split_dots = self._split_large_blob(binary_image, contour)
                if split_dots:
                    dots.extend(split_dots)
                    continue

            if area > self.area_max:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-6)
            if circularity < self.circularity_min:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            equivalent_radius = float(np.sqrt(area / np.pi))
            dots.append(
                {
                    "center": (float(center_x), float(center_y)),
                    "radius": equivalent_radius,
                    "area": float(area),
                }
            )

        return dots

    def _split_large_blob(
        self, binary_image: np.ndarray, contour: np.ndarray
    ) -> list[dict[str, float | tuple[float, float]]] | None:
        """Split a merged blob into multiple synthetic dot detections."""
        mask = np.zeros_like(binary_image, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.erode(mask, kernel, iterations=1)

        distance = cv2.distanceTransform(mask_eroded, cv2.DIST_L2, 5)
        max_value = distance.max()
        if max_value < 2.0:
            return None

        peak_mask = (distance > 0.5 * max_value).astype(np.uint8)
        peak_mask = cv2.morphologyEx(
            peak_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )

        num_labels, _, _, centroids = cv2.connectedComponentsWithStats(peak_mask)
        if num_labels <= 1:
            return None

        dots: list[dict[str, float | tuple[float, float]]] = []
        for index in range(1, num_labels):
            center_x, center_y = centroids[index]
            y = int(round(center_y))
            x = int(round(center_x))
            if y < 0 or y >= distance.shape[0] or x < 0 or x >= distance.shape[1]:
                continue

            radius = float(distance[y, x])
            if radius <= 0.5:
                continue

            dots.append(
                {
                    "center": (float(center_x), float(center_y)),
                    "radius": radius,
                    "area": float(np.pi * radius * radius),
                }
            )

        return dots if len(dots) > 1 else None

    def classify_dot(self, dot_features: dict[str, object]) -> int:
        """Fallback rule-based classification when clustering is not available."""
        radius = float(dot_features.get("radius", 0.0))
        if radius <= 0:
            return 1

        diameter = 2.0 * radius
        if diameter < self.b12:
            return 1
        if diameter < self.b23:
            return 2
        if diameter < self.b34:
            return 3
        return 4

    def analyze_image(self, image_path: str) -> dict[str, object]:
        """Run the full analysis pipeline for a single image."""
        prepared_image, photo_mode = load_and_prepare_grayscale_image(image_path)
        processed = self.preprocess_image(image_path)
        dots = self.detect_dots(processed)

        if not dots:
            return {
                "total_dots": 0,
                "by_size": {class_id: 0 for class_id in range(1, 5)},
                "classifications": [],
                "cluster_centers": None,
                "prepared_image": prepared_image,
                "photo_mode": photo_mode,
            }

        diameters = [2.0 * float(dot["radius"]) for dot in dots]
        centers, labels = self._cluster_diameters(diameters)

        classifications = []
        by_size = {class_id: 0 for class_id in range(1, 5)}

        if centers is not None and labels is not None:
            for dot, size_class in zip(dots, labels):
                class_id = int(size_class)
                classifications.append({"dot": dot, "class": class_id})
                by_size[class_id] += 1
        else:
            for dot in dots:
                class_id = self.classify_dot(dot)
                classifications.append({"dot": dot, "class": class_id})
                by_size[class_id] += 1

        return {
            "total_dots": len(dots),
            "by_size": by_size,
            "classifications": classifications,
            "cluster_centers": centers.tolist() if centers is not None else None,
            "prepared_image": prepared_image,
            "photo_mode": photo_mode,
        }


def compute_accuracy(
    predicted_by_size: dict[int, int], ground_truth_by_size: dict[int, int]
) -> tuple[dict[int, float], float]:
    """Compute per-class accuracy and a simple mean score."""
    per_class_accuracy: dict[int, float] = {}
    total_accuracy = 0.0

    for class_id in range(1, 5):
        ground_truth = ground_truth_by_size.get(class_id, 0)
        predicted = predicted_by_size.get(class_id, 0)

        if ground_truth > 0:
            accuracy = 100.0 - (abs(ground_truth - predicted) / ground_truth * 100.0)
            accuracy = max(0.0, accuracy)
        else:
            accuracy = 100.0 if predicted == 0 else 0.0

        per_class_accuracy[class_id] = accuracy
        total_accuracy += accuracy

    return per_class_accuracy, total_accuracy / 4.0


def save_results_json(
    image_path: str,
    results: dict[str, object],
    output_path: str,
    classifier: DotClassifier,
) -> None:
    """Save analysis output as JSON."""
    total_dots = int(results["total_dots"])
    by_size = results["by_size"]
    class_summary = {}

    for class_id in range(1, 5):
        count = int(by_size.get(class_id, 0))
        percent = (count / total_dots * 100.0) if total_dots > 0 else 0.0
        size_info = classifier.size_classes[class_id - 1]
        class_summary[str(class_id)] = {
            "size_px": size_info["size"],
            "count": count,
            "percent": percent,
        }

    payload: dict[str, object] = {
        "image": str(image_path),
        "total_dots": total_dots,
        "class_summary": class_summary,
    }

    try:
        parsed = parse_ground_truth_from_path(image_path)
        ground_truth_by_size = parsed["gt_by_size"]
        per_class_accuracy, mean_accuracy = compute_accuracy(by_size, ground_truth_by_size)
        payload["gt_stats"] = {
            "spacing": parsed["spacing"],
            "noise": parsed["noise"],
            "gt_by_size": {str(key): int(value) for key, value in ground_truth_by_size.items()},
            "per_class_stats": {
                str(class_id): {
                    "gt": int(ground_truth_by_size[class_id]),
                    "pred": int(by_size.get(class_id, 0)),
                    "delta": int(by_size.get(class_id, 0) - ground_truth_by_size[class_id]),
                    "acc": per_class_accuracy[class_id],
                }
                for class_id in range(1, 5)
            },
            "mean_accuracy": mean_accuracy,
        }
    except ValueError:
        pass

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)


def save_annotated_preview(
    image_path: str, results: dict[str, object], output_path: str
) -> None:
    """Draw colored circles for the predicted class of each dot."""
    prepared_image = results.get("prepared_image")
    if isinstance(prepared_image, np.ndarray):
        image_gray = prepared_image.copy()
    else:
        image_gray, _ = load_and_prepare_grayscale_image(image_path)

    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    class_colors = {
        1: (0, 0, 255),
        2: (0, 255, 0),
        3: (255, 0, 0),
        4: (0, 255, 255),
    }

    for item in results.get("classifications", []):
        dot = item.get("dot", {})
        class_id = int(item.get("class", 1))
        center_x, center_y = dot.get("center", (None, None))
        radius = float(dot.get("radius", 0.0))
        if center_x is None or center_y is None or radius <= 0:
            continue

        center = (int(round(center_x)), int(round(center_y)))
        radius_int = max(1, int(round(radius)))
        cv2.circle(image_color, center, radius_int, class_colors.get(class_id, (255, 255, 255)), 2)

    cv2.imwrite(output_path, image_color)


def print_single_image_report(
    image_path: str, results: dict[str, object], classifier: DotClassifier
) -> None:
    """Print a readable report for one image."""
    print("=" * 60)
    print("DOT CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Image: {Path(image_path).name}")
    print(f"Detected dots: {results['total_dots']}")
    print()
    print("Class distribution:")

    for class_id in range(1, 5):
        count = results["by_size"][class_id]
        percentage = (count / results["total_dots"] * 100.0) if results["total_dots"] else 0.0
        size_info = classifier.size_classes[class_id - 1]
        print(
            f"  Class {class_id} ({size_info['size']:.2f}px): "
            f"{count:4d} dots ({percentage:5.1f}%)"
        )

    try:
        parsed = parse_ground_truth_from_path(image_path)
    except ValueError:
        print()
        print("No ground-truth metadata found in the filename.")
        return

    per_class_accuracy, mean_accuracy = compute_accuracy(results["by_size"], parsed["gt_by_size"])
    print()
    print("Ground-truth comparison:")
    for class_id in range(1, 5):
        ground_truth = parsed["gt_by_size"][class_id]
        predicted = results["by_size"].get(class_id, 0)
        delta = predicted - ground_truth
        print(
            f"  Class {class_id}: GT={ground_truth:4d}, "
            f"PRED={predicted:4d}, DELTA={delta:+4d}, "
            f"ACC={per_class_accuracy[class_id]:5.1f}%"
        )
    print(f"Mean per-class accuracy: {mean_accuracy:5.2f}%")


def evaluate_directory(images_dir: str, base_size: float = 5.0) -> tuple[float, dict[int, float]]:
    """Evaluate the classifier on a directory of valid dataset images."""
    dataset_images = iter_dataset_images(images_dir)
    if not dataset_images:
        raise RuntimeError(f"No valid dataset images found in {images_dir}.")

    classifier = DotClassifier(base_size=base_size)
    image_accuracies: list[float] = []
    per_class_accuracies = {class_id: [] for class_id in range(1, 5)}

    print("=" * 60)
    print(f"DATASET EVALUATION: {images_dir}")
    print("=" * 60)
    print()

    for image_path in dataset_images:
        parsed = parse_ground_truth_from_path(image_path)
        results = classifier.analyze_image(str(image_path))
        predicted_by_size = results["by_size"]
        ground_truth_by_size = parsed["gt_by_size"]
        per_class_accuracy, image_accuracy = compute_accuracy(
            predicted_by_size, ground_truth_by_size
        )

        for class_id in range(1, 5):
            per_class_accuracies[class_id].append(per_class_accuracy[class_id])
        image_accuracies.append(image_accuracy)

        print("-" * 60)
        print(image_path.name)
        print(f"  GT   : {ground_truth_by_size} (sum={sum(ground_truth_by_size.values())})")
        print(f"  PRED : {predicted_by_size} (sum={results['total_dots']})")
        acc_str = ", ".join(
            f"{class_id}={per_class_accuracy[class_id]:5.2f}%"
            for class_id in range(1, 5)
        )
        print(f"  ACC  : {acc_str} -> mean={image_accuracy:5.2f}%")

    dataset_accuracy = sum(image_accuracies) / len(image_accuracies)
    per_class_mean = {
        class_id: sum(values) / len(values) if values else 0.0
        for class_id, values in per_class_accuracies.items()
    }

    print()
    print("=" * 60)
    print("GLOBAL SUMMARY")
    print("=" * 60)
    print(f"Processed images: {len(dataset_images)}")
    print(f"Mean image accuracy: {dataset_accuracy:5.2f}%")
    print("Mean per-class accuracy:")
    for class_id in range(1, 5):
        print(f"  Class {class_id}: {per_class_mean[class_id]:5.2f}%")
    print()

    return dataset_accuracy, per_class_mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classical OpenCV-based dot detection and classification."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single image")
    analyze_parser.add_argument("image", help="Path to the image to analyze")
    analyze_parser.add_argument("--base-size", type=float, default=5.0)
    analyze_parser.add_argument("--save-json", dest="save_json", help="Optional JSON output path")
    analyze_parser.add_argument(
        "--save-preview", dest="save_preview", help="Optional annotated preview image path"
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a dataset directory")
    eval_parser.add_argument("images_dir", help="Directory with dataset PNG images")
    eval_parser.add_argument("--base-size", type=float, default=5.0)

    args = parser.parse_args()

    if args.command == "analyze":
        classifier = DotClassifier(base_size=args.base_size)
        results = classifier.analyze_image(args.image)
        print_single_image_report(args.image, results, classifier)

        if args.save_json:
            save_results_json(args.image, results, args.save_json, classifier)
            print(f"Saved JSON results to {args.save_json}")

        if args.save_preview:
            save_annotated_preview(args.image, results, args.save_preview)
            print(f"Saved annotated preview to {args.save_preview}")
        return

    if args.command == "eval":
        evaluate_directory(args.images_dir, base_size=args.base_size)


if __name__ == "__main__":
    main()
