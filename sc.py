from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from dot_classifier import load_and_prepare_grayscale_image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BW_DIR = SCRIPT_DIR / "test_images_bw"
DEFAULT_COLOR_DIR = SCRIPT_DIR / "test_images_color"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "dot_cnn.h5"


@dataclass
class SizeClass:
    id: int
    size: float
    color: Tuple[int, int, int]


SIZE_CLASSES: List[SizeClass] = [
    SizeClass(id=1, size=5.0, color=(255, 0, 0)),
    SizeClass(id=2, size=6.0, color=(0, 255, 0)),
    SizeClass(id=3, size=7.2, color=(0, 0, 255)),
    SizeClass(id=4, size=8.64, color=(255, 255, 0)),
]

N_CLASSES = len(SIZE_CLASSES)
PATCH_SIZE = 32


def list_images(folder: str | Path, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[str]:
    paths: list[str] = []
    folder_str = str(folder)
    for extension in extensions:
        paths.extend(glob.glob(os.path.join(folder_str, f"*{extension}")))
    return sorted(paths)


def parse_ground_truth_from_name(filename: str | Path) -> Tuple[int, int, int, int]:
    """
    Expected format:
    spacing_noise_class1_class2_class3_class4.png
    """
    base = os.path.splitext(os.path.basename(str(filename)))[0]
    parts = base.split("_")
    if len(parts) != 6:
        raise ValueError(
            f"Cannot parse ground truth from filename: {filename}. "
            "Expected spacing_noise_class1_class2_class3_class4.png"
        )
    c1, c2, c3, c4 = map(int, parts[-4:])
    return c1, c2, c3, c4


def list_dataset_images(folder: str | Path) -> List[str]:
    """Return only dataset images that match the expected filename pattern."""
    valid_paths: list[str] = []
    for path in list_images(folder):
        try:
            parse_ground_truth_from_name(path)
        except ValueError:
            continue
        valid_paths.append(path)
    return valid_paths


def make_centered_patch(
    image: np.ndarray, center_x: int, center_y: int, patch_size: int = PATCH_SIZE
) -> np.ndarray:
    """Extract a square patch around a dot center, padding with white if needed."""
    height, width = image.shape[:2]
    half = patch_size // 2

    x1 = center_x - half
    y1 = center_y - half
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    patch = np.full((patch_size, patch_size), 255, dtype=image.dtype)

    source_x1 = max(0, x1)
    source_y1 = max(0, y1)
    source_x2 = min(width, x2)
    source_y2 = min(height, y2)

    patch_x1 = source_x1 - x1
    patch_y1 = source_y1 - y1
    patch_x2 = patch_x1 + (source_x2 - source_x1)
    patch_y2 = patch_y1 + (source_y2 - source_y1)

    if source_x2 > source_x1 and source_y2 > source_y1:
        patch[patch_y1:patch_y2, patch_x1:patch_x2] = image[source_y1:source_y2, source_x1:source_x2]

    return patch


def simulate_phone_patch(patch: np.ndarray) -> np.ndarray:
    """Create a photo-like version of a synthetic training patch."""
    augmented = patch.astype(np.float32)

    if np.random.rand() < 0.9:
        sigma = np.random.uniform(0.4, 1.4)
        augmented = cv2.GaussianBlur(augmented, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if np.random.rand() < 0.9:
        alpha = np.random.uniform(0.85, 1.15)
        beta = np.random.uniform(-18.0, 18.0)
        augmented = augmented * alpha + beta

    if np.random.rand() < 0.8:
        noise = np.random.normal(0.0, np.random.uniform(2.0, 7.0), augmented.shape)
        augmented = augmented + noise

    if np.random.rand() < 0.8:
        quality = int(np.random.uniform(35, 70))
        success, encoded = cv2.imencode(
            ".jpg", np.clip(augmented, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if success:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            if decoded is not None:
                augmented = decoded.astype(np.float32)

    return np.clip(augmented, 0, 255).astype(np.uint8)


def build_dataset_from_folders(
    bw_dir: str | Path = DEFAULT_BW_DIR,
    color_dir: str | Path = DEFAULT_COLOR_DIR,
    max_images: int | None = None,
    augment_phone_like: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build training samples from matching color and grayscale images."""
    if not Path(bw_dir).is_dir():
        raise RuntimeError(f"Black-and-white dataset directory does not exist: {bw_dir}")
    if not Path(color_dir).is_dir():
        raise RuntimeError(f"Color dataset directory does not exist: {color_dir}")

    color_paths = list_dataset_images(color_dir)
    if max_images is not None:
        color_paths = color_paths[:max_images]
    if not color_paths:
        raise RuntimeError(f"No dataset images found in color directory: {color_dir}")

    patches: List[np.ndarray] = []
    labels: List[int] = []

    for color_path in color_paths:
        filename = os.path.basename(color_path)
        bw_path = os.path.join(str(bw_dir), filename)
        if not os.path.exists(bw_path):
            print(f"[WARN] Missing BW pair for {filename}, skipping.")
            continue

        color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
        bw_image = cv2.imread(bw_path, cv2.IMREAD_GRAYSCALE)
        if color_image is None or bw_image is None:
            print(f"[WARN] Could not load {filename}, skipping.")
            continue

        color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        for size_class in SIZE_CLASSES:
            target_color = np.array(size_class.color, dtype=np.uint8)
            mask = np.all(color_rgb == target_color, axis=-1).astype(np.uint8) * 255
            num_labels, _, _, centroids = cv2.connectedComponentsWithStats(mask)

            for component_index in range(1, num_labels):
                center_x, center_y = centroids[component_index]
                patch = make_centered_patch(
                    bw_image, int(round(center_x)), int(round(center_y)), PATCH_SIZE
                )
                patches.append(patch)
                labels.append(size_class.id - 1)
                if augment_phone_like:
                    patches.append(simulate_phone_patch(patch))
                    labels.append(size_class.id - 1)

        print(f"[INFO] {filename}: collected samples so far = {len(labels)}")

    if not patches:
        raise RuntimeError("Failed to build a dataset: no samples were collected.")

    x_data = np.stack(patches, axis=0).astype("float32") / 255.0
    x_data = x_data[..., np.newaxis]
    y_data = tf.keras.utils.to_categorical(labels, num_classes=N_CLASSES)

    print(f"[INFO] Dataset ready: X={x_data.shape}, y={y_data.shape}")
    return x_data, y_data


def build_model(input_shape: tuple[int, int, int] = (PATCH_SIZE, PATCH_SIZE, 1)) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    bw_dir: str | Path = DEFAULT_BW_DIR,
    color_dir: str | Path = DEFAULT_COLOR_DIR,
    max_images: int | None = None,
    epochs: int = 10,
    batch_size: int = 128,
    augment_phone_like: bool = True,
) -> None:
    print(f"[INFO] Training on BW data from: {bw_dir}")
    print(f"[INFO] Training on color data from: {color_dir}")
    x_data, y_data = build_dataset_from_folders(
        bw_dir=bw_dir,
        color_dir=color_dir,
        max_images=max_images,
        augment_phone_like=augment_phone_like,
    )

    model = build_model(input_shape=x_data.shape[1:])
    model.fit(
        x_data,
        y_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=True,
    )

    model.save(str(model_path))
    print(f"[INFO] Saved model to {model_path}")


def extract_patches_from_bw(bw_image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Segment dots from a BW image and extract normalized patches."""
    _, binary = cv2.threshold(
        bw_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary)

    candidate_areas = [
        stats[index, cv2.CC_STAT_AREA]
        for index in range(1, num_labels)
        if stats[index, cv2.CC_STAT_AREA] >= 4
    ]
    median_area = float(np.median(candidate_areas)) if candidate_areas else None

    patches: list[np.ndarray] = []
    centers: list[tuple[int, int]] = []

    for component_index in range(1, num_labels):
        area = stats[component_index, cv2.CC_STAT_AREA]
        if area < 4:
            continue
        if median_area is not None:
            if area < max(4.0, 0.35 * median_area):
                continue
            if area > 4.5 * median_area:
                continue

        center_x, center_y = centroids[component_index]
        patch = make_centered_patch(
            bw_image, int(round(center_x)), int(round(center_y)), PATCH_SIZE
        )
        patches.append(patch)
        centers.append((int(round(center_x)), int(round(center_y))))

    if not patches:
        return np.empty((0, PATCH_SIZE, PATCH_SIZE, 1), dtype="float32"), []

    x_data = np.stack(patches, axis=0).astype("float32") / 255.0
    x_data = x_data[..., np.newaxis]
    return x_data, centers


def predict_on_image(model: tf.keras.Model, bw_path: str | Path) -> Dict[int, int]:
    """Return the predicted count for each class on a single BW image."""
    bw_image, _ = load_and_prepare_grayscale_image(str(bw_path))

    patches, _ = extract_patches_from_bw(bw_image)
    if patches.shape[0] == 0:
        print(f"[WARN] No dots detected in {bw_path}")
        return {size_class.id: 0 for size_class in SIZE_CLASSES}

    probabilities = model.predict(patches, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    counts = {size_class.id: 0 for size_class in SIZE_CLASSES}
    for prediction in predictions:
        counts[int(prediction) + 1] += 1
    return counts


def compute_per_class_accuracy(
    predicted: Dict[int, int], ground_truth: tuple[int, int, int, int]
) -> tuple[dict[int, float], float]:
    """Compute per-class accuracy and the mean image accuracy."""
    per_class_accuracy: dict[int, float] = {}
    total_accuracy = 0.0

    for class_id, gt_count in enumerate(ground_truth, start=1):
        pred_count = predicted.get(class_id, 0)
        if gt_count > 0:
            accuracy = 100.0 - (abs(pred_count - gt_count) / gt_count * 100.0)
            accuracy = max(0.0, accuracy)
        else:
            accuracy = 100.0 if pred_count == 0 else 0.0

        per_class_accuracy[class_id] = accuracy
        total_accuracy += accuracy

    return per_class_accuracy, total_accuracy / 4.0


def evaluate_folder(
    model_path: str | Path = DEFAULT_MODEL_PATH, bw_dir: str | Path = DEFAULT_BW_DIR
) -> None:
    if not Path(bw_dir).is_dir():
        raise RuntimeError(f"Black-and-white dataset directory does not exist: {bw_dir}")

    model = tf.keras.models.load_model(str(model_path))
    paths = list_dataset_images(bw_dir)
    if not paths:
        raise RuntimeError(f"No valid dataset images found in {bw_dir}")

    total_ground_truth = np.zeros(N_CLASSES, dtype=np.int64)
    total_predicted = np.zeros(N_CLASSES, dtype=np.int64)
    per_image_accuracies: list[float] = []

    for path in paths:
        ground_truth = np.array(parse_ground_truth_from_name(path))
        predicted_dict = predict_on_image(model, path)
        predicted = np.array(
            [predicted_dict[1], predicted_dict[2], predicted_dict[3], predicted_dict[4]]
        )

        total_ground_truth += ground_truth
        total_predicted += predicted

        image_accuracy = 1.0 - float(np.sum(np.abs(predicted - ground_truth))) / float(
            np.sum(ground_truth)
        )
        image_accuracy = max(0.0, image_accuracy)
        per_image_accuracies.append(image_accuracy)

        print(f"Image: {os.path.basename(path)}")
        print(f"  GT:   {ground_truth.tolist()}")
        print(f"  Pred: {predicted.tolist()}")
        print(f"  Acc:  {image_accuracy * 100:.2f}%")
        print("-" * 40)

    overall_mean_accuracy = float(np.mean(per_image_accuracies)) if per_image_accuracies else 0.0
    print("=== SUMMARY ===")
    print(f"Total dots (GT):   {total_ground_truth.tolist()}")
    print(f"Total dots (pred): {total_predicted.tolist()}")
    print(f"Mean image accuracy: {overall_mean_accuracy * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CNN-based dot classification. Training uses matching image pairs "
            "from test_images_color and test_images_bw."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the CNN model")
    train_parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Where to save the trained model",
    )
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument(
        "--max_images", type=int, default=None, help="Optional limit for training images"
    )
    train_parser.add_argument(
        "--no-phone-augment",
        action="store_true",
        help="Disable photo-like augmentation during training",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict counts for one BW image")
    predict_parser.add_argument(
        "filename",
        type=str,
        help="Dataset filename or an absolute path to a BW image",
    )
    predict_parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained model",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a BW dataset folder")
    eval_parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained model",
    )
    eval_parser.add_argument(
        "--bw_dir",
        type=str,
        default=None,
        help="Optional BW dataset directory",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            model_path=args.model_path,
            bw_dir=DEFAULT_BW_DIR,
            color_dir=DEFAULT_COLOR_DIR,
            max_images=args.max_images,
            epochs=args.epochs,
            batch_size=args.batch_size,
            augment_phone_like=not args.no_phone_augment,
        )
        return

    if args.command == "predict":
        if os.path.isabs(args.filename) or os.path.sep in args.filename:
            bw_path = args.filename
        else:
            bw_path = os.path.join(str(DEFAULT_BW_DIR), args.filename)

        if not os.path.exists(bw_path):
            raise FileNotFoundError(f"BW image not found: {bw_path}")

        model = tf.keras.models.load_model(args.model_path)
        counts = predict_on_image(model, bw_path)
        print(f"Image: {os.path.basename(bw_path)}")
        for size_class in SIZE_CLASSES:
            print(f"  Class {size_class.id} (size={size_class.size}): {counts[size_class.id]}")

        try:
            ground_truth = parse_ground_truth_from_name(bw_path)
            per_class_accuracy, mean_accuracy = compute_per_class_accuracy(
                counts, ground_truth
            )
            print()
            print("Ground-truth comparison:")
            for class_id, gt_count in enumerate(ground_truth, start=1):
                pred_count = counts.get(class_id, 0)
                delta = pred_count - gt_count
                print(
                    f"  Class {class_id}: GT={gt_count:4d}, "
                    f"PRED={pred_count:4d}, DELTA={delta:+4d}, "
                    f"ACC={per_class_accuracy[class_id]:5.1f}%"
                )
            print(f"Mean per-class accuracy: {mean_accuracy:5.2f}%")
        except ValueError:
            print()
            print("No ground-truth metadata found in the filename.")
        return

    if args.command == "eval":
        evaluate_folder(model_path=args.model_path, bw_dir=args.bw_dir or DEFAULT_BW_DIR)


if __name__ == "__main__":
    main()
