from __future__ import annotations

import hashlib
import json
import random
import time
import platform
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = False
SEED = 42
MICROSOFT_DATA_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_TO_LABEL = {"cat": 0, "cats": 0, "dog": 1, "dogs": 1}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can fail for some operators; seeded RNGs are the main reproducibility control.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    p = project_root() / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir() -> Path:
    p = data_root() / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def experiments_dir() -> Path:
    p = project_root() / "experiments"
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_download_microsoft_dataset(
    destination: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """Download and safely extract Microsoft's Cats-vs-Dogs archive."""
    raw_dir = Path(destination).expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "kagglecatsanddogs_5340.zip"
    extracted_hint = raw_dir / "PetImages"
    if extracted_hint.exists() and any(extracted_hint.iterdir()):
        if progress_callback:
            progress_callback(1.0, "Dataset đã được giải nén sẵn.")
        return raw_dir

    if not archive.exists():
        if progress_callback:
            progress_callback(0.0, "Đang tải archive từ Microsoft (khoảng 786 MB)...")
        request = urllib.request.Request(
            MICROSOFT_DATA_URL,
            headers={"User-Agent": "CNN-Research-Lab/1.0"},
        )
        partial = archive.with_suffix(archive.suffix + ".part")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                total = int(response.headers.get("Content-Length", "0") or 0)
                received = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    received += len(chunk)
                    if progress_callback and total:
                        progress_callback(min(received / total, 0.99), f"Đã tải {received / (1024**2):.0f} MB")
            partial.replace(archive)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    else:
        if progress_callback:
            progress_callback(0.5, "Đã có archive; bỏ qua bước tải.")

    if progress_callback:
        progress_callback(0.99, "Đang giải nén dữ liệu...")
    with zipfile.ZipFile(archive) as zf:
        root = raw_dir.resolve()
        for member in zf.infolist():
            target = (raw_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Tệp trong archive có đường dẫn không an toàn: {member.filename}")
        zf.extractall(raw_dir)
    if progress_callback:
        progress_callback(1.0, "Tải và giải nén hoàn tất.")
    return raw_dir


def _class_dirs_for_candidate(folder: Path) -> Optional[dict[str, Path]]:
    try:
        children = [p for p in folder.iterdir() if p.is_dir()]
    except (OSError, PermissionError):
        return None
    by_lower = {p.name.lower(): p for p in children}
    cat = next((by_lower[n] for n in ("cat", "cats") if n in by_lower), None)
    dog = next((by_lower[n] for n in ("dog", "dogs") if n in by_lower), None)
    if cat is not None and dog is not None:
        return {"cat": cat, "dog": dog}
    return None


def resolve_class_dirs(root: str | Path) -> dict[str, Path]:
    """Accept either the PetImages directory or its parent (searching 2 directory levels)."""
    root = Path(root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {root}")
    direct = _class_dirs_for_candidate(root)
    if direct:
        return direct
    candidates: list[Path] = []
    try:
        first = [p for p in root.iterdir() if p.is_dir()]
    except PermissionError:
        first = []
    for p in first:
        candidates.append(p)
        try:
            candidates.extend(q for q in p.iterdir() if q.is_dir())
        except PermissionError:
            continue
    for candidate in candidates:
        found = _class_dirs_for_candidate(candidate)
        if found:
            return found
    raise FileNotFoundError(
        "Không tìm thấy hai thư mục lớp Cat và Dog. Hãy đặt dữ liệu theo dạng "
        "data/raw/PetImages/Cat/* và data/raw/PetImages/Dog/*, hoặc chỉ định thư mục chứa Cat/Dog."
    )


def _image_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scan_dataset(root: str | Path) -> tuple[pd.DataFrame, dict]:
    """Validate images, collect a manifest, and remove exact duplicate files by SHA-256."""
    class_dirs = resolve_class_dirs(root)
    rows: list[dict] = []
    invalid: list[dict] = []
    seen_hashes: dict[str, int] = {}
    conflicting_hashes: set[str] = set()
    conflicting_row_indices: set[int] = set()
    duplicates = 0
    cross_class_conflicts = 0
    for class_name, label in (("cat", 0), ("dog", 1)):
        base = class_dirs[class_name]
        files = sorted(p for p in base.rglob("*") if p.is_file())
        for path in files:
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    width, height = im.size
                    if width < 1 or height < 1:
                        raise ValueError("Kích thước ảnh không hợp lệ")
                digest = _image_hash(path)
                if digest in conflicting_hashes:
                    duplicates += 1
                    continue
                if digest in seen_hashes:
                    duplicates += 1
                    previous_index = seen_hashes[digest]
                    if rows[previous_index]["label"] != label:
                        # Identical image content with conflicting labels: remove both labels.
                        conflicting_hashes.add(digest)
                        conflicting_row_indices.add(previous_index)
                        cross_class_conflicts += 1
                    continue
                seen_hashes[digest] = len(rows)
                rows.append({
                    "path": str(path.resolve()),
                    "label": label,
                    "class_name": "Dog" if label == 1 else "Cat",
                    "width": int(width),
                    "height": int(height),
                    "min_side": int(min(width, height)),
                    "max_side": int(max(width, height)),
                    "aspect_ratio": float(width / height),
                    "bytes": int(path.stat().st_size),
                    "sha256": digest,
                })
            except Exception as exc:
                invalid.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    df = pd.DataFrame([row for index, row in enumerate(rows) if index not in conflicting_row_indices])
    if df.empty:
        raise ValueError("Không tìm thấy ảnh hợp lệ trong hai thư mục Cat/Dog.")
    summary = {
        "root": str(Path(root).expanduser().resolve()),
        "class_dirs": {k: str(v) for k, v in class_dirs.items()},
        "valid_images_after_dedup": int(len(df)),
        "valid_cat": int((df.label == 0).sum()),
        "valid_dog": int((df.label == 1).sum()),
        "invalid_files": int(len(invalid)),
        "exact_duplicates_removed": int(duplicates),
        "cross_class_conflicts_removed": int(cross_class_conflicts),
        "invalid_examples": invalid[:100],
        "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return df, summary


def make_splits(manifest: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Stratified 70/15/15 split, matching the CS535 research notebook."""
    if len(manifest) < 20:
        raise ValueError("Cần ít nhất 20 ảnh hợp lệ để chia train/validation/test có ý nghĩa.")
    train_df, temp_df = train_test_split(
        manifest, test_size=0.30, stratify=manifest["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "validation"
    test_df = test_df.copy(); test_df["split"] = "test"
    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


class ResizeNoUpscalePad:
    """Resize down only; preserve aspect ratio and pad to a square canvas."""
    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(self.size / width, self.size / height, 1.0)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        if (new_width, new_height) != (width, height):
            image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(image, ((self.size - new_width) // 2, (self.size - new_height) // 2))
        return canvas


def make_transform(size: int = 128, train: bool = False, augment: bool = True):
    ops = [ResizeNoUpscalePad(size)]
    if train and augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class CatsDogsDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        with Image.open(row["path"]) as im:
            image = im.convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image, label, str(row["path"])


class ConfigurableCNN(nn.Module):
    def __init__(self, depth=3, base_filters=32, kernel_size=3, dropout=0.3,
                 batchnorm=True, pool="max", activation="relu"):
        super().__init__()
        if depth not in range(2, 6):
            raise ValueError("depth phải nằm trong khoảng 2–5")
        layers: list[nn.Module] = []
        in_channels = 3
        out_channels = int(base_filters)
        for idx in range(int(depth)):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=int(kernel_size),
                                    stride=1, padding="same", bias=not batchnorm))
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.SiLU() if activation == "silu" else nn.ReLU(inplace=True))
            if pool == "max":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif pool == "avg":
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels = min(out_channels * 2, 256)
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def build_model(config: dict) -> nn.Module:
    name = config.get("model_name", "configurable_cnn")
    pretrained = bool(config.get("pretrained", False))
    if name == "configurable_cnn":
        return ConfigurableCNN(
            depth=int(config.get("depth", 3)),
            base_filters=int(config.get("base_filters", 32)),
            kernel_size=int(config.get("kernel_size", 3)),
            dropout=float(config.get("dropout", 0.3)),
            batchnorm=bool(config.get("batchnorm", True)),
            pool=config.get("pool", "max"),
            activation=config.get("activation", "relu"),
        )
    model_factories = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
    }
    if name not in model_factories:
        raise ValueError(f"Kiến trúc không được hỗ trợ: {name}")
    factory, default_weights = model_factories[name]
    model = factory(weights=default_weights if pretrained else None)
    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name in ("mobilenet_v3_small", "efficientnet_b0", "convnext_tiny"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def binary_metrics(y_true, probability, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    probability = np.asarray(probability).astype(float).reshape(-1)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    try:
        auc = float(roc_auc_score(y_true, probability))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y_true, probability))
    except ValueError:
        ap = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": auc,
        "pr_auc_avg_precision": ap,
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "brier": float(brier_score_loss(y_true, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
    }


def _run_epoch(model, loader, device, optimizer=None, criterion=None, freeze_batchnorm: bool = False):
    is_train = optimizer is not None
    model.train(is_train)
    if is_train and freeze_batchnorm:
        # Keep pretrained backbone BatchNorm running statistics fixed in feature-extraction mode.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    criterion = criterion or nn.BCEWithLogitsLoss()
    total_loss = 0.0
    y_true: list[float] = []
    probs: list[float] = []
    paths: list[str] = []
    for images, labels, batch_paths in loader:
        images = images.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(images).reshape(-1)
            loss = criterion(logits, labels.reshape(-1))
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
        paths.extend(list(batch_paths))
    return total_loss / max(1, len(loader.dataset)), np.asarray(y_true), np.asarray(probs), paths


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    if name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _prediction_frame(paths, y_true, probs) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    pred = (probs >= 0.5).astype(int)
    names = np.where(pred == 1, "Dog", "Cat")
    return pd.DataFrame({
        "path": list(paths), "y_true": y_true,
        "actual": np.where(y_true == 1, "Dog", "Cat"),
        "prob_dog": probs, "prediction": pred, "predicted_class": names,
        "correct": pred == y_true,
        "confidence_in_prediction": np.maximum(probs, 1 - probs),
    })


def train_experiment(split_df: pd.DataFrame, config: dict,
                     progress_callback: Optional[Callable[[dict], None]] = None) -> Path:
    """Train using train/validation only. Internal test is intentionally not touched here."""
    seed_everything(int(config.get("seed", SEED)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + hashlib.sha1(str(time.time_ns()).encode()).hexdigest()[:6]
    run_dir = experiments_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "validation"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Không có train/validation split. Hãy quét và chuẩn bị dataset trước.")
    max_train = int(config.get("max_train_images", 0))
    if max_train > 0 and len(train_df) > max_train:
        # Stratified sampling keeps the quick-development subset class-balanced.
        pieces = []
        for label, group in train_df.groupby("label"):
            n = max(1, round(max_train * len(group) / len(train_df)))
            pieces.append(group.sample(n=min(n, len(group)), random_state=int(config.get("seed", SEED))))
        train_df = pd.concat(pieces).sample(frac=1, random_state=int(config.get("seed", SEED)))

    size = int(config.get("image_size", 128))
    batch_size = int(config.get("batch_size", 32))
    workers = 0  # stable on Windows and avoids multiprocessing surprises in Streamlit
    train_loader = DataLoader(
        CatsDogsDataset(train_df, make_transform(size, train=True, augment=bool(config.get("augmentation", True)))),
        batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CatsDogsDataset(val_df, make_transform(size, train=False, augment=False)),
        batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(config).to(device)
    transfer_strategy = config.get("transfer_strategy", "fine_tuning" if config.get("pretrained", False) else "scratch")
    if config.get("model_name") != "configurable_cnn" and config.get("pretrained", False):
        if transfer_strategy == "feature_extraction":
            for parameter in model.parameters():
                parameter.requires_grad = False
            if config.get("model_name") == "resnet18":
                head = model.fc
            else:
                head = model.classifier[-1]
            for parameter in head.parameters():
                parameter.requires_grad = True
        else:
            # Fine-tuning: all backbone and head parameters are trainable.
            for parameter in model.parameters():
                parameter.requires_grad = True
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("Model không có tham số được phép cập nhật.")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = _make_optimizer(config.get("optimizer", "AdamW"), trainable_parameters,
                                float(config.get("learning_rate", 1e-3)),
                                float(config.get("weight_decay", 1e-4)))
    epochs = int(config.get("epochs", 3))
    history: list[dict] = []
    best_auc = -float("inf")
    best_val_loss = float("inf")
    best_state = None
    start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_y, train_p, _ = _run_epoch(
            model, train_loader, device, optimizer, criterion,
            freeze_batchnorm=(transfer_strategy == "feature_extraction"),
        )
        val_loss, val_y, val_p, val_paths = _run_epoch(model, val_loader, device, None, criterion)
        train_m = binary_metrics(train_y, train_p)
        val_m = binary_metrics(val_y, val_p)
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_m["accuracy"], "val_accuracy": val_m["accuracy"],
            "train_roc_auc": train_m["roc_auc"], "val_roc_auc": val_m["roc_auc"],
            "train_f1": train_m["f1"], "val_f1": val_m["f1"],
            "epoch_seconds": time.time() - t0,
        }
        history.append(row)
        auc = val_m["roc_auc"]
        current_auc = auc if np.isfinite(auc) else -float("inf")
        better = current_auc > best_auc or (current_auc == best_auc and val_loss < best_val_loss)
        if better:
            best_auc = current_auc
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if progress_callback:
            progress_callback({
                "epoch": epoch, "epochs": epochs, "train_loss": train_loss,
                "val_loss": val_loss, "val_accuracy": val_m["accuracy"],
                "val_roc_auc": val_m["roc_auc"], "epoch_seconds": row["epoch_seconds"],
            })

    if best_state is not None:
        model.load_state_dict(best_state)
    # Validation prediction is allowed for error analysis and model development.
    val_loss, val_y, val_p, val_paths = _run_epoch(model, val_loader, device, None, criterion)
    val_metrics = binary_metrics(val_y, val_p)
    val_metrics["loss"] = float(val_loss)
    val_preds = _prediction_frame(val_paths, val_y, val_p)
    val_preds.to_csv(run_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    torch.save({"model_state": model.state_dict(), "config": config}, run_dir / "checkpoint.pt")
    saved_config = dict(config)
    manifest_path = Path(config.get("dataset_manifest", ""))
    split_fingerprint = None
    if manifest_path.is_file():
        split_fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    saved_config.update({
        "run_id": run_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device), "train_images_used": int(len(train_df)),
        "validation_images": int(len(val_df)), "parameters": count_parameters(model),
        "transfer_strategy": transfer_strategy,
        "elapsed_seconds": float(time.time() - start),
        "split_manifest_sha256": split_fingerprint,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "preprocessing": "ResizeNoUpscalePad; RGB; ImageNet mean/std; augmentation only on train",
        "test_policy": "not evaluated during training/model selection",
    })
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(saved_config, f, ensure_ascii=False, indent=2)
    with (run_dir / "validation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(val_metrics, f, ensure_ascii=False, indent=2, allow_nan=True)
    return run_dir


def list_experiments() -> list[dict]:
    rows: list[dict] = []
    for config_path in sorted(experiments_dir().glob("*/config.json"), reverse=True):
        run_dir = config_path.parent
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metrics_path = run_dir / "validation_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            rows.append({**config, **{f"val_{k}": v for k, v in metrics.items()}, "run_dir": str(run_dir)})
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def load_experiment(run_dir: str | Path, device: Optional[str] = None):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)
    # The checkpoint already contains all pretrained weights; do not redownload the original backbone.
    model = build_model({**config, "pretrained": False})
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(target_device)
    model.eval()
    return model, config, target_device


def evaluate_internal_test(run_dir: str | Path, split_csv: str | Path,
                           progress_callback: Optional[Callable[[float], None]] = None) -> tuple[dict, pd.DataFrame]:
    """Explicit final test evaluation; do not use its results to tune/select models."""
    run_dir = Path(run_dir)
    split_df = pd.read_csv(split_csv)
    test_df = split_df[split_df["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError("Tập test trống; hãy tạo lại split trước khi đánh giá.")
    model, config, device = load_experiment(run_dir)
    loader = DataLoader(
        CatsDogsDataset(test_df, make_transform(int(config.get("image_size", 128)), train=False, augment=False)),
        batch_size=int(config.get("batch_size", 32)), shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    _, y, p, paths = _run_epoch(model, loader, device, None, nn.BCEWithLogitsLoss())
    metrics = binary_metrics(y, p)
    preds = _prediction_frame(paths, y, p)
    preds.to_csv(run_dir / "internal_test_predictions.csv", index=False)
    payload = dict(metrics)
    payload.update({"evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "split": "internal_test"})
    (run_dir / "internal_test_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    if progress_callback:
        progress_callback(1.0)
    return payload, preds


def evaluate_challenge_folder(run_dir: str | Path, challenge_root: str | Path) -> tuple[dict, pd.DataFrame]:
    """Evaluate optional folders cat/, dog/, and ood/ without updating model weights."""
    root = Path(challenge_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy challenge folder: {root}")
    model, config, device = load_experiment(run_dir)
    transform = make_transform(int(config.get("image_size", 128)), train=False, augment=False)
    rows = []
    class_folders = {"cat": 0, "dog": 1, "ood": None}
    for child in root.iterdir():
        key = child.name.lower()
        if not child.is_dir() or key not in class_folders:
            continue
        label = class_folders[key]
        for path in sorted(p for p in child.rglob("*") if p.is_file()):
            try:
                with Image.open(path) as im:
                    tensor = transform(im.convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = float(torch.sigmoid(model(tensor).reshape(-1))[0].item())
                rows.append({
                    "path": str(path), "group": key, "y_true": label,
                    "actual": key.title() if label is not None else "OOD",
                    "prob_dog": prob, "prediction": int(prob >= 0.5),
                    "predicted_class": "Dog" if prob >= 0.5 else "Cat",
                    "confidence_in_prediction": max(prob, 1 - prob),
                })
            except Exception as exc:
                rows.append({"path": str(path), "group": key, "y_true": label,
                             "actual": key.title(), "error": f"{type(exc).__name__}: {exc}"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Challenge folder phải có thư mục con cat/, dog/ hoặc ood/ chứa ảnh.")
    for required in ("y_true", "prob_dog", "prediction", "confidence_in_prediction", "group"):
        if required not in frame.columns:
            frame[required] = np.nan
    labelled = frame[frame["y_true"].notna() & frame["prob_dog"].notna()].copy()
    metrics: dict = {"evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_images": int(len(frame))}
    if not labelled.empty and labelled["y_true"].nunique() == 2:
        metrics.update(binary_metrics(labelled["y_true"].astype(int), labelled["prob_dog"].astype(float)))
    ood = frame[(frame["group"] == "ood") & frame["prob_dog"].notna()]
    if not ood.empty:
        ood_conf = np.maximum(ood["prob_dog"].astype(float), 1 - ood["prob_dog"].astype(float))
        metrics.update({
            "ood_images": int(len(ood)),
            "ood_mean_confidence": float(ood_conf.mean()),
            "ood_high_confidence_over_0_9": int((ood_conf > 0.9).sum()),
            "ood_note": "Confidence is not a calibrated OOD detector; binary classifiers still force cat/dog labels.",
        })
    frame.to_csv(Path(run_dir) / "challenge_predictions.csv", index=False)
    (Path(run_dir) / "challenge_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    return metrics, frame
