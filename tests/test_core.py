from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image

from cnnlab.core import ConfigurableCNN, make_splits, ResizeNoUpscalePad, binary_metrics, scan_dataset
import torch


def test_make_splits_is_reproducible_and_70_15_15():
    df = pd.DataFrame({
        "path": [f"/tmp/{i}.jpg" for i in range(200)],
        "label": [0] * 100 + [1] * 100,
        "class_name": ["Cat"] * 100 + ["Dog"] * 100,
    })
    a = make_splits(df, seed=42).sort_values("path").reset_index(drop=True)
    b = make_splits(df, seed=42).sort_values("path").reset_index(drop=True)
    assert a["split"].tolist() == b["split"].tolist()
    sizes = a.groupby("split").size().to_dict()
    assert sizes == {"test": 30, "train": 140, "validation": 30}


def test_no_upscale_padding():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "small.png"
        Image.new("RGB", (12, 8), "white").save(p)
        with Image.open(p) as im:
            out = ResizeNoUpscalePad(32)(im)
        assert out.size == (32, 32)
        # The original content remains 12 x 8 (centered) rather than being enlarged.
        assert out.getpixel((10, 12)) == (255, 255, 255)
        assert out.getpixel((0, 0)) == (0, 0, 0)


def test_configurable_cnn_binary_output_shape():
    model = ConfigurableCNN(depth=2, base_filters=8, kernel_size=3)
    y = model(torch.randn(2, 3, 64, 64))
    assert tuple(y.shape) == (2, 1)


def test_binary_metrics_return_confusion_matrix_counts():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.6, 0.9])
    assert m["tn"] == 1 and m["fp"] == 1 and m["fn"] == 0 and m["tp"] == 2
    assert 0 <= m["accuracy"] <= 1


def test_scan_removes_identical_images_with_conflicting_labels():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cat_dir = root / "Cat"; dog_dir = root / "Dog"
        cat_dir.mkdir(); dog_dir.mkdir()
        duplicate = Image.new("RGB", (16, 16), (128, 128, 128))
        duplicate.save(cat_dir / "same_cat.png")
        duplicate.save(dog_dir / "same_dog.png")
        Image.new("RGB", (16, 16), (255, 0, 0)).save(cat_dir / "cat_only.png")
        Image.new("RGB", (16, 16), (0, 0, 255)).save(dog_dir / "dog_only.png")
        manifest, report = scan_dataset(root)
        assert len(manifest) == 2
        assert set(manifest["label"].tolist()) == {0, 1}
        assert report["cross_class_conflicts_removed"] == 1
