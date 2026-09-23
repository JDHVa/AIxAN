import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm import tqdm
from PIL import Image

ROOT = Path("dataset_workspace")
RAW_DIR = ROOT / "raw"
ALET_DIR = RAW_DIR / "metu_alet"
RF_TOOLS = RAW_DIR / "mechanical_tools_100"
RF_PARTS = RAW_DIR / "mechanical_parts"
OUTPUT_DIR = ROOT / "merged_dataset"

for split in ("train", "val", "test"):
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# Made with AI to make a better uniform dataset
CLASS_MAP = {
    # Hammers / mallets
    "hammer": "hammer",
    "mallet": "hammer",
    "sledgehammer": "hammer",
    # Screwdrivers
    "screwdriver": "screwdriver",
    "flathead_screwdriver": "screwdriver",
    "phillips_screwdriver": "screwdriver",
    # Wrenches / spanners
    "wrench": "wrench",
    "spanner": "wrench",
    "open_end_wrench": "wrench",
    "box_wrench": "wrench",
    "adjustable_wrench": "wrench",
    "monkey_wrench": "wrench",
    "torque_wrench": "wrench",
    # Pliers
    "pliers": "pliers",
    "needle_nose_pliers": "pliers",
    "cutting_pliers": "pliers",
    "slip_joint_pliers": "pliers",
    # Saws
    "saw": "saw",
    "hacksaw": "saw",
    "handsaw": "saw",
    "jigsaw": "saw",
    # Drills
    "drill": "drill",
    "power_drill": "drill",
    "drill_bit": "drill_bit",
    # Chisels
    "chisel": "chisel",
    "cold_chisel": "chisel",
    # Files / rasps
    "file": "file",
    "rasp": "file",
    # Tape measure
    "tape_measure": "tape_measure",
    "measuring_tape": "tape_measure",
    # Level
    "level": "level",
    "spirit_level": "level",
    # Scissors / snips
    "scissors": "scissors",
    "tin_snips": "scissors",
    # Knife / cutter
    "knife": "knife",
    "utility_knife": "knife",
    "box_cutter": "knife",
    # Clamp
    "clamp": "clamp",
    "c_clamp": "clamp",
    "vise_grip": "clamp",
    # Socket / ratchet
    "socket": "socket",
    "ratchet": "ratchet",
    # Nuts, bolts, fasteners
    "nut": "nut",
    "bolt": "bolt",
    "screw": "screw",
    "washer": "washer",
    "nail": "nail",
    "rivet": "rivet",
    # Mechanical parts
    "bearing": "bearing",
    "gear": "gear",
    "spring": "spring",
    "bushing": "bushing",
    "pulley": "pulley",
    "chain": "chain",
    "belt": "belt",
    "axle": "axle",
    "sandpaper": "sandpaper",
    "grinder": "grinder",
    "soldering_iron": "soldering_iron",
    "multimeter": "multimeter",
    "paint_brush": "paint_brush",
    "roller": "paint_roller",
    "trowel": "trowel",
    "shovel": "shovel",
    "rake": "rake",
    "axe": "axe",
    "hoe": "hoe",
    "fork": "garden_fork",
    "staple_gun": "staple_gun",
    "caulk_gun": "caulk_gun",
    "heat_gun": "heat_gun",
    "flashlight": "flashlight",
    "extension_cord": "extension_cord",
    "tape": "tape",
}


def normalize_class(raw: str) -> str | None:
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_MAP.get(key, key)


all_classes: set[str] = set()


def download_metu_alet():
    print("\n[1/3] Downloading METU-ALET from GitHub...")
    if ALET_DIR.exists() and any(ALET_DIR.iterdir()):
        print("  Already downloaded, skipping.")
        return

    ALET_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/metu-kovan/METU-ALET.git",
            str(ALET_DIR),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR cloning METU-ALET: {result.stderr}")
        sys.exit(1)
    print("  Done.")


def coco_to_yolo(json_path: Path, images_dir: Path, out_labels_dir: Path):
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}
    cat_map = {cat["id"]: normalize_class(cat["name"]) for cat in coco["categories"]}

    from collections import defaultdict

    img_annotations: dict = defaultdict(list)
    for ann in coco["annotations"]:
        img_annotations[ann["image_id"]].append(ann)

    results = {}
    for img_id, img_info in tqdm(id_to_img.items(), desc="  COCO→YOLO"):
        filename = img_info["file_name"]
        img_path = images_dir / filename
        if not img_path.exists():
            continue

        w, h = img_info["width"], img_info["height"]
        lines = []

        for ann in img_annotations[img_id]:
            class_name = cat_map.get(ann["category_id"])
            if class_name is None:
                continue
            all_classes.add(class_name)

            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{class_name} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_file = out_labels_dir / (Path(filename).stem + ".txt")
        label_file.write_text("\n".join(lines))
        results[filename] = label_file

    return results


def process_metu_alet():
    print("\n  Processing METU-ALET annotations...")
    ann_dir = ALET_DIR / "annotations"
    img_dir = ALET_DIR / "images"

    if not ann_dir.exists():
        print(f"  WARNING: {ann_dir} not found — check repo structure.")
        return []

    items = []
    for split_json in ann_dir.glob("*.json"):
        tmp_labels = RAW_DIR / "alet_labels_tmp" / split_json.stem
        mapping = coco_to_yolo(split_json, img_dir, tmp_labels)
        for fname, lpath in mapping.items():
            items.append(
                {
                    "image": img_dir / fname,
                    "label": lpath,
                    "source": "alet",
                }
            )
    return items


def download_roboflow(api_key: str):
    print("\n[2/3] Downloading Roboflow datasets...")
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  Run: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    datasets = []

    print("  Downloading Mechanical tools-100...")
    proj = rf.workspace("ug-student-bmstu-ru").project("mechanical-tools-100")
    ds = proj.version(1).download("yolov8", location=str(RF_TOOLS), overwrite=False)
    datasets.append(("mechanical_tools_100", RF_TOOLS))

    print("  Downloading Mechanical Parts...")
    proj = rf.workspace("mazhar-cakir").project("mechanical-parts")
    ds = proj.version(1).download("yolov8", location=str(RF_PARTS), overwrite=False)
    datasets.append(("mechanical_parts", RF_PARTS))

    return datasets


def collect_roboflow_items(ds_dir: Path, source_tag: str) -> list:
    items = []
    for split in ("train", "valid", "test"):
        img_dir = ds_dir / split / "images"
        label_dir = ds_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*.[jp][pn]g"):
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            items.append(
                {
                    "image": img_path,
                    "label": label_path,
                    "source": source_tag,
                }
            )

    yaml_path = ds_dir / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        raw_classes = meta.get("names", [])
        idx_to_name = {i: normalize_class(n) for i, n in enumerate(raw_classes)}
        for item in items:
            _rewrite_label_names(item["label"], idx_to_name)

    return items


def _rewrite_label_names(label_path: Path, idx_to_name: dict):
    lines = label_path.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            new_lines.append(line)
            continue
        name = idx_to_name.get(idx)
        if name:
            all_classes.add(name)
            new_lines.append(" ".join([name] + parts[1:]))
    label_path.write_text("\n".join(new_lines))


def merge_and_split(all_items: list, train=0.80, val=0.10):
    print(f"\n[3/3] Merging {len(all_items)} items → train/val/test split...")

    sorted_classes = sorted(all_classes)
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}

    random.seed(42)
    random.shuffle(all_items)

    n = len(all_items)
    n_train = int(n * train)
    n_val = int(n * val)

    splits = {
        "train": all_items[:n_train],
        "val": all_items[n_train : n_train + n_val],
        "test": all_items[n_train + n_val :],
    }

    counters = {"train": 0, "val": 0, "test": 0}

    for split, items in splits.items():
        img_out = OUTPUT_DIR / "images" / split
        lbl_out = OUTPUT_DIR / "labels" / split

        for item in tqdm(items, desc=f"  {split:5s}"):
            src_img = item["image"]
            src_lbl = item["label"]

            if not src_img.exists() or not src_lbl.exists():
                continue

            stem = f"{item['source']}_{src_img.stem}"
            dst_img = img_out / (stem + src_img.suffix)
            dst_lbl = lbl_out / (stem + ".txt")

            shutil.copy2(src_img, dst_img)

            raw_lines = src_lbl.read_text().strip().splitlines()
            out_lines = []
            for line in raw_lines:
                parts = line.split()
                if not parts:
                    continue
                class_name = parts[0]
                idx = class_to_idx.get(class_name)
                if idx is None:
                    continue
                out_lines.append(" ".join([str(idx)] + parts[1:]))

            dst_lbl.write_text("\n".join(out_lines))
            counters[split] += 1

    return sorted_classes, counters


def write_yaml(classes: list):
    yaml_path = OUTPUT_DIR / "data.yaml"
    config = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"\n  data.yaml written → {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roboflow-key",
        required=True,
        help="Free Roboflow API key from app.roboflow.com",
    )
    parser.add_argument(
        "--skip-alet",
        action="store_true",
        help="Skip METU-ALET download (e.g. if repo is private)",
    )
    args = parser.parse_args()

    all_items = []

    if not args.skip_alet:
        download_metu_alet()
        alet_items = process_metu_alet()
        print(f"  METU-ALET: {len(alet_items)} items collected.")
        all_items.extend(alet_items)
    else:
        print("\n[1/3] METU-ALET skipped.")

    download_roboflow(args.roboflow_key)

    for tag, ds_dir in [
        ("mechanical_tools", RF_TOOLS),
        ("mechanical_parts", RF_PARTS),
    ]:
        items = collect_roboflow_items(ds_dir, tag)
        print(f"  {tag}: {len(items)} items collected.")
        all_items.extend(items)

    if not all_items:
        print("\nERROR: No items collected. Check paths and API key.")
        sys.exit(1)

    final_classes, counters = merge_and_split(all_items)
    yaml_path = write_yaml(final_classes)

    print("\n" + "=" * 52)
    print("  Dataset ready!")
    print(f"  Classes  : {len(final_classes)}")
    print(f"  Train    : {counters['train']} images")
    print(f"  Val      : {counters['val']} images")
    print(f"  Test     : {counters['test']} images")
    print(f"  Output   : {OUTPUT_DIR.resolve()}")
    print(f"  YAML     : {yaml_path}")
    print("=" * 52)
    print("\nNext step — train YOLOv8:")
    print("  pip install ultralytics")
    print(f"  yolo detect train data={yaml_path} model=yolov8s.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
