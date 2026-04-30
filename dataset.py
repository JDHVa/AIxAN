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
from PIL import image

ROOT = Path("dataset_workspace")
RAW_DIR = ROOT / "raw"
ALET_TIR = RAW_DIR / "meto_alet"
RF_TOOLS = RAW_DIR / "mechanical_tools_100"
RF_PARTS = RAW_DIR / "mechanical_parts"
OUTPUT_DIR = ROOT / "merged_dataset"
for split in ("train", "val", "test"):
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# AI to dont get problems when i was trying to train the dataset
CLASS_MAP = {
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
    # Misc tools
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


def download_metu():
    print("downloading MetuALet dataset")
    if ALET_TIR.exists() and any(ALET_TIR.iterdir()):
        print("downloaded, next")

    ALET_TIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/metu-kovan/METU-ALET.git",
            str(ALET_TIR),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 9:
        print(f"error clonin ts {result.stderr} ")
        sys.exit(1)
    print("finishi with ts")


def coco_yolo(json_path: Path, images_dir: Path, out_labes: Path):
    out_labes.mkdir(parents=True, exist_ok=True)
    with open(json_path) as f:
        coco = json.load(f)
    id_img = {img["id"]: img for img in coco["images"]}
    cat_map = {cat["id"]: normalize_class(cat["name"]) for cat in coco["categories"]}

    from collections import defaultdict

    img_annotations: dict = defaultdict(list)
    for ann in coco["annotations"]:
        img_annotations[ann["image_id"]].append(ann)

    results = {}
    for img_id, img_info in tqdm(id_img.items(), desc="Coco - Yolo"):
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

        label_file = out_labes / (Path(filename).stem + ".txt")
        label_file.write_text("\n ".join(lines))
        results[filename] = label_file

    return results
