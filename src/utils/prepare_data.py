# # prepare_data.py
# from pathlib import Path
# import pandas as pd
# from PIL import Image
# import io
# from tqdm import tqdm

# DATA_DIR = Path("/home/kadam_sanket/projects/sign_extraction/src/dataset/dataset_yolo")

# # Mapping your parquet files to dataset splits
# PARQUET_FILES = {
#     "train": "train-00000-of-00001.parquet",
#     "val": "validation-00000-of-00001.parquet",
#     "test": "test-00000-of-00001.parquet",
# }

# def coco_to_yolo(bbox, img_w, img_h):
#     """Converts COCO [x_min, y_min, w, h] to YOLO [x_center, y_center, w, h] normalized."""
#     x_min, y_min, w, h = bbox
#     x_center = (x_min + w / 2.0) / img_w
#     y_center = (y_min + h / 2.0) / img_h
#     norm_w = w / img_w
#     norm_h = h / img_h
#     return x_center, y_center, norm_w, norm_h

# def process_split(split_name, parquet_path):
#     print(f"Processing split: {split_name} from {parquet_path}...")
#     df = pd.read_parquet(parquet_path)

#     img_dir = DATA_DIR / "images" / split_name
#     lbl_dir = DATA_DIR / "labels" / split_name
#     img_dir.mkdir(parents=True, exist_ok=True)
#     lbl_dir.mkdir(parents=True, exist_ok=True)

#     for idx, row in tqdm(df.iterrows(), total=len(df)):
#         # Extract Image
#         img_raw = row["image"]
#         if isinstance(img_raw, dict) and "bytes" in img_raw:
#             img = Image.open(io.BytesIO(img_raw["bytes"])).convert("RGB")
#         elif isinstance(img_raw, bytes):
#             img = Image.open(io.BytesIO(img_raw)).convert("RGB")
#         else:
#             img = img_raw.convert("RGB")

#         w, h = img.size
#         img_filename = f"{split_name}_{idx:05d}.jpg"
#         txt_filename = f"{split_name}_{idx:05d}.txt"

#         img.save(img_dir / img_filename, "JPEG")

#         # Extract Bounding Boxes
#         label_lines = []
#         objects = row.get("objects", {})

#         # Standard Hugging Face detection schema extraction
#         if isinstance(objects, dict) and "bbox" in objects:
#             bboxes = objects["bbox"]
#             categories = objects.get("category", [0] * len(bboxes))
#             for cat, bbox in zip(categories, bboxes):
#                 xc, yc, nw, nh = coco_to_yolo(bbox, w, h)
#                 label_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

#         # Write YOLO label file
#         with open(lbl_dir / txt_filename, "w") as f:
#             f.write("\n".join(label_lines))

# if __name__ == "__main__":
#     for split, p_file in PARQUET_FILES.items():
#         if Path(p_file).exists():
#             process_split(split, p_file)
#         else:
#             print(f"Warning: {p_file} not found in current folder.")

# src/utils/prepare_data.py
import io
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Project root (where you will execute `uv run`)
BASE_DIR = Path(".")

# Input directory containing train.parquet, validation.parquet, test.parquet
INPUT_DIR = BASE_DIR / "src" / "dataset" / "dataset_yolo"

# Output directory for the converted YOLO images/ and labels/
OUTPUT_DIR = BASE_DIR / "src" / "dataset" / "dataset_yolo"

# Filenames in your directory (adjust extension if they don't have .parquet)
PARQUET_FILES = {
    "train": INPUT_DIR / "train-00000-of-00001.parquet",       # or INPUT_DIR / "train"
    "val": INPUT_DIR / "validation-00000-of-00001.parquet",  # or INPUT_DIR / "val"
    "test": INPUT_DIR / "test-00000-of-00001.parquet",         # or INPUT_DIR / "test"
}

def coco_to_yolo(bbox, img_w, img_h):
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / img_w
    y_center = (y_min + h / 2.0) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    return x_center, y_center, norm_w, norm_h

def process_split(split_name, parquet_path):
    print(f"Processing split: {split_name} from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    img_dir = OUTPUT_DIR / "images" / split_name
    lbl_dir = OUTPUT_DIR / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Extract Image
        img_raw = row["image"]
        if isinstance(img_raw, dict) and "bytes" in img_raw:
            img = Image.open(io.BytesIO(img_raw["bytes"])).convert("RGB")
        elif isinstance(img_raw, bytes):
            img = Image.open(io.BytesIO(img_raw)).convert("RGB")
        else:
            img = img_raw.convert("RGB")

        w, h = img.size
        img_filename = f"{split_name}_{idx:05d}.jpg"
        txt_filename = f"{split_name}_{idx:05d}.txt"

        img.save(img_dir / img_filename, "JPEG")

        # Extract Bounding Boxes
        label_lines = []
        objects = row.get("objects", {})

        if isinstance(objects, dict) and "bbox" in objects:
            bboxes = objects["bbox"]
            categories = objects.get("category", [0] * len(bboxes))
            for cat, bbox in zip(categories, bboxes):
                xc, yc, nw, nh = coco_to_yolo(bbox, w, h)
                label_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        with open(lbl_dir / txt_filename, "w") as f:
            f.write("\n".join(label_lines))

if __name__ == "__main__":
    for split, p_file in PARQUET_FILES.items():
        if Path(p_file).exists():
            process_split(split, p_file)
        else:
            print(f"Error: {p_file} not found. Check filename and path.")