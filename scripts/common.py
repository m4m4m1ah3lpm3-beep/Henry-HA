"""
common.py - Cong thuc dung CHUNG cho ca TN3K va BUSI.

Moi thay doi o day se ap dung cho ca hai dataset. KHONG sua rieng trong
prep_tn3k.py hay prep_busi.py, vi nhu vay hai bo se khong con so sanh duoc
(anh huong RQ3 va Buoc 6 - Regression).
"""

import hashlib
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- tham so
MIN_AREA = 50        # bo vung nhieu nho hon 50 pixel
RING_SIZE = 15       # be rong vanh nen quanh ton thuong, dung de tinh contrast
MASK_THRESH = 127    # nguong nhi phan hoa mask

COLUMNS = [
    "image_id", "dataset", "class_label", "orig_split",
    "source_image", "source_mask",
    "width", "height",
    "n_lesions", "lesion_area_px", "lesion_ratio",
    "contrast", "edge_sharpness", "mean_intensity",
    "image_hash", "split",
]


# ------------------------------------------------------------- doc / ghi
def imread_gray(path):
    """Doc anh xam. Dung np.fromfile vi cv2.imread loi voi duong dan
    co dau tieng Viet, khoang trang hoac dau ngoac."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Khong doc duoc anh: {path}")
    return img


def imwrite(path, img):
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Khong ghi duoc anh: {path}")
    buf.tofile(str(path))


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def find_by_stem(folder, stem):
    """Tim file cung ten, khong quan tam duoi file (.jpg / .png)."""
    for p in sorted(Path(folder).iterdir()):
        if p.stem == stem:
            return p
    return None


# -------------------------------------------------------------- xu ly mask
def merge_masks(mask_paths, shape):
    """Gop nhieu file mask thanh MOT mask nhi phan (phep OR).
    BUSI co anh kem _mask_1; TN3K chi co 1 file."""
    h, w = shape
    merged = np.zeros((h, w), np.uint8)
    for mp in mask_paths:
        m = imread_gray(mp)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        merged |= (m > MASK_THRESH).astype(np.uint8)
    return merged


def clean_mask(mask):
    """Bo cac vung nho hon MIN_AREA. Tra ve mask da lam sach va danh sach
    bounding box [x, y, w, h, area] cua tung vung con lai."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep, boxes = [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            keep.append(i)
            boxes.append(stats[i, :5].tolist())
    cleaned = np.isin(lab, keep).astype(np.uint8) if keep else np.zeros_like(mask)
    return cleaned, boxes


# ---------------------------------------------------------- dac trung anh
def extract_features(img, mask):
    """Tinh cac dac trung dung cho EDA (Buoc 3) va Regression (Buoc 6).
    Anh khong co ton thuong -> contrast va edge_sharpness = None (NULL)."""
    h, w = img.shape
    lesion = mask > 0
    area = int(lesion.sum())
    feats = {
        "width": w,
        "height": h,
        "lesion_area_px": area,
        "lesion_ratio": area / (h * w),
        "mean_intensity": float(img.mean()),
        "contrast": None,
        "edge_sharpness": None,
    }
    if area == 0:
        return feats

    # contrast: chenh sang giua ton thuong va vanh nen bao quanh no
    k = np.ones((RING_SIZE, RING_SIZE), np.uint8)
    ring = (cv2.dilate(mask, k) > 0) & ~lesion
    if ring.any():
        feats["contrast"] = float(
            abs(img[lesion].mean() - img[ring].mean()) / 255.0)

    # edge_sharpness: do lon gradient trung binh tren vien ton thuong
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)) > 0
    if edge.any():
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        feats["edge_sharpness"] = float(np.hypot(gx, gy)[edge].mean())

    return feats


def yolo_lines(boxes, w, h, cls_id=0):
    """Doi bounding box sang dinh dang YOLO: cls x_center y_center w h,
    tat ca da chuan hoa ve khoang 0-1."""
    out = []
    for x, y, bw, bh, _ in boxes:
        out.append(f"{cls_id} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                   f"{bw / w:.6f} {bh / h:.6f}")
    return out


# ------------------------------------------------------------- ghi ket qua
def prepare_out_dirs(out):
    out = Path(out)
    for d in ("images", "masks", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    return out


def save_metadata(rows, out):
    out = Path(out)
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df = df.sort_values("image_id").reset_index(drop=True)
    df.to_csv(out / "metadata.csv", index=False, encoding="utf-8")
    with sqlite3.connect(out / "metadata.db") as con:
        df.to_sql("images", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_split ON images(split)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_hash ON images(image_hash)")
    return df


def report(df, extra_group=None):
    """In bang tom tat de doi chieu voi lan chay truoc."""
    print("\n--- Tom tat ---")
    print(f"Tong so anh              : {len(df)}")
    if extra_group:
        for k, v in df[extra_group].value_counts().sort_index().items():
            print(f"  {k:<22} : {v}")
    print(f"Anh khong co ton thuong  : {(df.n_lesions == 0).sum()}")
    print(f"Anh co >1 ton thuong     : {(df.n_lesions > 1).sum()}")

    print("\nSo ton thuong tren anh:")
    for k, v in df.n_lesions.value_counts().sort_index().items():
        print(f"  {k} ton thuong : {v} anh")

    dup = df.image_hash.value_counts()
    dup = dup[dup > 1]
    print(f"\nNhom anh bi trung lap    : {len(dup)}")
    for h in dup.index:
        ids = df.loc[df.image_hash == h, "image_id"].tolist()
        splits = df.loc[df.image_hash == h, "orig_split"].nunique()
        canh_bao = "  <-- TRUNG GIUA 2 TAP GOC!" if splits > 1 else ""
        print(f"  {h[:16]}... : {', '.join(ids)}{canh_bao}")

    print(f"\nlesion_ratio trung binh  : "
          f"{df.lesion_ratio.mean() * 100:.2f}%")
    print(f"So anh thieu contrast     : {df.contrast.isna().sum()} "
          f"(dung bang so anh khong co ton thuong)")
