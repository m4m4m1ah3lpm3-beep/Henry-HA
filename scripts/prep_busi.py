"""
prep_busi.py - Chuan hoa dataset BUSI (breast ultrasound).

Cach chay:
    python prep_busi.py --src D:\\ADY\\raw\\BUSI\\Dataset_BUSI_with_GT ^
                        --out D:\\ADY\\data_busi

--src phai tro den thu muc CHUA 3 thu muc benign / malignant / normal.

Ket qua mong doi:
    Tong so anh              : 780   (benign 437, malignant 210, normal 133)
    Anh khong co ton thuong  : 133
    Anh co >1 ton thuong     : ~15
    Nhom anh bi trung lap    : 1     (benign 433 == malignant 145)

Khac biet so voi TN3K:
  - Mot anh co the co nhieu file mask (_mask, _mask_1) -> gop lai bang phep OR
  - Ten file goc co khoang trang va dau ngoac -> doi sang benign_0001
  - Co lop 'normal' khong ton thuong -> mask den, file label rong
"""

import argparse
import re
from pathlib import Path

import common as C

CLASSES = ("benign", "malignant", "normal")


def make_id(cls, stem):
    """'benign (1)' -> benign_0001. Dat ten theo SO TRONG TEN FILE GOC,
    nen chay lai bao nhieu lan cung ra cung mot image_id."""
    m = re.search(r"\((\d+)\)", stem)
    if m:
        return f"{cls}_{int(m.group(1)):04d}"
    digits = "".join(ch for ch in stem if ch.isdigit())
    return f"{cls}_{int(digits):04d}" if digits else f"{cls}_{stem}"


def main(src, out):
    src = Path(src)
    out = C.prepare_out_dirs(out)

    missing_cls = [c for c in CLASSES if not (src / c).is_dir()]
    if missing_cls:
        raise SystemExit(
            f"\nKhong thay thu muc {missing_cls} trong {src}\n"
            f"--src phai tro den thu muc chua benign/malignant/normal.\n"
            f"Cac thu muc dang co: "
            f"{[p.name for p in src.iterdir() if p.is_dir()][:20]}")

    rows, multi_mask = [], 0

    for cls in CLASSES:
        folder = src / cls
        imgs = [p for p in sorted(folder.glob("*.png"))
                if "_mask" not in p.stem]

        for p in imgs:
            iid = make_id(cls, p.stem)
            img = C.imread_gray(p)

            mask_paths = sorted(folder.glob(f"{p.stem}_mask*.png"))
            if len(mask_paths) > 1:
                multi_mask += 1
            mask = C.merge_masks(mask_paths, img.shape)
            mask, boxes = C.clean_mask(mask)

            feats = C.extract_features(img, mask)
            C.imwrite(out / "images" / f"{iid}.png", img)
            C.imwrite(out / "masks" / f"{iid}.png", mask * 255)
            # anh normal: file label rong -> YOLO coi la anh nen (background)
            (out / "labels" / f"{iid}.txt").write_text(
                "\n".join(C.yolo_lines(boxes, feats["width"], feats["height"])))

            rows.append(dict(
                image_id=iid, dataset="BUSI", class_label=cls,
                orig_split=None, source_image=p.name,
                source_mask=" | ".join(m.name for m in mask_paths) or None,
                n_lesions=len(boxes), image_hash=C.md5(p), split=None, **feats))

        print(f"[{cls:<9}] da xu ly: {len(imgs)} anh")

    df = C.save_metadata(rows, out)
    C.report(df, extra_group="class_label")

    print(f"\nSo anh co nhieu hon 1 FILE mask : {multi_mask}")
    print("  (co the khac so anh co >1 ton thuong, vi 2 mask chong nhau"
          " se gop thanh 1 vung)")

    sai = df[(df.class_label == "normal") & (df.n_lesions > 0)]
    print(f"Anh normal bi nham co ton thuong: {len(sai)}  (phai = 0)")
    if len(sai):
        print("  ", sai.image_id.tolist()[:10])

    print(f"\nDa ghi: {out}\\metadata.db va metadata.csv")
    print("Buoc tiep theo: split_busi.py (chia 70/15/15 phan tang, seed 42)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="thu muc chua benign/malignant/normal")
    ap.add_argument("--out", required=True, help="thu muc ket qua")
    a = ap.parse_args()
    main(a.src, a.out)
