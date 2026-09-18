"""
prep_tn3k.py - Chuan hoa dataset TN3K (thyroid nodule).

Cach chay:
    python prep_tn3k.py --src D:\\ADY\\raw\\TN3K --out D:\\ADY\\data_tn3k

Ket qua mong doi:
    Tong so anh              : 3493   (trainval 2879, test 614)
    Anh khong co ton thuong  : 0
    Anh co >1 ton thuong     : 294
    Nhom anh bi trung lap    : 0
"""

import argparse
import shutil
from pathlib import Path

import common as C


def find_dir(src, orig_split, kind):
    """TN3K tren Kaggle co nhieu cach dat ten thu muc:
    trainval-image / trainval_image / trainval/image ...
    Ham nay tu do tim thu muc phu hop."""
    src = Path(src)
    cands = [p for p in src.rglob("*") if p.is_dir()]
    for p in cands:
        rel = str(p.relative_to(src)).lower().replace("\\", "/")
        rel = rel.replace("_", "").replace("-", "").replace("/", "")
        if orig_split in rel and kind in rel:
            return p
    raise SystemExit(
        f"\nKhong tim thay thu muc '{orig_split}' + '{kind}' trong {src}\n"
        f"Cac thu muc dang co:\n  " +
        "\n  ".join(str(p.relative_to(src)) for p in cands[:30]))


def make_id(orig_split, stem):
    """trainval/0001.jpg -> trainval_0001.
    Dat ten theo TEN FILE GOC (khong theo thu tu duyet), nen chay lai
    bao nhieu lan cung ra cung mot image_id."""
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits and digits == stem.strip():
        return f"{orig_split}_{int(digits):04d}"
    return f"{orig_split}_{stem}"


def main(src, out):
    out = C.prepare_out_dirs(out)
    rows, missing = [], []

    for orig_split in ("trainval", "test"):
        img_dir = find_dir(src, orig_split, "image")
        mask_dir = find_dir(src, orig_split, "mask")
        print(f"[{orig_split}] anh : {img_dir}")
        print(f"[{orig_split}] mask: {mask_dir}")

        for p in sorted(img_dir.iterdir()):
            if not p.is_file():
                continue
            mp = C.find_by_stem(mask_dir, p.stem)
            if mp is None:
                missing.append(p.name)
                continue

            iid = make_id(orig_split, p.stem)
            img = C.imread_gray(p)
            mask = C.merge_masks([mp], img.shape)
            mask, boxes = C.clean_mask(mask)

            feats = C.extract_features(img, mask)
            C.imwrite(out / "images" / f"{iid}{p.suffix.lower()}", img)
            C.imwrite(out / "masks" / f"{iid}.png", mask * 255)
            (out / "labels" / f"{iid}.txt").write_text(
                "\n".join(C.yolo_lines(boxes, feats["width"], feats["height"])))

            rows.append(dict(
                image_id=iid, dataset="TN3K", class_label="nodule",
                orig_split=orig_split, source_image=p.name, source_mask=mp.name,
                n_lesions=len(boxes), image_hash=C.md5(p), split=None, **feats))

        n = sum(1 for r in rows if r["orig_split"] == orig_split)
        print(f"[{orig_split}] da xu ly: {n} anh\n")

    if missing:
        print(f"CANH BAO: {len(missing)} anh khong tim thay mask, vi du: "
              f"{missing[:5]}")

    df = C.save_metadata(rows, out)
    C.report(df, extra_group="orig_split")

    print(f"\nDa ghi: {out}\\metadata.db va metadata.csv")
    print("Buoc tiep theo: make_split.py --use-orig-split")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thu muc TN3K goc")
    ap.add_argument("--out", required=True, help="thu muc ket qua")
    a = ap.parse_args()
    main(a.src, a.out)
