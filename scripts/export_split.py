"""
export_split.py - Xuat bo du lieu da chia san theo train/val/test.

Cach chay (xuat ca hai bo mot lan):
    py export_split.py --src C:\\raw\\data_TN3K --out C:\\raw\\TN3K_split
    py export_split.py --src C:\\raw\\data_BUSI --out C:\\raw\\BUSI_split

Cau truc ket qua:
    TN3K_split\\
    ├─ images\\train\\  val\\  test\\
    ├─ labels\\train\\  val\\  test\\   (nhan YOLO; anh normal -> file rong)
    └─ masks\\ train\\  val\\  test\\   (dung cho U-Net, Swin-UNet)

Kem theo file data.yaml cho YOLO va SPLIT_INFO.txt de ghi lai cau hinh.

Anh co split = NULL (anh trung lap) tu dong bi bo qua.
"""

import argparse
import shutil
import sqlite3
from pathlib import Path

SPLITS = ("train", "val", "test")
SUB = ("images", "labels", "masks")


def tim_file(folder, stem):
    """Tim file theo ten, khong quan tam duoi file."""
    for p in folder.glob(stem + ".*"):
        return p
    return None


def export(src, dst, ghi_de):
    src, dst = Path(src), Path(dst)
    if dst.exists():
        if not ghi_de:
            raise SystemExit(f"{dst} da ton tai. Them --ghi-de de xoa va "
                             f"tao lai, hoac doi ten thu muc cu.")
        print(f"Xoa thu muc cu: {dst}")
        shutil.rmtree(dst)

    con = sqlite3.connect(src / "metadata.db")
    rows = con.execute(
        "SELECT image_id, split FROM images WHERE split IS NOT NULL"
    ).fetchall()
    bo_qua = con.execute(
        "SELECT COUNT(*) FROM images WHERE split IS NULL").fetchone()[0]

    for d in SUB:
        for s in SPLITS:
            (dst / d / s).mkdir(parents=True, exist_ok=True)

    thieu_mask, thieu_label = [], []
    for iid, sp in rows:
        # anh (bat buoc)
        p = tim_file(src / "images", iid)
        if p is None:
            raise SystemExit(f"Khong tim thay anh cua {iid} trong "
                             f"{src / 'images'}")
        shutil.copy2(p, dst / "images" / sp / p.name)

        # mask
        m = tim_file(src / "masks", iid)
        if m is None:
            thieu_mask.append(iid)
        else:
            shutil.copy2(m, dst / "masks" / sp / m.name)

        # nhan YOLO; khong co thi tao file rong (anh nen)
        lb = tim_file(src / "labels", iid)
        dich = dst / "labels" / sp / f"{iid}.txt"
        if lb is None:
            thieu_label.append(iid)
            dich.touch()
        else:
            shutil.copy2(lb, dich)

    # ---- data.yaml cho YOLO ----
    (dst / "data.yaml").write_text(
        f"# Duong dan tuyet doi tren may nay. Neu chuyen sang Colab,\n"
        f"# sua lai dong `path` cho khop.\n"
        f"path: {dst.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n  0: lesion\n", encoding="utf-8")

    # ---- bao cao ----
    print(f"\n--- {dst.name} ---")
    print(f"{'tap':<8}{'images':>8}{'labels':>8}{'masks':>8}{'rong':>8}")
    tong = {d: 0 for d in SUB}
    for s in SPLITS:
        dem = {d: len(list((dst / d / s).iterdir())) for d in SUB}
        rong = sum(1 for p in (dst / "labels" / s).iterdir()
                   if p.stat().st_size == 0)
        for d in SUB:
            tong[d] += dem[d]
        print(f"{s:<8}{dem['images']:>8}{dem['labels']:>8}"
              f"{dem['masks']:>8}{rong:>8}")
    print(f"{'TONG':<8}{tong['images']:>8}{tong['labels']:>8}"
          f"{tong['masks']:>8}")
    print("(cot 'rong' = so file nhan rong, tuc anh nen / anh normal)")

    if bo_qua:
        print(f"\nDa bo qua {bo_qua} anh co split = NULL (anh trung lap).")
    if thieu_mask:
        print(f"CANH BAO: {len(thieu_mask)} anh thieu mask: {thieu_mask[:5]}")
    if thieu_label:
        print(f"Ghi chu: {len(thieu_label)} anh khong co file nhan "
              f"-> da tao file rong.")

    canh_bao = ""
    if tong["images"] != tong["labels"] or tong["images"] != tong["masks"]:
        canh_bao = "  <-- LECH SO FILE, KIEM TRA LAI!"
    print(f"\nKiem tra images = labels = masks: "
          f"{'DAT' if not canh_bao else 'KHONG DAT'}{canh_bao}")

    # ---- ghi lai cau hinh ----
    info = [f"Dataset      : {dst.name}",
            f"Nguon        : {src.resolve()}",
            f"Seed         : 42",
            f"Anh bi loai  : {bo_qua} (trung lap, xem metadata.db)",
            ""]
    for s in SPLITS:
        info.append(f"{s:<8}: {len(list((dst / 'images' / s).iterdir()))} anh")
    (dst / "SPLIT_INFO.txt").write_text("\n".join(info), encoding="utf-8")
    print(f"\nDa ghi: {dst}\\data.yaml va SPLIT_INFO.txt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="thu muc data_TN3K hoac data_BUSI")
    p.add_argument("--out", required=True, help="thu muc ket qua")
    p.add_argument("--ghi-de", action="store_true",
                   help="xoa thu muc ket qua cu neu da ton tai")
    a = p.parse_args()
    export(a.src, a.out, a.ghi_de)
