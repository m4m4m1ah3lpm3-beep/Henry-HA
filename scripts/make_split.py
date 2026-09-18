"""
make_split.py - Chia train/val/test, ghi cot `split` vao metadata.db.

Dung CHUNG cho ca hai dataset:

  TN3K (giu tap test chinh thuc, chia trainval theo 7:1):
      py make_split.py --db C:\\raw\\data_TN3K\\metadata.db ^
                       --out C:\\raw\\data_TN3K\\splits --use-orig-split

  BUSI (khong co split chinh thuc -> tu chia 70/15/15 phan tang):
      py make_split.py --db C:\\raw\\data_BUSI\\metadata.db ^
                       --out C:\\raw\\data_BUSI\\splits --xoa-file-trung

Ket qua mong doi:
  TN3K : train 2519 / val 360 / test 614
  BUSI : train 546  / val 116 / test 116
         (benign 306/65/65, malignant 147/31/31, normal 93/20/20)

VE ANH TRUNG LAP:
  Anh giong het nhau (cung image_hash) LUON bi loai khoi viec chia
  (split = NULL), du co dung --xoa-file-trung hay khong.
  Them --xoa-file-trung thi xoa luon file anh/mask/label tren o dia.
  Dong metadata van duoc giu lai lam bang chung cho bao cao.
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def loai_anh_trung(df, data_dir, xoa_file):
    """Tim anh trung lap, danh dau de loai, va xoa file neu duoc yeu cau."""
    dup = df.duplicated("image_hash", keep=False)

    if not dup.any():
        print("\n--- Khong co anh trung lap ---")
        return dup

    print(f"\n--- Anh bi loai vi trung lap: {dup.sum()} ---")
    for h, g in df[dup].groupby("image_hash"):
        nhan = g.class_label.nunique()
        canh_bao = "  <-- NHAN MAU THUAN!" if nhan > 1 else ""
        print(f"  {h[:16]}... : {', '.join(g.image_id)}{canh_bao}")

    if not xoa_file:
        print("  -> Chi danh dau split = NULL, file van giu tren o dia.")
        print("     (them --xoa-file-trung neu muon xoa han file)")
        return dup

    print("  -> Dang xoa file tren o dia:")
    n_xoa = 0
    for iid in df.loc[dup, "image_id"]:
        for thu_muc in ("images", "masks", "labels"):
            for p in (data_dir / thu_muc).glob(f"{iid}.*"):
                p.unlink()
                print(f"     da xoa: {p.relative_to(data_dir)}")
                n_xoa += 1
    print(f"  -> Tong cong da xoa {n_xoa} file.")
    print("     (Dong metadata van duoc giu lai lam bang chung.)")
    return dup


def chia_phan_tang(df, test_size, seed, nhan):
    """Chia co phan tang theo lop neu co tu 2 lop tro len."""
    strat = df[nhan] if df[nhan].nunique() > 1 else None
    return train_test_split(df, test_size=test_size,
                            stratify=strat, random_state=seed)


def main(a):
    db_path = Path(a.db)
    data_dir = db_path.parent
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM images", con)
    print(f"Doc duoc {len(df)} anh tu {db_path}")

    df["split"] = None
    dup = loai_anh_trung(df, data_dir, a.xoa_file_trung)
    ok = df[~dup]
    print(f"\nAnh con lai de chia    : {len(ok)}")

    if a.use_orig_split:
        # TN3K: giu nguyen tap test cua tac gia
        test = ok[ok.orig_split == "test"]
        trainval = ok[ok.orig_split != "test"]
        if len(test) == 0:
            raise SystemExit("Khong tim thay anh nao co orig_split='test'. "
                             "Bo co --use-orig-split neu dataset khong co "
                             "split chinh thuc.")
        tr, va = chia_phan_tang(trainval, a.val_ratio, a.seed, "class_label")
        print(f"Giu tap test chinh thuc ({len(test)} anh), "
              f"chia trainval voi ty le val={a.val_ratio}")
    else:
        # BUSI: tu chia 70/15/15
        tr, tmp = chia_phan_tang(ok, a.test_ratio * 2, a.seed, "class_label")
        va, test = chia_phan_tang(tmp, 0.5, a.seed, "class_label")
        print(f"Tu chia phan tang theo class_label, seed={a.seed}")

    df.loc[tr.index, "split"] = "train"
    df.loc[va.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"

    # ---- ghi nguoc vao DB ----
    df.to_sql("images", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_split ON images(split)")
    con.commit()

    # ---- xuat danh sach de doi chieu ----
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        ids = sorted(df.loc[df.split == s, "image_id"])
        (out / f"{s}.txt").write_text("\n".join(ids), encoding="utf-8")

    # ---- bao cao ----
    print("\n--- Ket qua chia ---")
    bang = pd.crosstab(df.class_label, df.split.fillna("(loai bo)"))
    bang = bang.reindex(columns=[c for c in ("train", "val", "test",
                                             "(loai bo)") if c in bang])
    bang.loc["TONG"] = bang.sum()
    print(bang.to_string())

    print("\n--- Do kho giua cac tap (lesion_ratio trung binh) ---")
    for s in ("train", "val", "test"):
        g = df[df.split == s]
        print(f"  {s:<6}: {len(g):>5} anh, {g.lesion_ratio.mean() * 100:.2f}%")

    n_normal = df[(df.split == "test") & (df.n_lesions == 0)].shape[0]
    print(f"\nAnh KHONG co ton thuong trong tap test: {n_normal}")
    if n_normal == 0:
        print("  (Binh thuong voi TN3K. RQ2 se dung anh normal cua BUSI.)")

    print(f"\nDa ghi cot `split` vao {db_path}")
    print(f"Da xuat danh sach vao {out}")
    print("Buoc tiep theo: export_split.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="duong dan metadata.db")
    p.add_argument("--out", required=True, help="thu muc xuat danh sach")
    p.add_argument("--use-orig-split", action="store_true",
                   help="giu tap test chinh thuc (dung cho TN3K)")
    p.add_argument("--xoa-file-trung", action="store_true",
                   help="xoa han file anh/mask/label cua anh trung lap")
    p.add_argument("--val-ratio", type=float, default=0.125,
                   help="ty le val tach tu trainval, mac dinh 0.125 (7:1)")
    p.add_argument("--test-ratio", type=float, default=0.15,
                   help="ty le test khi tu chia, mac dinh 0.15")
    p.add_argument("--seed", type=int, default=SEED)
    main(p.parse_args())
