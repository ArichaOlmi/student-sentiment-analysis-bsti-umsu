from pathlib import Path
import re
import pandas as pd

# =========================
# VALIDASI DATASET (BAB 4.3.3)
# =========================
ALL_PATH = Path("data/processed/dataset_preprocessing_all.csv")
NON_PATH = Path("data/processed/dataset_preprocessing_non_empty.csv")

REQUIRED_ALL = [
    "id_respon","timestamp","layanan","komentar_teks","label_manual",
    "teks_casefold","tokens","tokens_nostop","tokens_stem","teks_bersih"
]
REQUIRED_NON = ["id_respon","timestamp","layanan","komentar_teks","label_manual","teks_bersih"]

ID_PATTERN = re.compile(r"^\d{3}-\d{2}$")
VALID_LABELS = {"positif","netral","negatif"}

def main():
    if not ALL_PATH.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {ALL_PATH}")
    if not NON_PATH.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {NON_PATH}")

    df_all = pd.read_csv(ALL_PATH, encoding="utf-8-sig")
    df_non = pd.read_csv(NON_PATH, encoding="utf-8-sig")

    results = []

    # 1) Kolom wajib
    missing_all = [c for c in REQUIRED_ALL if c not in df_all.columns]
    missing_non = [c for c in REQUIRED_NON if c not in df_non.columns]
    results.append(("Kolom wajib (all)", "OK" if not missing_all else f"Missing: {missing_all}"))
    results.append(("Kolom wajib (non-empty)", "OK" if not missing_non else f"Missing: {missing_non}"))

    # 2) Unik id_respon
    dup_id = int(df_all["id_respon"].duplicated().sum())
    results.append(("Unik id_respon", "OK" if dup_id == 0 else f"Duplikat={dup_id}"))

    # 3) Format id_respon
    invalid_id = int((~df_all["id_respon"].astype(str).str.match(ID_PATTERN)).sum())
    results.append(("Format id_respon (###-##)", "OK" if invalid_id == 0 else f"Invalid={invalid_id}"))

    # 4) Timestamp valid (parse dayfirst)
    ts = pd.to_datetime(df_all["timestamp"], errors="coerce", dayfirst=True)
    bad_ts = int(ts.isna().sum())
    results.append(("Timestamp valid (dayfirst)", "OK" if bad_ts == 0 else f"Gagal parse={bad_ts}"))

    # 5) Label valid
    labs = set(df_non["label_manual"].astype(str).str.strip().str.lower().unique())
    invalid_labels = sorted(list(labs - VALID_LABELS))
    results.append(("Label ∈ {positif,netral,negatif}", "OK" if not invalid_labels else f"Invalid: {invalid_labels}"))

    # 6) Jumlah data
    n_all = len(df_all)
    n_non = len(df_non)
    removed = n_all - n_non
    results.append(("Jumlah data (all)", str(n_all)))
    results.append(("Jumlah data (non-empty)", str(n_non)))
    results.append(("Dibuang (teks_bersih kosong/NaN)", str(removed)))

    # 7) Distribusi kelas
    dist = df_non["label_manual"].astype(str).str.strip().str.lower().value_counts()
    for lab in ["positif","netral","negatif"]:
        results.append((f"Jumlah label {lab}", str(int(dist.get(lab,0)))))

    report_df = pd.DataFrame(results, columns=["Pengecekan", "Hasil"])

    out_dir = Path("data/processed/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_excel(out_dir / "validation_report.xlsx", index=False)
    report_df.to_csv(out_dir / "validation_report.csv", index=False, encoding="utf-8-sig")

    print("=== VALIDATION REPORT ===")
    print(report_df.to_string(index=False))
    print(f"Saved: {out_dir}/validation_report.xlsx")

if __name__ == "__main__":
    main()
