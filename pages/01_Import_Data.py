import streamlit as st
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from src.ui_utils import load_css, empty_state, RAW_DIR, PROCESSED_DIR
from src.wide_to_long import read_csv_any, wide_to_long_df

st.set_page_config(
    page_title="Import Data",
    page_icon="📥",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()

st.title("Import Data")
st.caption("Upload CSV RAW/WIDE (Google Forms) atau LONG (dengan/ tanpa label). Lalu klik Import untuk menjadikannya dataset aktif.")

default_path = RAW_DIR / "raw_gform.csv"

out_long_path = PROCESSED_DIR / "dataset_long_620.csv"
out_labeled_path = PROCESSED_DIR / "dataset_long_620_labeled.csv"
meta_path = PROCESSED_DIR / "import_meta.json"


# ===================== Helpers =====================
def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip nama kolom agar 'label_manual ' kebaca jadi 'label_manual'."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def detect_format(df: pd.DataFrame) -> str:
    cols = set(df.columns.astype(str))
    required_long = {"id_respon", "layanan", "komentar_teks"}
    return "LONG" if required_long.issubset(cols) else "RAW/WIDE"

def norm_label(x):
    """
    Normalisasi label jadi hanya: positif/netral/negatif atau ''.
    (bisa kamu tambah mapping kalau labelmu bentuk lain)
    """
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s == "":
        return ""

    mapping = {
        # teks
        "positive": "positif", "positif": "positif", "pos": "positif", "p": "positif",
        "negative": "negatif", "negatif": "negatif", "neg": "negatif", "n": "negatif",
        "neutral": "netral", "netral": "netral", "neu": "netral",

        # angka umum
        "1": "positif", "+1": "positif",
        "0": "netral",
        "-1": "negatif",

        # kosong
        "none": "", "nan": "", "null": "", "-": "",
    }

    s2 = mapping.get(s, s)
    return s2 if s2 in {"positif", "netral", "negatif"} else ""

def ensure_label_manual(df: pd.DataFrame):
    """
    Pastikan kolom label_manual ada.
    Jika ada kolom dengan nama mirip (beda kapital/spasi), rename ke label_manual.
    Return: (df, nonempty_before, nonempty_after, contoh_unrecognized)
    """
    df = std_cols(df)

    # cari kolom label_manual tanpa peduli kapital
    cols_lower = {c.lower(): c for c in df.columns}
    if "label_manual" in cols_lower:
        real_name = cols_lower["label_manual"]
        if real_name != "label_manual":
            df = df.rename(columns={real_name: "label_manual"})
    else:
        df["label_manual"] = ""

    raw = df["label_manual"].astype(str).fillna("").str.strip()
    nonempty_before = int(raw.ne("").sum())

    normalized = df["label_manual"].apply(norm_label)
    nonempty_after = int(normalized.astype(str).str.strip().ne("").sum())

    # nilai yang terisi tapi jadi kosong (berarti tidak dikenali)
    mask_unrec = raw.ne("") & normalized.astype(str).str.strip().eq("")
    contoh_unrec = raw[mask_unrec].unique().tolist()[:15] if mask_unrec.any() else []

    df["label_manual"] = normalized
    return df, nonempty_before, nonempty_after, contoh_unrec

def save_meta(meta: dict):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def load_meta():
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def show_paginated_table(df: pd.DataFrame, *, key_prefix: str):
    if df is None or df.empty:
        empty_state("Tidak ada data untuk ditampilkan.")
        return

    st.markdown("#### Preview Data (Pagination)")
    q = st.text_input("Cari", "", key=f"{key_prefix}_q", placeholder="ketik kata kunci...")
    view = df.copy()

    if q.strip():
        ql = q.strip().lower()
        mask = pd.Series(False, index=view.index)
        for col in view.columns:
            mask = mask | view[col].astype(str).str.lower().str.contains(ql, na=False)
        view = view[mask]

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        page_size = st.selectbox("Baris/halaman", [25, 50, 100, 200, 500], index=1, key=f"{key_prefix}_ps")
    total_pages = max(1, (len(view) + page_size - 1) // page_size)
    with c2:
        page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, step=1, key=f"{key_prefix}_page")
    with c3:
        st.caption(f"Total baris setelah filter: **{len(view)}**")

    start = (page - 1) * page_size
    end = start + page_size
    st.dataframe(view.iloc[start:end], use_container_width=True)
    st.caption(f"Menampilkan baris {start+1}–{min(end, len(view))} dari {len(view)} (halaman {page}/{total_pages}).")


# ===================== UI =====================
uploaded = st.file_uploader("Upload CSV", type=["csv"], key="import_uploader")

colA, colB = st.columns([1, 1])
with colA:
    use_default = st.button("Pakai File Default")
with colB:
    run_import = st.button("Import Sekarang", type="primary")  # CTA utama

# Tampilkan meta dataset aktif terakhir
st.markdown("#### Dataset Aktif Terakhir")
meta = load_meta()
if meta:
    st.dataframe(
        pd.DataFrame(
            [
                ["Sumber", meta.get("source", "-")],
                ["Nama file", meta.get("source_name", "-")],
                ["Waktu import", meta.get("imported_at", "-")],
                ["Format", meta.get("format", "-")],
                ["Jumlah baris", meta.get("rows", "-")],
                ["Label terisi (sebelum normalisasi)", meta.get("label_before", "-")],
                ["Label terisi (setelah normalisasi)", meta.get("label_after", "-")],
                ["Output long", str(out_long_path)],
                ["Output labeled", str(out_labeled_path)],
            ],
            columns=["Item", "Nilai"],
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Belum ada dataset aktif. Silakan import terlebih dahulu.")

# Empty state awal
if not uploaded and not default_path.exists() and not out_long_path.exists():
    empty_state("Belum ada file. Upload CSV atau pastikan file default tersedia di `data/raw/raw_gform.csv`.")
    st.stop()

# ===================== ACTION =====================
if run_import:
    if not uploaded and not use_default:
        st.warning("Upload file dulu, atau klik 'Pakai File Default'.")
        st.stop()

    if use_default and not default_path.exists():
        st.error("File default tidak ditemukan. Pastikan `data/raw/raw_gform.csv` ada.")
        st.stop()

    try:
        with st.spinner("Memproses file..."):
            src_file = uploaded if uploaded else str(default_path)
            src_name = uploaded.name if uploaded else "raw_gform.csv"

            df_in = std_cols(read_csv_any(src_file))
            fmt = detect_format(df_in)

            if fmt == "LONG":
                df_long = df_in.copy()
                fmt_text = "LONG"
            else:
                df_long, _, _ = wide_to_long_df(df_in)
                df_long = std_cols(df_long)
                fmt_text = "RAW/WIDE → LONG"

            # ===== FIX UTAMA: pastikan label_manual kebaca & dinormalisasi =====
            df_long, non_before, non_after, contoh_unrec = ensure_label_manual(df_long)

            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

            # selalu simpan long aktif
            df_long.to_csv(out_long_path, index=False, encoding="utf-8-sig")

            # simpan labeled hanya jika ada label setelah normalisasi
            if non_after > 0:
                df_long.to_csv(out_labeled_path, index=False, encoding="utf-8-sig")
            else:
                if out_labeled_path.exists():
                    out_labeled_path.unlink()

            # simpan meta
            meta = {
                "source": "upload" if uploaded else "default",
                "source_name": src_name,
                "imported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "format": fmt_text,
                "rows": int(len(df_long)),
                "label_before": f"{non_before}/{len(df_long)}",
                "label_after": f"{non_after}/{len(df_long)}",
            }
            save_meta(meta)

        st.success("Import berhasil. Dataset aktif sudah diperbarui.")

        # kalau banyak label tidak dikenali, tampilkan contohnya
        if contoh_unrec:
            st.warning("Ada nilai label yang tidak dikenali dan dikosongkan.")
            st.write("Contoh nilai label yang tidak dikenali:", contoh_unrec)

        # preview dari file aktif (biar pasti bukan “cache”)
        df_active = pd.read_csv(out_long_path, encoding="utf-8-sig")
        show_paginated_table(df_active, key_prefix="import_after")

    except Exception as e:
        st.error(f"Gagal import: {e}")

# ===================== Preview dataset aktif tanpa import ulang =====================
elif out_long_path.exists():
    st.markdown("---")
    st.markdown("#### Preview Dataset Aktif (dari data/processed)")
    df_active = pd.read_csv(out_long_path, encoding="utf-8-sig")
    show_paginated_table(df_active, key_prefix="import_existing")