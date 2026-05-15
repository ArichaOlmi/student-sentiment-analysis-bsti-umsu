import streamlit as st
from pathlib import Path
import shutil
import json

from src.ui_utils import load_css, PROCESSED_DIR

APP_TITLE = "Sistem Klasifikasi Sentimen BSTI UMSU"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()

st.title("📊 " + APP_TITLE)
st.caption("Gunakan menu di sidebar: Import Data → Labeling → Preprocessing → Training & Evaluasi → Dashboard → Export.")

# ===================== STATUS FILES =====================
st.markdown("### Status Pipeline")

# File standar output kamu
long_path = PROCESSED_DIR / "dataset_long_620.csv"
labeled_path = PROCESSED_DIR / "dataset_long_620_labeled.csv"
preproc_non = PROCESSED_DIR / "dataset_preprocessing_non_empty.csv"

tfidf_dir = PROCESSED_DIR / "tfidf"
nb_dir = PROCESSED_DIR / "naive_bayes"

vectorizer_path = tfidf_dir / "tfidf_vectorizer.joblib"
model_path = nb_dir / "model_multinomial_nb.joblib"
metrics_path = nb_dir / "metrics.json"
cm_path = nb_dir / "confusion_matrix.png"
report_path = nb_dir / "classification_report.csv"

meta_path = PROCESSED_DIR / "import_meta.json"
report_xlsx = PROCESSED_DIR / "laporan_sentimen.xlsx"

def exists_badge(p: Path) -> str:
    return "✅" if p.exists() else "—"

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Dataset Long", exists_badge(long_path))
with c2:
    st.metric("Dataset Labeled", exists_badge(labeled_path))
with c3:
    st.metric("Preprocessing", exists_badge(preproc_non))
with c4:
    st.metric("Model NB", exists_badge(model_path))
with c5:
    st.metric("Metrics", exists_badge(metrics_path))

c6, c7, c8, c9, c10 = st.columns(5)
with c6:
    st.metric("Vectorizer", exists_badge(vectorizer_path))
with c7:
    st.metric("Confusion Matrix", exists_badge(cm_path))
with c8:
    st.metric("Classification Report", exists_badge(report_path))
with c9:
    st.metric("Import Meta", exists_badge(meta_path))
with c10:
    st.metric("Laporan Excel", exists_badge(report_xlsx))

# Tampilkan info dataset aktif terakhir kalau ada
if meta_path.exists():
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        st.markdown("#### Dataset Aktif Terakhir")
        st.dataframe(
            [
                {"Item": "Sumber", "Nilai": meta.get("source", "-")},
                {"Item": "Nama file", "Nilai": meta.get("source_name", "-")},
                {"Item": "Waktu import", "Nilai": meta.get("imported_at", "-")},
                {"Item": "Format", "Nilai": meta.get("format", "-")},
                {"Item": "Jumlah baris", "Nilai": meta.get("rows", "-")},
                {"Item": "Label sebelum normalisasi", "Nilai": meta.get("label_before", "-")},
                {"Item": "Label setelah normalisasi", "Nilai": meta.get("label_after", "-")},
            ],
            use_container_width=True,
            hide_index=True
        )
    except Exception:
        st.info("import_meta.json ada, tapi gagal dibaca.")

st.markdown("---")

# ===================== RESET PIPELINE =====================
st.markdown("### Reset Pipeline")
st.caption(
    "Reset akan menghapus seluruh hasil proses lama di folder `data/processed/` "
    "(dataset long, labeled, preprocessing, TF-IDF, model, evaluasi, laporan). "
    "Setelah reset, mulai ulang dari Import Data."
)

confirm = st.checkbox("Saya paham. Saya ingin menghapus semua hasil proses lama.")
confirm_text = st.text_input('Ketik: RESET untuk konfirmasi', "")

reset_btn = st.button(
    "Reset Semua Data Proses",
    type="primary",
    disabled=not (confirm and confirm_text.strip().upper() == "RESET")
)

if reset_btn:
    try:
        # Hapus folder processed
        if PROCESSED_DIR.exists():
            shutil.rmtree(PROCESSED_DIR)

        # Buat ulang folder processed kosong
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # Bersihkan session state supaya cursor/page tidak nyangkut
        for k in list(st.session_state.keys()):
            del st.session_state[k]

        st.success("Reset selesai. Silakan lanjut ke Import Data untuk memulai ulang.")
        st.rerun()

    except PermissionError as e:
        st.error(
            "Reset gagal karena ada file yang sedang dibuka/terkunci (biasanya oleh Excel atau preview file). "
            "Tutup semua file di `data/processed/` lalu coba lagi.\n\n"
            f"Detail: {e}"
        )
    except Exception as e:
        st.error(f"Reset gagal: {e}")