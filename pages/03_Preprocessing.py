import streamlit as st
import pandas as pd

from src.ui_utils import load_css, empty_state, PROCESSED_DIR
from src.preprocess import run_preprocess

st.set_page_config(page_title="Preprocessing", page_icon="🧹", layout="wide", initial_sidebar_state="expanded")
load_css()

st.title("Preprocessing")
st.caption("Membersihkan komentar agar siap diekstraksi TF-IDF dan digunakan pada training model.")

# ===== PATH =====
in_path = PROCESSED_DIR / "dataset_long_620_labeled.csv"
out_all = PROCESSED_DIR / "dataset_preprocessing_all.csv"
out_non = PROCESSED_DIR / "dataset_preprocessing_non_empty.csv"

# ===== EMPTY STATE =====
if not in_path.exists():
    empty_state("Belum ada file hasil labeling (`dataset_long_620_labeled.csv`). Selesaikan Labeling terlebih dahulu.")
    st.stop()

df = pd.read_csv(in_path, encoding="utf-8-sig")

# ===== Konfigurasi (tanpa card) =====
col1, col2, col3 = st.columns(3)
with col1:
    use_stopword = st.checkbox("Stopword removal", value=True)
with col2:
    use_stemming = st.checkbox("Stemming (lebih berat)", value=False)
with col3:
    remove_numbers = st.checkbox("Hapus angka", value=True)

# ===== 1 CTA =====
run_btn = st.button("Jalankan Preprocessing", type="primary")

if run_btn:
    with st.spinner("Preprocessing berjalan..."):
        df_all, df_non = run_preprocess(
            df,
            use_stopword=use_stopword,
            use_stemming=use_stemming,
            remove_numbers=remove_numbers
        )

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(out_all, index=False, encoding="utf-8-sig")
        df_non.to_csv(out_non, index=False, encoding="utf-8-sig")

    st.success("Preprocessing selesai dan output tersimpan.")
    st.write(f"Total (all): {len(df_all)} | Non-empty: {len(df_non)}")

    if "label_manual" in df_non.columns:
        st.write("Distribusi label (non-empty):")
        st.write(df_non["label_manual"].value_counts())

    st.markdown("#### Preview (non-empty)")
    cols_preview = [c for c in ["id_respon", "layanan", "komentar_teks", "teks_bersih", "label_manual"] if c in df_non.columns]
    st.dataframe(df_non[cols_preview].head(50), use_container_width=True)

else:
    if out_non.exists():
        prev = pd.read_csv(out_non, encoding="utf-8-sig")
        st.info(f"Hasil preprocessing sudah ada ({len(prev)} baris). Kamu bisa lanjut ke Training & Evaluasi.")
        cols_preview = [c for c in ["id_respon", "teks_bersih", "label_manual"] if c in prev.columns]
        st.dataframe(prev[cols_preview].head(30), use_container_width=True)
    else:
        empty_state("Belum ada output preprocessing. Klik tombol 'Jalankan Preprocessing' untuk memulai.")