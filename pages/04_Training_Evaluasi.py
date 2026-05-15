import streamlit as st
import pandas as pd
import json

from src.ui_utils import load_css, empty_state, PROCESSED_DIR
from src.tfidf_vectorize import build_tfidf_split, save_tfidf_artifacts
from src.train_nb import train_multinomial_nb, save_nb_model
from src.evaluate_nb import evaluate_and_save

st.set_page_config(page_title="Training & Evaluasi", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
load_css()

st.title("Training & Evaluasi")
st.caption("Ekstraksi fitur TF-IDF dan klasifikasi Multinomial Naive Bayes. Proses berjalan hanya saat tombol ditekan.")

# ===== PATH =====
in_path = PROCESSED_DIR / "dataset_preprocessing_non_empty.csv"
tfidf_dir = PROCESSED_DIR / "tfidf"
nb_dir = PROCESSED_DIR / "naive_bayes"
model_path = nb_dir / "model_multinomial_nb.joblib"
metrics_path = nb_dir / "metrics.json"

contoh_path = tfidf_dir / "contoh_tfidf_10_dokumen.xlsx"
train_npz = tfidf_dir / "X_train_tfidf.npz"
test_npz = tfidf_dir / "X_test_tfidf.npz"

if not in_path.exists():
    empty_state("Belum ada hasil preprocessing (`dataset_preprocessing_non_empty.csv`). Jalankan Preprocessing terlebih dahulu.")
    st.stop()

df = pd.read_csv(in_path, encoding="utf-8-sig")

# ===== PARAMETER (tanpa card) =====
c1, c2, c3, c4 = st.columns(4)
with c1:
    ngram = st.selectbox("ngram_range", options=[(1, 1), (1, 2)], index=0)
with c2:
    min_df = st.number_input("min_df", value=2, step=1)
with c3:
    max_df = st.number_input("max_df", value=0.90, step=0.05)
with c4:
    max_features = st.number_input("max_features", value=5000, step=500)

c5, c6, c7 = st.columns(3)
with c5:
    test_size = st.slider("test_size", 0.10, 0.40, 0.20, 0.05)
with c6:
    alpha = st.number_input("alpha (NB)", value=1.0, step=0.1)
with c7:
    random_state = st.number_input("random_state", value=42, step=1)

# ===== OUTPUT OPTIONS =====
o1, o2 = st.columns(2)
with o1:
    save_matrices = st.checkbox("Simpan matriks TF-IDF (npz)", value=False)
with o2:
    export_sample_excel = st.checkbox("Export contoh TF-IDF (10 dokumen)", value=False)

train_btn = st.button("Train Model", type="primary")

# ===== TRAINING (hanya saat tombol ditekan) =====
if train_btn:
    try:
        with st.spinner("Membentuk TF-IDF, training NB, dan evaluasi..."):
            X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, info = build_tfidf_split(
                df,
                ngram_range=ngram,
                min_df=int(min_df),
                max_df=float(max_df),
                max_features=int(max_features) if max_features else None,
                test_size=float(test_size),
                random_state=int(random_state),
            )

            save_tfidf_artifacts(
                tfidf_dir,
                vectorizer=vectorizer,
                X_train_tfidf=X_train_tfidf,
                X_test_tfidf=X_test_tfidf,
                y_train=y_train,
                y_test=y_test,
                info=info,
                save_matrices=save_matrices,
            )

            # contoh tfidf
            if export_sample_excel:
                feature_names = vectorizer.get_feature_names_out()
                sample = X_train_tfidf[:10].toarray()
                sample_df = pd.DataFrame(sample, columns=feature_names)
                top_cols = sample_df.mean(axis=0).sort_values(ascending=False).head(20).index.tolist()
                sample_small = sample_df[top_cols]
                sample_small.insert(0, "label_manual", y_train.iloc[:10].values)
                tfidf_dir.mkdir(parents=True, exist_ok=True)
                sample_small.to_excel(contoh_path, index=False)

            model = train_multinomial_nb(X_train_tfidf, y_train, alpha=float(alpha))
            save_nb_model(model, model_path)

            metrics = evaluate_and_save(
                model,
                X_test_tfidf,
                y_test,
                out_dir=nb_dir,
                labels_order=["negatif", "netral", "positif"],
            )

        st.success("Training & evaluasi selesai. Silakan buka Dashboard.")

    except Exception as e:
        st.error(str(e))
        st.stop()

# ===== HASIL TERAKHIR (selalu tampil kalau ada) =====
st.markdown("---")
if metrics_path.exists():
    st.info("Hasil training terakhir sudah tersedia.")
    try:
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
    except Exception:
        st.write("metrics.json ada, tapi gagal dibaca.")
else:
    st.caption("Belum ada metrics.json. Klik Train Model untuk membuat hasil evaluasi.")

# ===== DOWNLOADS (selalu tampil jika file ada) =====
st.markdown("### Download Output")

dcol1, dcol2, dcol3 = st.columns([1, 1, 1])

with dcol1:
    if contoh_path.exists():
        st.download_button(
            "Download contoh TF-IDF (xlsx)",
            data=contoh_path.read_bytes(),
            file_name="contoh_tfidf_10_dokumen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.caption("contoh TF-IDF belum ada.")

with dcol2:
    if train_npz.exists():
        st.download_button(
            "Download X_train_tfidf.npz",
            data=train_npz.read_bytes(),
            file_name="X_train_tfidf.npz",
            mime="application/octet-stream"
        )
    else:
        st.caption("X_train_tfidf.npz belum ada.")

with dcol3:
    if test_npz.exists():
        st.download_button(
            "Download X_test_tfidf.npz",
            data=test_npz.read_bytes(),
            file_name="X_test_tfidf.npz",
            mime="application/octet-stream"
        )
    else:
        st.caption("X_test_tfidf.npz belum ada.")