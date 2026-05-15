import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
import joblib

from src.ui_utils import load_css, kpi, card, empty_state, PROCESSED_DIR

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
load_css()

st.title("Dashboard")
st.caption("Ringkasan hasil analisis sentimen dan evaluasi model (membaca artefak hasil training terakhir).")

# ===== PATH =====
nb_dir = PROCESSED_DIR / "naive_bayes"
tfidf_dir = PROCESSED_DIR / "tfidf"

metrics_path = nb_dir / "metrics.json"
cm_img = nb_dir / "confusion_matrix.png"
report_csv = nb_dir / "classification_report.csv"
model_path = nb_dir / "model_multinomial_nb.joblib"
vectorizer_path = tfidf_dir / "tfidf_vectorizer.joblib"

preproc_path = PROCESSED_DIR / "dataset_preprocessing_non_empty.csv"
pred_path = nb_dir / "predictions_all.csv"  # file prediksi full (akan dibuat via tombol)

# ===== EMPTY STATE =====
if not metrics_path.exists():
    empty_state("Belum ada hasil training. Jalankan Training & Evaluasi terlebih dahulu.")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

# ===== TIME: last trained =====
# pakai waktu modifikasi metrics.json sebagai "training terakhir"
JAKARTA_TZ = timezone(timedelta(hours=7))
last_trained_dt = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=JAKARTA_TZ)
last_trained_str = last_trained_dt.strftime("%d-%m-%Y %H:%M:%S WIB")

# ===== KPI ROW =====
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
with c2:
    kpi("Macro F1", f"{metrics.get('macro_f1', 0):.4f}")
with c3:
    kpi("Weighted F1", f"{metrics.get('weighted_f1', 0):.4f}")
with c4:
    kpi("N data uji", str(metrics.get("n_test", "—")))
with c5:
    kpi("Terakhir training", last_trained_dt.strftime("%d-%m-%Y"))

st.caption(f"Terakhir training: **{last_trained_str}**")
st.markdown("")

# ===== MINI BUTTON: lokasi output =====
with st.expander("📁 Lokasi output (folder artefak)"):
    st.write("**TF-IDF artifacts:**")
    st.code(str(tfidf_dir))
    st.write("**Naive Bayes artifacts:**")
    st.code(str(nb_dir))
    st.caption("Catatan: Streamlit tidak bisa membuka File Explorer secara langsung. Salin path di atas dan buka manual.")

# ===== DISTRIBUSI SENTIMEN =====
card("Distribusi Sentimen (Data Non-Empty)", "Ringkasan jumlah data per kelas setelah preprocessing.")
if preproc_path.exists():
    df_pre = pd.read_csv(preproc_path, encoding="utf-8-sig")
    if "label_manual" in df_pre.columns:
        dist = df_pre["label_manual"].astype(str).value_counts()
        st.bar_chart(dist)
    else:
        st.info("Kolom label_manual tidak ditemukan pada dataset preprocessing.")
else:
    st.info("File dataset_preprocessing_non_empty.csv belum tersedia. Jalankan Preprocessing untuk melihat distribusi label.")

st.markdown("")

# ===== TABS =====
tab_pred, tab_top, tab_eval = st.tabs(["Prediksi & Filter", "Top Kata per Kelas", "Evaluasi Model"])

# =========================================================
# TAB 1: PREDIKSI + FILTER LAYANAN + TABEL
# =========================================================
with tab_pred:
    card("Prediksi Full Dataset", "Filter berdasarkan layanan/sentimen dan telusuri hasil prediksi.")

    # Jika prediksi belum ada, sediakan 1 CTA untuk generate
    if not pred_path.exists():
        st.info("File prediksi belum tersedia. Klik tombol di bawah untuk membuat prediksi full dataset (sekali saja).")

        # empty state tambahan jika prasyarat tidak ada
        if (not preproc_path.exists()) or (not model_path.exists()) or (not vectorizer_path.exists()):
            empty_state("Prasyarat belum lengkap. Pastikan: Preprocessing selesai + model & vectorizer sudah tersimpan dari Training.")
        else:
            gen_btn = st.button("Generate Prediksi Full Dataset", type="primary")
            if gen_btn:
                with st.spinner("Membuat prediksi untuk seluruh data..."):
                    df_pre = pd.read_csv(preproc_path, encoding="utf-8-sig")

                    # load model & vectorizer
                    model = joblib.load(model_path)
                    vectorizer = joblib.load(vectorizer_path)

                    X_all = df_pre["teks_bersih"].astype(str).fillna("")
                    X_all_tfidf = vectorizer.transform(X_all)

                    y_pred = model.predict(X_all_tfidf)

                    # prob maksimum (opsional, berguna buat confidence)
                    try:
                        proba = model.predict_proba(X_all_tfidf)
                        prob_max = proba.max(axis=1)
                    except Exception:
                        prob_max = None

                    out = df_pre.copy()
                    out["prediksi"] = y_pred
                    if prob_max is not None:
                        out["prob_max"] = prob_max

                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    out.to_csv(pred_path, index=False, encoding="utf-8-sig")

                st.success(f"Prediksi selesai dan disimpan: {pred_path}")
                st.rerun()

    # Jika sudah ada, tampilkan filter + tabel
    if pred_path.exists():
        df_pred = pd.read_csv(pred_path, encoding="utf-8-sig")

        # Filter controls (bukan CTA berat)
        col1, col2, col3 = st.columns([1.2, 1.2, 2])
        layanan_list = sorted(df_pred["layanan"].dropna().astype(str).unique().tolist()) if "layanan" in df_pred.columns else []
        with col1:
            layanan = st.selectbox("Filter layanan", ["Semua"] + layanan_list)
        with col2:
            sent = st.selectbox("Filter prediksi", ["Semua", "positif", "netral", "negatif"])
        with col3:
            q = st.text_input("Cari kata (komentar/teks)", "")

        view = df_pred.copy()
        if layanan != "Semua" and "layanan" in view.columns:
            view = view[view["layanan"].astype(str) == layanan]
        if sent != "Semua":
            view = view[view["prediksi"].astype(str) == sent]
        if q.strip():
            # cari di komentar_teks kalau ada, fallback ke teks_bersih
            if "komentar_teks" in view.columns:
                view = view[view["komentar_teks"].astype(str).str.contains(q, case=False, na=False)]
            else:
                view = view[view["teks_bersih"].astype(str).str.contains(q, case=False, na=False)]

        st.caption(f"Jumlah data setelah filter: **{len(view)}**")

        # Pagination sederhana
        pcol1, pcol2 = st.columns([1, 1])
        with pcol1:
            page_size = st.selectbox("Baris per halaman", [25, 50, 100, 200], index=1)
        total_pages = max(1, (len(view) + page_size - 1) // page_size)
        with pcol2:
            page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, step=1)

        start = (page - 1) * page_size
        end = start + page_size
        slice_df = view.iloc[start:end].copy()

        cols_show = [c for c in ["id_respon", "layanan", "komentar_teks", "teks_bersih", "label_manual", "prediksi", "prob_max"] if c in slice_df.columns]
        st.dataframe(slice_df[cols_show], use_container_width=True)

        st.caption(f"Menampilkan baris {start+1}–{min(end, len(view))} dari {len(view)} (halaman {page}/{total_pages}).")

# =========================================================
# TAB 2: TOP KATA TF-IDF PER KELAS
# =========================================================
with tab_top:
    card("Top Kata per Kelas (berdasarkan model NB)", "Mengambil kata paling representatif dari bobot log-probability MultinomialNB.")

    if (not model_path.exists()) or (not vectorizer_path.exists()):
        empty_state("Model atau vectorizer belum tersedia. Jalankan Training & Evaluasi terlebih dahulu.")
    else:
        top_k = st.slider("Jumlah top kata per kelas", min_value=10, max_value=50, value=20, step=5)

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # MultinomialNB punya feature_log_prob_ dan classes_
        if not hasattr(model, "feature_log_prob_"):
            st.error("Model yang dimuat bukan MultinomialNB atau tidak memiliki feature_log_prob_.")
        else:
            feature_names = vectorizer.get_feature_names_out()
            classes = list(model.classes_)

            # Tampilkan per kelas sebagai 3 kolom
            cols = st.columns(len(classes)) if len(classes) <= 3 else st.columns(3)

            for i, cls in enumerate(classes):
                col = cols[i % len(cols)]
                with col:
                    st.markdown(f"**Kelas: {cls}**")
                    scores = model.feature_log_prob_[i]
                    top_idx = scores.argsort()[::-1][:top_k]
                    top_terms = [feature_names[j] for j in top_idx]
                    top_scores = [float(scores[j]) for j in top_idx]
                    df_top = pd.DataFrame({"term": top_terms, "score": top_scores})
                    st.dataframe(df_top, use_container_width=True, height=420)

# =========================================================
# TAB 3: EVALUASI (CM + REPORT)
# =========================================================
with tab_eval:
    card("Confusion Matrix", "Perbandingan kelas aktual vs prediksi pada data uji.")
    if cm_img.exists():
        st.image(str(cm_img), use_container_width=True)
    else:
        st.info("File confusion_matrix.png belum tersedia.")

    st.markdown("")
    card("Classification Report", "Precision, recall, F1-score per kelas.")
    if report_csv.exists():
        rep = pd.read_csv(report_csv, encoding="utf-8-sig")
        # rapihin kolom index kalau ada
        if "Unnamed: 0" in rep.columns:
            rep = rep.rename(columns={"Unnamed: 0": "label"})
        st.dataframe(rep, use_container_width=True)
    else:
        st.info("File classification_report.csv belum tersedia.")