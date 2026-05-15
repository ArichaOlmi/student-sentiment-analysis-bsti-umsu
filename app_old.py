import re
from pathlib import Path
from io import BytesIO
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import joblib
from scipy.sparse import save_npz

# (Opsional) menu modern
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# =========================
# 0) Konfigurasi Aplikasi
# =========================
ENABLE_LOGIN = False  # ubah ke True jika ingin menampilkan halaman login sederhana

st.set_page_config(
    page_title="Sistem Klasifikasi Sentimen BSTI UMSU",
    page_icon=None,
    layout="wide"
)

# Load CSS
css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# =========================
# 1) Util: baca CSV aman
# =========================
def read_csv_any(file) -> pd.DataFrame:
    return pd.read_csv(file, encoding="utf-8-sig", sep=None, engine="python")

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# =========================
# 2) Wide → Long (Google Form)
# =========================
KEYWORDS = {
    "Portal Akademik (WEB)": ["portal", "akademik", "ceritakan"],
    "MyClass UMSU": ["myclass", "ceritakan"],
    "WiFi Kampus": ["wifi", "ceritakan"],
    "Helpdesk/Technical Support": ["helpdesk", "ceritakan"],
    "Saran Peningkatan": ["saran", "peningkatan"],
}

def find_col(cols, must_contain):
    for c in cols:
        cl = str(c).lower()
        if all(k in cl for k in must_contain):
            return c
    return None

def detect_timestamp_col(cols):
    for c in cols:
        cl = str(c).lower().strip()
        if cl == "timestamp" or "timestamp" in cl:
            return c
    return None

def wide_to_long_auto(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = df.columns.tolist()

    ts_col = detect_timestamp_col(cols)
    if not ts_col:
        raise ValueError("Kolom Timestamp tidak ditemukan pada CSV mentah. Pastikan file berasal dari Google Form.")

    found_cols = {}
    for layanan, kw in KEYWORDS.items():
        col = find_col(cols, kw)
        found_cols[layanan] = col

    missing = [k for k,v in found_cols.items() if v is None]
    if missing:
        # fallback: tampilkan kandidat agar user bisa cek header
        kandidat = [c for c in cols if ("ceritakan" in str(c).lower() or "saran" in str(c).lower())]
        raise ValueError(f"Kolom komentar tidak lengkap. Missing: {missing}. Kandidat kolom: {kandidat[:10]}")

    # id_pengisian (001..)
    df["id_pengisian"] = [f"{i+1:03d}" for i in range(len(df))]

    rows = []
    for _, r in df.iterrows():
        ts = r[ts_col]
        idp = r["id_pengisian"]
        urut = 0
        for layanan, colname in found_cols.items():
            urut += 1
            komentar = str(r[colname]).strip()
            if komentar == "" or komentar.lower() in ["nan", "none"]:
                continue
            urut_komentar = f"{urut:02d}"
            id_respon = f"{idp}-{urut_komentar}"
            rows.append({
                "id_respon": id_respon,
                "id_pengisian": idp,
                "urut_komentar": urut_komentar,
                "timestamp": ts,
                "layanan": layanan,
                "komentar_teks": komentar,
            })

    out = pd.DataFrame(rows)
    info = {
        "timestamp_col": ts_col,
        "kolom_komentar": found_cols,
        "jumlah_pengisian": int(len(df)),
        "jumlah_komentar_hasil": int(len(out)),
        "kosong_dilewati": int(len(df)*len(found_cols) - len(out)),
    }
    return out, info


# =========================
# 3) Preprocessing teks
# =========================
STOPWORDS = set("""
yang dan di ke dari pada untuk dengan sebagai adalah itu ini atau jika maka karena agar supaya serta
dalam atas bawah antara kepada oleh menjadi telah sudah belum akan bisa dapat harus hanya saja juga
lagi lebih kurang sangat begitu seperti yaitu yakni tentang terhadap para tiap setiap
kami kita saya aku kamu anda dia mereka beliau kalian
sebuah suatu seorang beberapa banyak sedikit
bagi guna demi namun tetapi tapi sedangkan sementara walau meski walaupun
pun lah kah nya ku mu
""".split())

NEGASI = {"tidak", "tak", "bukan", "jangan", "belum", "kurang", "tanpa"}
STOPWORDS = STOPWORDS - NEGASI

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
NON_LETTER_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text).lower().strip()
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = NON_LETTER_RE.sub(" ", t)
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    return t

def tokenize_basic(text: str):
    if not text:
        return []
    return [tok for tok in text.split() if len(tok) > 1]

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def try_stem_tokens(tokens):
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        stemmer = StemmerFactory().create_stemmer()
        return [stemmer.stem(t) for t in tokens]
    except Exception:
        return tokens

def preprocess_dataframe(df_long: pd.DataFrame, do_stemming: bool = True):
    required = ["komentar_teks"]
    missing = [c for c in required if c not in df_long.columns]
    if missing:
        raise ValueError(f"Kolom wajib untuk preprocessing tidak ditemukan: {missing}")

    out = df_long.copy()
    out["teks_casefold"] = out["komentar_teks"].apply(normalize_text)
    toks = out["teks_casefold"].apply(tokenize_basic)
    toks_nostop = toks.apply(remove_stopwords)
    toks_stem = toks_nostop.apply(try_stem_tokens) if do_stemming else toks_nostop

    out["tokens"] = toks.apply(lambda x: str(list(x)))
    out["tokens_nostop"] = toks_nostop.apply(lambda x: str(list(x)))
    out["tokens_stem"] = toks_stem.apply(lambda x: str(list(x)))
    out["teks_bersih"] = toks_stem.apply(lambda x: " ".join(x))

    non_empty = out[out["teks_bersih"].astype(str).str.strip().ne("")].copy()
    return out, non_empty


# =========================
# 4) TF-IDF & Naive Bayes
# =========================
def run_tfidf(df_non: pd.DataFrame, params: dict):
    required = ["teks_bersih", "label_manual"]
    missing = [c for c in required if c not in df_non.columns]
    if missing:
        raise ValueError(f"Kolom wajib untuk TF-IDF tidak ditemukan: {missing}")

    X = df_non["teks_bersih"].astype(str).fillna("")
    y = df_non["label_manual"].astype(str).str.strip().str.lower()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y
    )

    vec = TfidfVectorizer(
        ngram_range=params["ngram_range"],
        min_df=params["min_df"],
        max_df=params["max_df"],
        max_features=params["max_features"]
    )

    X_train_t = vec.fit_transform(X_train)
    X_test_t = vec.transform(X_test)
    return vec, X_train_t, X_test_t, y_train, y_test

def train_naive_bayes(X_train_t, y_train, X_test_t, y_test, alpha: float):
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_t, y_train)

    y_pred = model.predict(X_test_t)
    acc = accuracy_score(y_test, y_pred)

    labels = sorted(y_test.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    report_text = classification_report(y_test, y_pred, digits=4)
    return model, acc, labels, cm, report, report_text


# =========================
# 5) Header (mirip Bab 3)
# =========================
st.markdown(
    """
    <div class="topbar">
      <div class="topbar-left">Sistem Klasifikasi Sentimen BSTI UMSU</div>
      <div class="topbar-right">User: Admin/Peneliti</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# 6) Login (opsional)
# =========================
def page_login():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Halaman Login")
    st.markdown("Masuk ke sistem (mode penelitian).")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.text_input("Username", value="admin", disabled=True)
        st.text_input("Password", value="********", disabled=True)
    with c2:
        st.selectbox("Role", ["Admin"], index=0, disabled=True)
        st.write("")
        if st.button("Login"):
            st.session_state.logged_in = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# 7) Sidebar Menu (mirip Bab 3)
# =========================
PAGES = ["Import Data", "Labeling", "Training & Evaluasi", "Dashboard", "Laporan/Export"]

with st.sidebar:
    st.markdown("### Menu")
    if HAS_OPTION_MENU:
        selected = option_menu(
            menu_title=None,
            options=PAGES,
            icons=[""] * len(PAGES),
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#ffffff"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px", "padding": "10px 12px", "--hover-color": "#fff0f7"},
                "nav-link-selected": {"background-color": "#fff0f7", "color": "#b3005a", "font-weight": "700"},
            },
        )
    else:
        selected = st.radio("", PAGES, index=0)

    st.markdown("---")
    if st.button("Reset"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# =========================
# 8) Page: Import Data
# =========================
def page_import():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Halaman Import Data")

    st.caption("Halaman ini digunakan untuk memasukkan data hasil kuesioner Google Forms ke dalam sistem.")

    colA, colB = st.columns([2, 1])
    with colA:
        up = st.file_uploader("Upload File CSV (Google Forms)", type=["csv"])
    with colB:
        default_path = Path("data/raw/raw_gform.csv")
        st.write("")
        st.write("")
        if default_path.exists():
            if st.button("Pakai File Default"):
                df_raw = pd.read_csv(default_path, encoding="utf-8-sig", sep=None, engine="python")
                st.session_state.df_raw = df_raw
                st.session_state.raw_name = str(default_path)

    if up is not None:
        df_raw = read_csv_any(up)
        st.session_state.df_raw = df_raw
        st.session_state.raw_name = up.name

    if "df_raw" in st.session_state:
        df_raw = st.session_state.df_raw
        st.success(f"File terbaca: {st.session_state.raw_name} | baris: {len(df_raw)} | kolom: {df_raw.shape[1]}")

        st.markdown("**Preview Data (10 baris pertama):**")
        st.dataframe(df_raw.head(10), use_container_width=True)

        # Deteksi tipe dataset
        is_long = ("layanan" in df_raw.columns and "komentar_teks" in df_raw.columns)
        if is_long:
            st.info("Dataset terdeteksi sebagai format long (sudah ada layanan & komentar_teks).")
        else:
            st.info("Dataset terdeteksi sebagai data mentah Google Form (wide). Sistem akan mengubah wide → long.")

        # Aksi import
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            run_import = st.button("Import & Simpan")
        with col2:
            clear = st.button("Clear Data")

        status_box = col3.empty()

        if clear:
            st.session_state.pop("df_raw", None)
            st.session_state.pop("df_long", None)
            st.session_state.pop("raw_name", None)
            st.rerun()

        if run_import:
            with st.spinner("Memproses impor data..."):
                if is_long:
                    df_long = df_raw.copy()
                    info = {"jumlah_komentar_hasil": len(df_long)}
                else:
                    df_long, info = wide_to_long_auto(df_raw)

                # Simpan ke folder data/processed
                out_dir = Path("data/processed")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "dataset_long_620.csv"
                df_long.to_csv(out_path, index=False, encoding="utf-8-sig")

                st.session_state.df_long = df_long
                st.session_state.long_path = str(out_path)

                # Status
                invalid = int((df_long["komentar_teks"].astype(str).str.strip() == "").sum()) if "komentar_teks" in df_long.columns else 0
                duplicate = int(df_long["komentar_teks"].astype(str).str.strip().duplicated().sum()) if "komentar_teks" in df_long.columns else 0
                masuk = int(len(df_long))

                status_box.markdown(
                    f"<div class='status'>Status: invalid={invalid} | duplicate={duplicate} | masuk={masuk}</div>",
                    unsafe_allow_html=True
                )

                st.success(f"Import selesai. File tersimpan: {out_path}")

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# 9) Page: Labeling
# =========================
def page_labeling():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Halaman Labeling Manual")
    st.caption("Menampilkan daftar komentar dan dropdown label (positif/netral/negatif).")

    # Load dataset long
    labeled_path = Path("data/processed/dataset_long_620_labeled.csv")
    long_path = Path("data/processed/dataset_long_620.csv")

    if "df_long" not in st.session_state:
        if labeled_path.exists():
            st.session_state.df_long = pd.read_csv(labeled_path, encoding="utf-8-sig")
            st.session_state.long_path = str(labeled_path)
        elif long_path.exists():
            st.session_state.df_long = pd.read_csv(long_path, encoding="utf-8-sig")
            st.session_state.long_path = str(long_path)

    if "df_long" not in st.session_state:
        st.warning("Data long belum tersedia. Silakan impor data terlebih dahulu pada menu Import Data.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    df = st.session_state.df_long.copy()

    if "label_manual" not in df.columns:
        df["label_manual"] = ""
    if "catatan_label" not in df.columns:
        df["catatan_label"] = ""

    # Filter & pencarian
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        keyword = st.text_input("Filter & Pencarian (kata kunci)", value="")
    with f2:
        status = st.selectbox("Status label", ["Semua", "Belum dilabel", "Sudah dilabel"], index=0)
    with f3:
        layanan_opt = ["Semua"] + sorted(df["layanan"].dropna().unique().tolist()) if "layanan" in df.columns else ["Semua"]
        layanan_filter = st.selectbox("Layanan", layanan_opt, index=0)

    df_f = df.copy()
    if keyword.strip():
        df_f = df_f[df_f["komentar_teks"].astype(str).str.contains(keyword, case=False, na=False)]
    if layanan_filter != "Semua" and "layanan" in df_f.columns:
        df_f = df_f[df_f["layanan"] == layanan_filter]

    labeled_mask = df_f["label_manual"].astype(str).str.strip().ne("")
    if status == "Belum dilabel":
        df_f = df_f[~labeled_mask]
    elif status == "Sudah dilabel":
        df_f = df_f[labeled_mask]

    # Progress
    total = len(df)
    labeled_total = int(df["label_manual"].astype(str).str.strip().ne("").sum())
    progress = 0 if total == 0 else labeled_total / total
    st.markdown(f"<div class='status'>Progress: {labeled_total} dilabel dari {total} data ({progress*100:.1f}%)</div>", unsafe_allow_html=True)

    # Pagination
    page_size = st.selectbox("Jumlah baris per halaman", [10, 20, 50], index=1)
    max_page = max(1, int(np.ceil(len(df_f) / page_size)))
    page = st.number_input("Halaman", min_value=1, max_value=max_page, value=1, step=1)
    start = (page - 1) * page_size
    end = start + page_size

    df_show = df_f.iloc[start:end].copy()

    st.markdown("**Daftar Komentar & Label:**")
    edited = st.data_editor(
        df_show[["id_respon", "komentar_teks", "label_manual"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "label_manual": st.column_config.SelectboxColumn(
                "label_manual",
                options=["", "positif", "netral", "negatif"],
                required=False
            )
        },
        disabled=["id_respon", "komentar_teks"]
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        save_btn = st.button("Simpan")
    with c2:
        save_next_btn = st.button("Simpan & Next")

    if save_btn or save_next_btn:
        # merge edited back to df
        update_map = dict(zip(edited["id_respon"], edited["label_manual"].astype(str)))
        df["label_manual"] = df.apply(lambda r: update_map.get(r["id_respon"], r["label_manual"]), axis=1)

        # Save
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "dataset_long_620_labeled.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        st.session_state.df_long = df
        st.session_state.long_path = str(out_path)
        st.success(f"Data labeling tersimpan: {out_path}")

        if save_next_btn and page < max_page:
            st.session_state._next_page = page + 1
            st.rerun()

    # Apply next page if set
    if "_next_page" in st.session_state:
        st.session_state.pop("_next_page", None)

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# 10) Page: Training & Evaluasi
# =========================
def page_training():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Halaman Training & Evaluasi")
    st.caption("Tombol jalankan proses (preprocessing → TF-IDF → split → train → evaluasi) dan menampilkan metrik serta confusion matrix.")

    # Pastikan dataset labeled tersedia
    labeled_path = Path("data/processed/dataset_long_620_labeled.csv")
    if labeled_path.exists():
        df_long = pd.read_csv(labeled_path, encoding="utf-8-sig")
    else:
        st.warning("Dataset berlabel belum ditemukan. Silakan lakukan labeling atau pastikan file dataset_long_620_labeled.csv ada.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if "label_manual" not in df_long.columns:
        st.warning("Kolom label_manual belum ada. Pastikan dataset sudah dilabel.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Parameter & aksi
    p1, p2, p3 = st.columns([1, 1, 2])
    with p1:
        alpha = st.number_input("alpha", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    with p2:
        random_state = st.number_input("random_state", value=42, step=1)
    with p3:
        st.write("")
        st.write("")
        run_btn = st.button("Jalankan Proses")

    if run_btn:
        with st.spinner("Menjalankan preprocessing, TF-IDF, training, dan evaluasi..."):
            # Preprocessing
            df_pre_all, df_pre_non = preprocess_dataframe(df_long, do_stemming=True)
            # Drop empty and ensure label exists
            df_pre_non = df_pre_non[df_pre_non["label_manual"].astype(str).str.strip().ne("")].copy()
            df_pre_non["label_manual"] = df_pre_non["label_manual"].astype(str).str.strip().str.lower()

            # Simpan preprocessing
            out_dir = Path("data/processed")
            out_dir.mkdir(parents=True, exist_ok=True)
            df_pre_all.to_csv(out_dir / "dataset_preprocessing_all.csv", index=False, encoding="utf-8-sig")
            df_pre_non.to_csv(out_dir / "dataset_preprocessing_non_empty.csv", index=False, encoding="utf-8-sig")

            # TF-IDF params (disederhanakan agar sesuai rancangan)
            tfidf_params = {
                "ngram_range": (1, 1),
                "min_df": 2,
                "max_df": 0.90,
                "max_features": 5000,
                "test_size": 0.20,
                "random_state": int(random_state),
            }

            vec, X_train_t, X_test_t, y_train, y_test = run_tfidf(df_pre_non, tfidf_params)

            # Simpan TF-IDF artefak
            tfidf_dir = Path("data/processed/tfidf")
            tfidf_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(vec, tfidf_dir / "tfidf_vectorizer.joblib")
            save_npz(tfidf_dir / "X_train_tfidf.npz", X_train_t)
            save_npz(tfidf_dir / "X_test_tfidf.npz", X_test_t)
            y_train.to_csv(tfidf_dir / "y_train.csv", index=False, header=["label_manual"], encoding="utf-8-sig")
            y_test.to_csv(tfidf_dir / "y_test.csv", index=False, header=["label_manual"], encoding="utf-8-sig")

            # Train NB
            model, acc, labels, cm, rep_dict, rep_text = train_naive_bayes(X_train_t, y_train, X_test_t, y_test, float(alpha))

            # Simpan NB artefak
            nb_dir = Path("data/processed/naive_bayes")
            nb_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, nb_dir / "model_multinomial_nb.joblib")
            cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
            cm_df.to_excel(nb_dir / "confusion_matrix.xlsx", index=True)
            Path(nb_dir / "classification_report.txt").write_text(rep_text, encoding="utf-8")

            # Store in session for dashboard
            st.session_state.df_pre_non = df_pre_non
            st.session_state.nb_model = model
            st.session_state.vec = vec
            st.session_state.eval = {"acc": acc, "labels": labels, "cm": cm_df, "report_dict": rep_dict, "report_text": rep_text}

            st.success("Proses selesai. Hasil evaluasi ditampilkan di bawah.")

    # Tampilkan hasil evaluasi jika ada
    if "eval" in st.session_state:
        ev = st.session_state.eval
        st.markdown("**Hasil Training & Evaluasi:**")
        st.markdown(f"<div class='status'>Accuracy: {ev['acc']:.4f}</div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Confusion Matrix**")
            st.dataframe(ev["cm"], use_container_width=True)
        with c2:
            st.markdown("**Metrik Evaluasi**")
            # ambil kelas & avg
            rep = ev["report_dict"]
            rows = []
            for k in rep.keys():
                if k in ["accuracy", "macro avg", "weighted avg"] or k in ev["labels"]:
                    if k == "accuracy":
                        continue
                    rows.append({
                        "kelas": k,
                        "precision": rep[k]["precision"],
                        "recall": rep[k]["recall"],
                        "f1-score": rep[k]["f1-score"],
                        "support": rep[k]["support"],
                    })
            met_df = pd.DataFrame(rows)
            st.dataframe(met_df, use_container_width=True)

        st.markdown("**Classification Report (teks):**")
        st.code(ev["report_text"])

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# 11) Page: Dashboard
# =========================
def page_dashboard():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Halaman Dashboard Sentimen")
    st.caption("Grafik proporsi sentimen dan daftar komentar per kelas untuk membantu interpretasi.")

    # sumber data dashboard: gunakan preprocessing non-empty jika ada
    df = None
    if "df_pre_non" in st.session_state:
        df = st.session_state.df_pre_non.copy()
    else:
        p = Path("data/processed/dataset_preprocessing_non_empty.csv")
        if p.exists():
            df = pd.read_csv(p, encoding="utf-8-sig")

    if df is None or len(df) == 0:
        st.warning("Data preprocessing non-empty belum tersedia. Jalankan Training & Evaluasi terlebih dahulu.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Filter (sesuaikan dengan ketersediaan data)
    f1, f2 = st.columns([1, 2])
    with f1:
        layanan_opt = ["Semua"] + sorted(df["layanan"].dropna().unique().tolist()) if "layanan" in df.columns else ["Semua"]
        layanan_filter = st.selectbox("Filter Layanan", layanan_opt, index=0)
    with f2:
        st.write("")

    df_f = df.copy()
    if layanan_filter != "Semua" and "layanan" in df_f.columns:
        df_f = df_f[df_f["layanan"] == layanan_filter]

    # Gunakan label_manual sebagai ringkasan dashboard
    label_col = "label_manual" if "label_manual" in df_f.columns else None
    if label_col is None:
        st.warning("Kolom label_manual tidak ditemukan.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    counts = df_f[label_col].astype(str).str.strip().str.lower().value_counts()
    pos = int(counts.get("positif", 0))
    neu = int(counts.get("netral", 0))
    neg = int(counts.get("negatif", 0))

    # Ringkasan
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='kpi'><div class='kpi-title'>Positif</div><div class='kpi-value'>{pos}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi'><div class='kpi-title'>Netral</div><div class='kpi-value'>{neu}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi'><div class='kpi-title'>Negatif</div><div class='kpi-value'>{neg}</div></div>", unsafe_allow_html=True)

    # Chart sederhana
    st.markdown("**Ringkasan Sentimen**")
    chart_df = pd.DataFrame({"sentimen": ["positif", "netral", "negatif"], "jumlah": [pos, neu, neg]})
    st.bar_chart(chart_df.set_index("sentimen"))

    # Top kata per kelas (dari teks_bersih)
    st.markdown("**Top kata per kelas (berdasarkan teks_bersih)**")
    if "teks_bersih" in df_f.columns:
        cols = st.columns(3)
        for i, lab in enumerate(["positif", "netral", "negatif"]):
            subset = df_f[df_f[label_col].astype(str).str.lower() == lab]
            words = " ".join(subset["teks_bersih"].astype(str).tolist()).split()
            top = Counter(words).most_common(10)
            text = "\n".join([f"- {w} ({c})" for w, c in top]) if top else "-"
            cols[i].markdown(f"**{lab.capitalize()}**\n\n{text}")
    else:
        st.info("Kolom teks_bersih belum tersedia untuk top kata.")

    # Daftar komentar per kelas
    st.markdown("**Daftar Komentar (per kelas)**")
    pilih_kelas = st.selectbox("Filter kelas", ["positif", "netral", "negatif"], index=0)
    df_list = df_f[df_f[label_col].astype(str).str.strip().str.lower() == pilih_kelas].copy()
    show_cols = [c for c in ["id_respon", "layanan", "komentar_teks"] if c in df_list.columns]
    st.dataframe(df_list[show_cols].head(50), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# 12) Page: Export
# =========================
def page_export():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Halaman Laporan / Export")
    st.caption("Unduh ringkasan metrik dan rekap sentimen.")

    # Ambil data yang tersedia
    df_labeled = None
    p_labeled = Path("data/processed/dataset_long_620_labeled.csv")
    if p_labeled.exists():
        df_labeled = pd.read_csv(p_labeled, encoding="utf-8-sig")

    df_non = None
    p_non = Path("data/processed/dataset_preprocessing_non_empty.csv")
    if p_non.exists():
        df_non = pd.read_csv(p_non, encoding="utf-8-sig")

    cm_path = Path("data/processed/naive_bayes/confusion_matrix.xlsx")
    report_path = Path("data/processed/naive_bayes/classification_report.txt")

    # Opsi export
    format_opt = st.selectbox("Format", ["Excel (.xlsx)"], index=0)

    st.markdown("**Konten yang diekspor:**")
    c1, c2 = st.columns(2)
    with c1:
        exp_ringkasan = st.checkbox("Ringkasan sentimen", value=True)
        exp_confusion = st.checkbox("Confusion matrix", value=True)
    with c2:
        exp_report = st.checkbox("Classification report", value=True)
        exp_data = st.checkbox("Data preprocessing non-empty", value=False)

    # Preview
    st.markdown("**Preview Laporan:**")
    if df_non is not None and "label_manual" in df_non.columns:
        ring = df_non["label_manual"].astype(str).str.strip().str.lower().value_counts().reset_index()
        ring.columns = ["sentimen", "jumlah"]
        st.dataframe(ring, use_container_width=True)
    else:
        st.info("Data preprocessing non-empty belum tersedia.")

    if st.button("Unduh Laporan"):
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            if exp_ringkasan and df_non is not None and "label_manual" in df_non.columns:
                ring.to_excel(writer, index=False, sheet_name="ringkasan_sentimen")

            if exp_confusion and cm_path.exists():
                cm_df = pd.read_excel(cm_path)
                cm_df.to_excel(writer, index=False, sheet_name="confusion_matrix")

            if exp_report and report_path.exists():
                rep_text = report_path.read_text(encoding="utf-8")
                pd.DataFrame({"classification_report": rep_text.splitlines()}).to_excel(writer, index=False, sheet_name="classification_report")

            if exp_data and df_non is not None:
                df_non.to_excel(writer, index=False, sheet_name="data_preprocessing_non_empty")

            if df_labeled is not None:
                df_labeled.head(200).to_excel(writer, index=False, sheet_name="data_labeled_sample")

        st.download_button(
            "Download",
            data=bio.getvalue(),
            file_name="laporan_sentimen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# 13) Router
# =========================
if ENABLE_LOGIN:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        page_login()
    else:
        if selected == "Import Data":
            page_import()
        elif selected == "Labeling":
            page_labeling()
        elif selected == "Training & Evaluasi":
            page_training()
        elif selected == "Dashboard":
            page_dashboard()
        elif selected == "Laporan/Export":
            page_export()
else:
    if selected == "Import Data":
        page_import()
    elif selected == "Labeling":
        page_labeling()
    elif selected == "Training & Evaluasi":
        page_training()
    elif selected == "Dashboard":
        page_dashboard()
    elif selected == "Laporan/Export":
        page_export()
