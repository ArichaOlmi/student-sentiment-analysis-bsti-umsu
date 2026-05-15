import streamlit as st
import pandas as pd

from src.ui_utils import load_css, empty_state, PROCESSED_DIR

st.set_page_config(
    page_title="Labeling",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()

st.title("Labeling")
st.caption("Ubah label langsung di tabel (dropdown). Gunakan filter + Prev/Next untuk mempercepat.")

# ===== PATH =====
long_path = PROCESSED_DIR / "dataset_long_620.csv"
labeled_path = PROCESSED_DIR / "dataset_long_620_labeled.csv"
out_path = labeled_path  # selalu simpan ke file labeled

# ===== EMPTY STATE =====
if not long_path.exists() and not labeled_path.exists():
    empty_state("Belum ada dataset. Silakan lakukan Import Data terlebih dahulu.")
    st.stop()

# ===== LOAD (prioritas file labeled jika ada) =====
src_path = labeled_path if labeled_path.exists() else long_path
df_master = pd.read_csv(src_path, encoding="utf-8-sig")

# kolom wajib
required = ["id_respon", "timestamp", "layanan", "komentar_teks"]
missing = [c for c in required if c not in df_master.columns]
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {missing}")
    st.stop()

# pastikan label_manual ada
if "label_manual" not in df_master.columns:
    df_master["label_manual"] = ""

# ===== helpers =====
def trunc(text, n=110):
    t = "" if pd.isna(text) else str(text)
    t = t.strip()
    return (t[: n - 1] + "…") if len(t) > n else t

def norm_label(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    mapping = {
        "positive": "positif", "positif": "positif", "pos": "positif", "p": "positif",
        "negative": "negatif", "negatif": "negatif", "neg": "negatif", "n": "negatif",
        "neutral": "netral", "netral": "netral", "neu": "netral",
        "none": "", "nan": "", "null": "", "-": ""
    }
    s = mapping.get(s, s)
    return s if s in {"positif", "netral", "negatif"} else ""

# normalisasi label
df_master["label_manual"] = df_master["label_manual"].apply(norm_label)

# status counts
s = df_master["label_manual"].astype(str).str.strip()
total = len(df_master)
labeled_n = int((s != "").sum())
unlabeled_n = total - labeled_n
st.caption(f"Total: **{total}** | Sudah dilabel: **{labeled_n}** | Belum dilabel: **{unlabeled_n}**")

# ===== FILTER BAR =====
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.6, 1.2])

default_filter_index = 0 if labeled_n > 0 else 1
with c1:
    label_filter = st.selectbox(
        "Filter label",
        ["Semua", "Belum dilabel", "Positif", "Netral", "Negatif"],
        index=default_filter_index
    )
with c2:
    layanan_list = sorted(df_master["layanan"].dropna().astype(str).unique().tolist())
    layanan = st.selectbox("Layanan", ["Semua"] + layanan_list)
with c3:
    q = st.text_input("Cari", "", placeholder="contoh: error, lemot, bagus...")
with c4:
    page_size = st.selectbox("Baris/halaman", [25, 50, 100, 200], index=1)

view = df_master.copy()

# filter label
if label_filter == "Belum dilabel":
    view = view[view["label_manual"].astype(str).str.strip() == ""]
elif label_filter == "Positif":
    view = view[view["label_manual"] == "positif"]
elif label_filter == "Netral":
    view = view[view["label_manual"] == "netral"]
elif label_filter == "Negatif":
    view = view[view["label_manual"] == "negatif"]

# filter layanan
if layanan != "Semua":
    view = view[view["layanan"].astype(str) == layanan]

# search
if q.strip():
    view = view[view["komentar_teks"].astype(str).str.contains(q, case=False, na=False)]

if len(view) == 0:
    empty_state("Tidak ada data sesuai filter. Coba ubah Filter label / Layanan / kata kunci.")
    st.stop()

# ===== CURSOR (Prev/Next) =====
ids = view["id_respon"].astype(str).tolist()

if "active_id" not in st.session_state or st.session_state["active_id"] not in ids:
    st.session_state["active_id"] = ids[0]

active_id = st.session_state["active_id"]
active_pos = ids.index(active_id)

total_pages = max(1, (len(view) + page_size - 1) // page_size)
if "page" not in st.session_state:
    st.session_state["page"] = (active_pos // page_size) + 1
st.session_state["page"] = max(1, min(total_pages, int(st.session_state["page"])))

n1, n2, n3 = st.columns([1, 1, 3])
with n1:
    prev_btn = st.button("← Prev")
with n2:
    next_btn = st.button("Next →")
with n3:
    st.caption(f"Posisi: **{active_pos+1}/{len(ids)}** | Halaman: **{st.session_state['page']}/{total_pages}**")

def jump_to(pos):
    pos = max(0, min(len(ids) - 1, pos))
    st.session_state["active_id"] = ids[pos]
    st.session_state["page"] = (pos // page_size) + 1
    st.rerun()

if prev_btn:
    jump_to(active_pos - 1)
if next_btn:
    jump_to(active_pos + 1)

# ===== PAGINATION =====
page = st.number_input(
    "Halaman",
    min_value=1,
    max_value=total_pages,
    value=int(st.session_state["page"]),
    step=1
)
st.session_state["page"] = int(page)

start = (page - 1) * page_size
end = start + page_size
page_df = view.iloc[start:end].copy()

# ===== TABLE (INLINE DROPDOWN) =====
page_df["_"] = page_df["id_respon"].astype(str).apply(lambda x: "▶" if x == active_id else "")
page_df["komentar_singkat"] = page_df["komentar_teks"].apply(lambda x: trunc(x, 110))

show_cols = ["_", "id_respon", "layanan", "komentar_singkat", "label_manual"]
page_show = page_df[show_cols].copy()

st.markdown("#### Daftar Komentar (edit label langsung di tabel)")
edited = st.data_editor(
    page_show,
    use_container_width=True,
    height=360,
    hide_index=True,
    disabled=["_", "id_respon", "layanan", "komentar_singkat"],
    column_config={
        "_": st.column_config.TextColumn(""),
        "komentar_singkat": st.column_config.TextColumn("Komentar", help="Komentar dipotong (truncate)."),
        "label_manual": st.column_config.SelectboxColumn(
            "label_manual",
            options=["", "positif", "netral", "negatif"],
            required=False,
            help="Pilih label tanpa mengetik"
        ),
    },
    key=f"label_editor_{label_filter}_{layanan}_{page}"
)

# ===== MERGE EDITS -> MASTER + AUTOSAVE =====
changed = False
before = page_show.set_index("id_respon")["label_manual"].astype(str)
after = edited.set_index("id_respon")["label_manual"].astype(str)

for rid, new_val in after.items():
    new_val = norm_label(new_val)
    old_val = norm_label(before.get(rid, ""))
    if new_val != old_val:
        idx_master = df_master.index[df_master["id_respon"].astype(str) == str(rid)]
        if len(idx_master) > 0:
            df_master.at[idx_master[0], "label_manual"] = new_val
            changed = True

if changed:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(out_path, index=False, encoding="utf-8-sig")

    # kalau filter "Belum dilabel", baris yang baru dilabel akan keluar -> auto pindah
    if label_filter == "Belum dilabel":
        if active_pos < len(ids) - 1:
            st.session_state["active_id"] = ids[active_pos + 1]
        elif active_pos > 0:
            st.session_state["active_id"] = ids[active_pos - 1]
        st.toast("Label tersimpan. Pindah ke data berikutnya.")
        st.rerun()
    else:
        st.toast("Label tersimpan.")