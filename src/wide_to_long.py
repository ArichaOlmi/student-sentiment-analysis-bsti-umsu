import pandas as pd

# ====== KEYWORD UNTUK MENCARI KOLOM KOMENTAR ======
KEYWORDS = {
    "Portal Akademik (WEB)": ["portal", "akademik", "ceritakan"],
    "MyClass UMSU": ["myclass", "ceritakan"],
    "WiFi Kampus": ["wifi", "ceritakan"],
    "Helpdesk/Technical Support": ["helpdesk", "ceritakan"],
    "Saran Peningkatan": ["saran", "peningkatan"]
}

def find_col(cols, must_contain):
    """Ambil kolom pertama yang mengandung semua kata kunci."""
    for c in cols:
        cl = c.lower()
        if all(k in cl for k in must_contain):
            return c
    return None

def detect_timestamp_col(cols):
    """Cari kolom timestamp (biasanya 'Timestamp' atau mengandung 'timestamp')."""
    for c in cols:
        cl = c.lower()
        if cl == "timestamp" or "timestamp" in cl:
            return c
    return None

def wide_to_long_df(df_raw: pd.DataFrame, keywords: dict = None) -> tuple[pd.DataFrame, dict, str]:
    """
    Transform wide → long dari DataFrame Google Forms.
    Return:
      - df_long: DataFrame hasil long
      - found: mapping layanan -> nama kolom komentar yang dipakai
      - ts_col: nama kolom timestamp yang dipakai
    """
    if keywords is None:
        keywords = KEYWORDS

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = df.columns.tolist()

    # Timestamp
    ts_col = detect_timestamp_col(cols)
    if not ts_col:
        raise ValueError("Kolom Timestamp tidak ditemukan. Cek header CSV kamu.")

    # Kolom komentar untuk tiap layanan
    found = {}
    for layanan, kw in keywords.items():
        found[layanan] = find_col(cols, kw)

    missing = [k for k, v in found.items() if v is None]
    if missing:
        # Biar Streamlit bisa menampilkan error yang jelas
        msg = "Kolom komentar belum ketemu untuk: " + ", ".join(missing)
        msg += "\n\nCoba cek header CSV dan sesuaikan KEYWORDS."
        raise ValueError(msg)

    # Buat id_pengisian (001..)
    df["id_pengisian"] = [f"{i+1:03d}" for i in range(len(df))]

    # Wide → Long
    rows = []
    for _, r in df.iterrows():
        ts = r[ts_col]
        idp = r["id_pengisian"]

        urut = 0
        for layanan, colname in found.items():
            urut += 1
            komentar = str(r[colname]).strip()

            if komentar == "" or komentar.lower() in ["nan", "none"]:
                continue

            urut_komentar = f"{urut:02d}"
            id_respon = f"{idp}-{urut_komentar}"

            rows.append({
                "id_respon": id_respon,
                "timestamp": ts,
                "layanan": layanan,
                "komentar_teks": komentar,
                "id_pengisian": idp,
                "urut_komentar": urut_komentar
            })

    out = pd.DataFrame(rows)
    return out, found, ts_col

def read_csv_any(file_or_path):
    """Baca CSV robust (BOM + auto delimiter)."""
    return pd.read_csv(file_or_path, encoding="utf-8-sig", sep=None, engine="python")