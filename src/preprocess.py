import re
import pandas as pd

# Stopwords sederhana (bisa kamu tambah)
STOPWORDS = set("""
yang dan di ke dari pada untuk dengan sebagai adalah itu ini atau jika maka karena agar supaya serta
dalam atas bawah antara kepada oleh menjadi telah sudah belum akan bisa dapat harus hanya saja juga
lagi lebih kurang sangat begitu seperti yaitu yakni tentang terhadap para tiap setiap
kami kita saya aku kamu anda dia mereka beliau kalian
sebuah suatu seorang beberapa banyak sedikit
bagi guna demi namun tetapi tapi sedangkan sementara walau meski walaupun
pun lah kah nya ku mu
""".split())

# Kata negasi jangan dihapus
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

def tokenize(text: str):
    if not text:
        return []
    return [tok for tok in text.split() if len(tok) > 1]

def remove_stopwords(tokens):
    return [tok for tok in tokens if tok not in STOPWORDS]

def _get_stemmer():
    # Dibuat sekali saja (lebih cepat)
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()

def run_preprocess(
    df: pd.DataFrame,
    use_stopword: bool = True,
    use_stemming: bool = False,   # default OFF biar cepat
    remove_numbers: bool = True,  # disiapkan kalau nanti mau dipakai
):
    """
    Input: df hasil labeling (wajib ada: id_respon, timestamp, layanan, komentar_teks, label_manual)
    Output:
      - df_all: berisi kolom antara (teks_casefold, tokens, tokens_nostop, tokens_stem, teks_bersih)
      - df_non: df_all yang teks_bersih tidak kosong
    """
    required = ["id_respon", "timestamp", "layanan", "komentar_teks", "label_manual"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    out = df.copy()

    # 1) Case folding + cleaning
    out["teks_casefold"] = out["komentar_teks"].apply(normalize_text)

    # 2) Tokenizing
    out["tokens"] = out["teks_casefold"].apply(tokenize)

    # 3) Stopword removal (opsional)
    if use_stopword:
        out["tokens_nostop"] = out["tokens"].apply(remove_stopwords)
    else:
        out["tokens_nostop"] = out["tokens"]

    # 4) Stemming (opsional) - dibuat lebih cepat:
    #    join tokens_nostop -> stem 1x per dokumen (bukan per token)
    if use_stemming:
        stemmer = _get_stemmer()
        def stem_doc(tokens):
            txt = " ".join(tokens)
            if not txt:
                return []
            return stemmer.stem(txt).split()
        out["tokens_stem"] = out["tokens_nostop"].apply(stem_doc)
    else:
        out["tokens_stem"] = out["tokens_nostop"]

    # 5) Gabungkan jadi teks akhir
    out["teks_bersih"] = out["tokens_stem"].apply(lambda x: " ".join(x))

    df_non = out[out["teks_bersih"].astype(str).str.strip().ne("")].copy()
    return out, df_non