import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_split(
    df: pd.DataFrame,
    *,
    text_col: str = "teks_bersih",
    label_col: str = "label_manual",
    test_size: float = 0.20,
    random_state: int = 42,
    ngram_range: tuple = (1, 1),
    min_df: int = 2,
    max_df: float = 0.90,
    max_features: int | None = 5000,
):
    """
    Membentuk TF-IDF + split train-test.
    - Label kosong dibuang.
    - Stratify otomatis dimatikan jika ada kelas < 2 agar tidak error.

    Return:
      X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, info(dict)
    """
    required = [text_col, label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    X = df[text_col].astype(str).fillna("")
    y = df[label_col].astype(str).str.strip().str.lower()

    # buang label kosong
    mask = y.ne("")
    X = X[mask]
    y = y[mask]

    if len(y) < 10:
        raise ValueError(f"Data berlabel terlalu sedikit untuk training: {len(y)} baris.")

    # cek minimal per kelas
    vc = y.value_counts()
    too_small = vc[vc < 2]
    use_stratify = True

    if len(too_small) > 0:
        # fallback: stratify dimatikan biar tidak error
        use_stratify = False

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if use_stratify else None,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    info = {
        "ngram_range": list(ngram_range),
        "min_df": int(min_df),
        "max_df": float(max_df),
        "max_features": None if max_features is None else int(max_features),
        "test_size": float(test_size),
        "train_size": float(1 - test_size),
        "random_state": int(random_state),
        "stratify_used": use_stratify,
        "label_counts": vc.to_dict(),
        "train_n": int(len(X_train)),
        "test_n": int(len(X_test)),
        "n_features": int(len(vectorizer.get_feature_names_out())),
        "X_train_shape": [int(X_train_tfidf.shape[0]), int(X_train_tfidf.shape[1])],
        "X_test_shape": [int(X_test_tfidf.shape[0]), int(X_test_tfidf.shape[1])],
    }

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, info


def save_tfidf_artifacts(
    out_dir: Path,
    *,
    vectorizer,
    X_train_tfidf=None,
    X_test_tfidf=None,
    y_train=None,
    y_test=None,
    info: dict | None = None,
    save_matrices: bool = False,
):
    """
    Simpan artefak TF-IDF.
    - Selalu simpan vectorizer (.joblib) + info (.json) + y_train/y_test (.csv)
    - Simpan matriks npz opsional (biar tidak berat).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # vectorizer
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")

    # info parameter
    if info is not None:
        (out_dir / "tfidf_info.json").write_text(
            __import__("json").dumps(info, indent=2),
            encoding="utf-8"
        )

    # label train/test
    if (y_train is not None) and (y_test is not None):
        y_train.to_csv(out_dir / "y_train.csv", index=False, header=["label_manual"], encoding="utf-8-sig")
        y_test.to_csv(out_dir / "y_test.csv", index=False, header=["label_manual"], encoding="utf-8-sig")

    # matriks npz (opsional)
    if save_matrices and (X_train_tfidf is not None) and (X_test_tfidf is not None):
        from scipy.sparse import save_npz
        save_npz(out_dir / "X_train_tfidf.npz", X_train_tfidf)
        save_npz(out_dir / "X_test_tfidf.npz", X_test_tfidf)