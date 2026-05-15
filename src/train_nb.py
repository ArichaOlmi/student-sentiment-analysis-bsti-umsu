from pathlib import Path
import joblib
from sklearn.naive_bayes import MultinomialNB

def train_multinomial_nb(X_train, y_train, *, alpha: float = 1.0):
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def save_nb_model(model, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)