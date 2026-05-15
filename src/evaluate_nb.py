from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


def evaluate_and_save(model, X_test, y_test, *, out_dir: Path, labels_order=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    # report
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(out_dir / "classification_report.csv", index=True, encoding="utf-8-sig")

    # labels otomatis (FIX utama)
    labels_order = sorted(list(set(list(y_test) + list(y_pred))))
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    # plot
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(labels_order)), labels_order, rotation=45, ha="right")
    plt.yticks(range(len(labels_order)), labels_order)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_test": int(len(y_test)),
        "labels_order": labels_order,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics