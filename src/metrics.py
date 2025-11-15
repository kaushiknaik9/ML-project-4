import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_thresholds(model, X_test, y_test, save_dir="plots"):
    """
    Computes precision, recall, and F1 for thresholds from 0 to 1.
    Saves plots into /plots directory.
    """

    thresholds = np.arange(0.0, 1.01, 0.01)
    precision_vals, recall_vals, f1_vals = [], [], []

    probs = model.predict_proba(X_test)[:, 1]

    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision_vals.append(precision_score(y_test, preds, zero_division=0))
        recall_vals.append(recall_score(y_test, preds))
        f1_vals.append(f1_score(y_test, preds))

    # PLOTS
    plt.figure()
    plt.plot(thresholds, precision_vals)
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title("Threshold vs Precision")
    plt.savefig(f"{save_dir}/threshold_vs_precision.png")
    plt.close()

    plt.figure()
    plt.plot(thresholds, recall_vals)
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title("Threshold vs Recall")
    plt.savefig(f"{save_dir}/threshold_vs_recall.png")
    plt.close()

    plt.figure()
    plt.plot(thresholds, f1_vals)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold vs F1 Score")
    plt.savefig(f"{save_dir}/threshold_vs_f1.png")
    plt.close()
    
    return thresholds, precision_vals, recall_vals, f1_vals
