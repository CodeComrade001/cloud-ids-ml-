from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.metrics import roc_auc_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    result = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": None
    }

    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            result["roc_auc"] = roc_auc_score(y_test, y_score, multi_class="ovr")

        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            result["roc_auc"] = roc_auc_score(y_test, y_score, multi_class="ovr")

    except:
        result["roc_auc"] = None

    return result