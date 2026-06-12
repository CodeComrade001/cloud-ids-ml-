# =========================
# 5) MODIFY train() FUNCTION (trainer.py)
# REPLACE ENTIRE FILE
# =========================

from sklearn.model_selection import GridSearchCV

def train(model, params, X_train, y_train, dataset_name=None):

    if dataset_name == "ioT_context":
        cv_value = 2
        jobs = 1
    else:
        cv_value = 3
        jobs = -1

    if params and len(params) > 0:
        grid = GridSearchCV(
            model,
            params,
            cv=cv_value,
            n_jobs=jobs
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_

    model.fit(X_train, y_train)
    return model