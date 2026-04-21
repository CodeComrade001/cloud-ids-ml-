from sklearn.model_selection import GridSearchCV

def train(model, params, X_train, y_train):
    if params and len(params) > 0:
        grid = GridSearchCV(model, params, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model