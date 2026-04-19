from sklearn.model_selection import GridSearchCV

def train(model, params, X_train, y_train):
    grid = GridSearchCV(model, params, cv=10, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_