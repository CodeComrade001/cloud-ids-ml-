from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import clone


def train(model, params, X_train, y_train, dataset_name=None):

    # Default settings
    cv_value = 3
    jobs = -1
    active_params = params

    # Heavy dataset optimization
    if dataset_name == "ioT_context":
        cv_value = 2
        jobs = 1

        # Reduce SVM grid only for huge dataset
        if model.__class__.__name__ == "SVC":
            active_params = {
                "C": [1],
                "kernel": ["rbf"]
            }

    # ==========================
    # AUTO FIX FOR TINY CLASSES
    # ==========================
    min_class_count = y_train.value_counts().min()

    # If smallest class has fewer samples than cv folds,
    # reduce folds automatically
    cv_value = min(cv_value, min_class_count)

    # If impossible to do CV, disable grid search
    if cv_value < 2:
        active_params = {}

    # ==========================
    # TRAIN
    # ==========================
    if active_params and len(active_params) > 0:
        grid = GridSearchCV(
            estimator=model,
            param_grid=active_params,
            cv=cv_value,
            n_jobs=jobs
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_

    model.fit(X_train, y_train)
    return model


class BaseModelHandler:
    def train(self, model, X_train, y_train, params, dataset_name):
        return train(model, params, X_train, y_train, dataset_name)


class SVMHandler(BaseModelHandler):
    def train(self, model, X_train, y_train, params, dataset_name):

        if dataset_name == "ioT_context":

            # Stratified sample instead of random sample
            if len(X_train) > 10000:
                X_train, _, y_train, _ = train_test_split(
                    X_train,
                    y_train,
                    train_size=10000,
                    stratify=y_train,
                    random_state=42
                )

            # Remove classes with fewer than 2 rows
            class_counts = y_train.value_counts()
            valid_classes = class_counts[class_counts >= 2].index

            mask = y_train.isin(valid_classes)
            X_train = X_train[mask]
            y_train = y_train[mask]

            # Optional: disable grid for speed
            params = {}

        return super().train(model, X_train, y_train, params, dataset_name)


class MLPHandler(BaseModelHandler):
    def train(self, model, X_train, y_train, params, dataset_name):

        model = clone(model)

        model.set_params(
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            random_state=42
        )

        if dataset_name == "ioT_context":

            if len(X_train) > 30000:
                X_train, _, y_train, _ = train_test_split(
                    X_train,
                    y_train,
                    train_size=30000,
                    stratify=y_train,
                    random_state=42
                )

            model.set_params(max_iter=300)

        else:
            model.set_params(max_iter=400)

        return super().train(model, X_train, y_train, params, dataset_name)

class LRHandler(BaseModelHandler):
    def train(self, model, X_train, y_train, params, dataset_name):

        model = clone(model)

        if dataset_name == "ioT_context":

            model.set_params(
                solver="saga",
                max_iter=1000,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42
            )

            params = {"C": [0.1, 1, 5]}

        else:
            model.set_params(
                max_iter=300
            )

        return super().train(model, X_train, y_train, params, dataset_name)

class GBHandler(BaseModelHandler):
    def train(self, model, X_train, y_train, params, dataset_name):

        model = clone(model)

        if dataset_name == "ioT_context":

            if len(X_train) > 60000:
                X_train, _, y_train, _ = train_test_split(
                    X_train,
                    y_train,
                    train_size=60000,
                    stratify=y_train,
                    random_state=42
                )

            model.set_params(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                random_state=42
            )

            params = {}

        return super().train(model, X_train, y_train, params, dataset_name)

MODEL_HANDLERS = {
    "KNN": BaseModelHandler(),
    "SVM": SVMHandler(),
    "MLP": MLPHandler(),
    "RF": BaseModelHandler(),
    "LR": LRHandler(),
    "DT": BaseModelHandler(),
    "NB": BaseModelHandler(),
    "GB": GBHandler(),
    "VOTE": BaseModelHandler()
}