from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def get_model(name):
    models = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "MLP": MLPClassifier(),
        "RF": RandomForestClassifier()
    }
    return models[name]

PARAM_GRIDS = {
    "KNN": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"]
    },

    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },

    "MLP": {
        "hidden_layer_sizes": [(50,), (100,)],
        "activation": ["relu", "tanh"],
        "max_iter": [200, 300]
    },

    "RF": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20]
    }
}