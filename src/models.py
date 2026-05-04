from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

def get_model(name):
    models = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "MLP": MLPClassifier(),
        "RF": RandomForestClassifier(),
        "LR": LogisticRegression(max_iter=300),
        "DT": DecisionTreeClassifier(),
        "NB": GaussianNB(),
        "GB": GradientBoostingClassifier(),

        "VOTE": VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier()),
                ("svm", SVC(probability=True)),
                ("lr", LogisticRegression(max_iter=300))
            ],
            voting="soft"
        )
    }

    return models[name]

PARAM_GRIDS = {
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    },

    "SVM": {
        "C": [1, 10],
        "kernel": ["linear", "rbf"]
    },

    "MLP": {
        "hidden_layer_sizes": [(50,), (100,)],
        "activation": ["relu"],
        "max_iter": [200]
    },

    "RF": {
        "n_estimators": [50, 100],
        "max_depth": [None, 10]
    },
    
    "LR": {
        "C": [1]
    },

    "DT": {
        "max_depth": [10, None]
    },

    "NB": {},

    "GB": {
        "n_estimators": [50]
    },

    "VOTE": {}
}