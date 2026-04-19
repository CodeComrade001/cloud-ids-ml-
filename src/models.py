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