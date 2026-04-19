from sklearn.decomposition import PCA

def apply_pca(X_train, X_test):
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test