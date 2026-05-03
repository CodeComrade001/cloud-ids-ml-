from sklearn.decomposition import PCA

def apply_pca(X_train, X_val, X_test):
    pca = PCA(n_components=0.95)

    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_val_pca, X_test_pca