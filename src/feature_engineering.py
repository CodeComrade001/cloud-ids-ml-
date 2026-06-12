class FeatureEngineer:
    def __init__(self):
        self.stats_ = {}

    def fit(self, X):
        X = X.copy()

        # Dataset 1 stats
        if "Payload_Size" in X.columns:
            self.stats_["payload_mean"] = X["Payload_Size"].mean()

        # Dataset 2 stats
        if "Duration" in X.columns:
            self.stats_["duration_mean"] = X["Duration"].mean()

        # Dataset 3 stats
        if "login_attempts" in X.columns:
            self.stats_["login_mean"] = X["login_attempts"].mean()

        return self

    def transform(self, X):
        X = X.copy()

        # Dataset 1 features
        if "Payload_Size" in X.columns:
            X["payload_norm"] = X["Payload_Size"] / (self.stats_.get("payload_mean", 1))

        if "Port" in X.columns:
            X["is_high_port"] = (X["Port"] > 1024).astype(int)

        # Dataset 2 features
        if "Duration" in X.columns:
            X["duration_norm"] = X["Duration"] / (self.stats_.get("duration_mean", 1))

        if "Min" in X.columns and "Max" in X.columns:
            X["packet_range"] = X["Max"] - X["Min"]

        if "IAT" in X.columns:
            X["burstiness"] = 1 / (X["IAT"] + 1e-6)

        # Dataset 3 features
        if "login_attempts" in X.columns:
            X["login_norm"] = X["login_attempts"] / (self.stats_.get("login_mean", 1))

        if "failed_logins" in X.columns:
            X["failure_ratio"] = X["failed_logins"] / (X["login_attempts"] + 1e-6)

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)