from sklearn.cluster import KMeans
import joblib

class Segmenter:
    def __init__(self, n_clusters=5):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def fit_predict(self, X):
        return self.model.fit_predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

