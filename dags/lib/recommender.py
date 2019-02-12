import pickle

from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score


class Recommender:
    def __init__(self, learning_rate=0.05, loss='warp'):
        self.model = LightFM(learning_rate=learning_rate, loss=loss)

    def fit(self, interactions, epochs):
        self.model.fit_partial(interactions=interactions, epochs=epochs)

    def evaluate_at_k(self, test_interactions, k):
        return precision_at_k(self.model, test_interactions, k=k).mean()

    def evaluate_auc(self, test_interactions):
        return auc_score(self.model, test_interactions).mean()

    def dump_model(self, path):
        with open(path, mode='wb') as file:
            pickle.dump(self, file)