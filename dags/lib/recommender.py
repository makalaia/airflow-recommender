import pickle

from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from .utils import create_dir


class Recommender:
    def __init__(self, learning_rate=0.05, loss='bpr'):
        self.model = LightFM(learning_rate=learning_rate, loss=loss)

    def fit(self, interactions, epochs):
        self.model.fit(interactions=interactions, epochs=epochs)

    def fit_partial(self, interactions, epochs):
        self.model.fit_partial(interactions=interactions, epochs=epochs)

    def fit_until_decay(self, interactions, val_interactions, max_epochs, patience=1):
        max_auc = 0
        assert max_epochs > 0

        for i in range(max_epochs):
            self.model.fit_partial(interactions=interactions, epochs=1)
            auc = self.evaluate_auc(val_interactions)
            if auc > max_auc:
                max_auc = auc
            elif patience == 0:
                break
            else:
                patience -= 1
        return i, max_auc

    def evaluate_at_k(self, test_interactions, k):
        return precision_at_k(self.model, test_interactions, k=k).mean()

    def evaluate_auc(self, test_interactions):
        return auc_score(self.model, test_interactions).mean()

    def dump_model(self, path):
        create_dir(path)
        with open(path, mode='wb') as file:
            pickle.dump(self, file)
