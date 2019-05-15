from src.DecisionTree import DecisionTree
from src.Classifier import Classifier
import random


class Forest(object):
    """docstring for Forest"""

    def __init__(self, dataset_object, forest_length=1):
        super(Forest, self).__init__()
        self.db = dataset_object
        self.T = forest_length
        self.Forest = []
        self.evaluation = {}

    def train(self, training_set):
        for train_set in training_set:
            self.Forest.append(DecisionTree(
                train_set,
                self.db.attributes,
                self.db.predictclass,
                self.db.numeric))

    def classify(self, entry):
        predictions = {}

        # Fill a dict with all the predictions and its scores.
        for tree in self.Forest:
            try:
                predictions[Classifier(entry, tree)] += 1
            except KeyError:
                predictions[Classifier(entry, tree)] = 1
        # Get the most voted prediction
        max = 0
        max_prediction = None
        for p in predictions:
            if(predictions[p] > max):
                max = predictions[p]
                max_prediction = p
        return max_prediction

    def test(self, dataset, debug=False):
        # Test the Forest with a given dataset

        hit_fail_matrix = {
            'hit': 0,
            'fail': 0
        }

        for entry in dataset:
            if self.classify(entry) == entry[self.db.predictclass]:
                hit_fail_matrix['hit'] += 1
            else:
                hit_fail_matrix['fail'] += 1
        if debug:
            print(100*hit_fail_matrix['hit'] / (hit_fail_matrix['hit'] + hit_fail_matrix['fail']), end=' ')
            print(hit_fail_matrix)

    def show(self):
        for tree in self.Forest:
            print(tree)

    def close(self):
        del(self)
