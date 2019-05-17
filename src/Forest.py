from src.DecisionTree import DecisionTree
from src.Classifier import Classifier


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
                self.db.target_attribute,
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

        real_class = [i[self.db.target_attribute] for i in dataset]
        predict_class = []
        for entry in dataset:
            predict_class.append(self.classify(entry))

        return(real_class, predict_class)

    def show(self):
        for tree in self.Forest:
            print(tree)

    def close(self):
        del(self)
