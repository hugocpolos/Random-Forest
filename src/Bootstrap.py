from random import randint


class Bootstrap(object):
    """docstring for Bootstrap"""

    def __init__(self, data):
        super(Bootstrap, self).__init__()
        self.data = data
        self.training_set = []
        self.test_set = []

    def generate_bootstrap(self, n):
        dataset_len = len(self.data)
        for index in range(n):
            train_lottery = [randint(0, dataset_len - 1)
                             for i in range(dataset_len)]

            test_lottery = [i for i in range(
                dataset_len) if i not in train_lottery]

            train_set = [self.data[i] for i in train_lottery]
            test_set = [self.data[i] for i in test_lottery]
            self.training_set.append(train_set)
            self.test_set.append(test_set)
