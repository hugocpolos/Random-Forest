import random


class CrossValidation(object):
    """docstring for CrossValidation"""

    def __init__(self, data, predictclass):
        super(CrossValidation, self).__init__()
        self.data = data
        self.predictclass = predictclass
        self.folds = []
        self.possible_classes = list(dict.fromkeys(
            [d[predictclass] for d in data]))
        self.generate_data_for_each_class()
        self.count_classes()

    def generate_data_for_each_class(self):
        self.data_per_class = {}
        for class_ in self.possible_classes:
            self.data_per_class[class_] = [
                c for c in self.data if c[self.predictclass] == class_]

    def count_classes(self):
        class_counter = {}

        for d in self.data:
            try:
                class_counter[d[self.predictclass]] += 1
            except KeyError:
                class_counter[d[self.predictclass]] = 1

        self.class_counter = class_counter

    def generate_stratified_folds(self, k=10):

        self.calculate_proportion(k)

        for i in range(k):
            new_fold = []

            for class_ in self.class_counter:
                for j in range(self.proportion[class_]['each_fold']):
                    new_sample = random.choice(self.data_per_class[class_])
                    self.data_per_class[class_].remove(new_sample)
                    new_fold.append(new_sample)
            self.folds.append(new_fold)

        # randomly choice the extra samples
        while(len(self.extra_samples) > 0):
            for fold in self.folds:
                try:
                    new_sample = random.choice(self.extra_samples)
                    self.extra_samples.remove(new_sample)
                    fold.append(new_sample)
                except IndexError:
                    pass

    def calculate_proportion(self, k):
        # Calculate the classes proportion for a given k
        proportion = {}
        for class_ in self.class_counter:
            div, rest = divmod(self.class_counter[class_], k)
            proportion[class_] = {}
            proportion[class_]['each_fold'] = div
            proportion[class_]['extra'] = rest

        self.proportion = proportion
        self.create_extra_samples()

    def create_extra_samples(self):
        extra_samples = []
        for class_ in self.class_counter:
            for j in range(self.proportion[class_]['extra']):
                new_sample = random.choice(self.data_per_class[class_])
                self.data_per_class[class_].remove(new_sample)
                extra_samples.append(new_sample)
        self.extra_samples = extra_samples


def split(a, n):
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == '__main__':
    data = [{'i': '1', '.': 'a'}, {'i': '2', '.': 'a'},
            {'i': '3', '.': 'a'}, {'i': '4', '.': 'b'}, {'i': '5', '.': 'b'},
            {'i': '6', '.': 'b'}, {'i': '7', '.': 'c'}, {'i': '8', '.': 'c'},
            {'i': '9', '.': 'c'}, {'i': '10', '.': 'd'}]
    possible_classes = ['a', 'b', 'c', 'd']
    c = CrossValidation(data, '.')
    c.generate_stratified_folds(3)
    print('Folds:')
    for set_ in c.folds:
        print (set_)
