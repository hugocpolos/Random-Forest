class CrossValidation(object):
    """docstring for CrossValidation"""

    def __init__(self, data, predictclass):
        super(CrossValidation, self).__init__()
        self.data = data
        self.predictclass = predictclass
        self.folds = []
        self.possible_classes = list(dict.fromkeys(
            [d[predictclass] for d in data]))
        self.data_per_class = {}
        for class_ in self.possible_classes:
            self.data_per_class[class_] = [
                c for c in self.data if c[predictclass] == class_]

    def generate_stratified_folds(self, k=10):
        # for fold in k:
        _temp_data_ = []
        max_len = 0
        for d in self.data_per_class:
            max_len = (len(self.data_per_class[d])) if (
                len(self.data_per_class[d])) > max_len else max_len

        for i in range(max_len):
            for d in self.data_per_class:
                try:
                    n = self.data_per_class[d][i]
                    _temp_data_.append(n)
                except:
                    pass

        self.folds = split(_temp_data_, k)


def split(a, n):
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]