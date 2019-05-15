class CrossValidation(object):
    """docstring for CrossValidation"""

    def __init__(self, data):
        super(CrossValidation, self).__init__()
        self.data = data
        self.folds = []

    def generate_folds(self, k=10):
        self.folds = split(self.data, k)


def split(a, n):
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
