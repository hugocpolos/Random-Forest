import csv


class Dataset(object):
    """
    Creates a Dataset object from a .csv file
    Args:
        filename (str): path to the csv file
        delimiter (str): delimiter character of the csv file,
                         the default value is ';'

    Attributes:
        attributes (list): List with the dataset attributes
        predictclass (str): The predicted attribute of this dataset
        data (list): List of dicts, each dict refers to one dataset line and
                     have the following format:

        {attribute1:value1,..., attributeN:valueN, predict class:predict value}

        values (list): 2-dimensional list containing just the values
                       of each attribute
        error (exception class): is None if the construction was successful
                                otherwise stores the Exception that crashed it.
    """

    def __init__(self, filename, delimiter=';'):
        super(Dataset, self).__init__()
        self.error = None
        try:
            _dict_ = csv.DictReader(open(filename), delimiter=delimiter)
            self.attributes = _dict_.fieldnames.copy()
            self.predictclass = self.attributes[-1]
            self.attributes.remove(self.predictclass)
            self.data = [dict(x) for x in _dict_]
            self.values = [list(x.values()) for x in self.data]
        except Exception as e:
            print(type(e))
            self.error = e

    def __str__(self):
        """
            Method to print the Dataset object into the terminal
            The execution of the function print(dataset_obj) is this method.

            The utf-8 codes are working fine at the ubuntu terminal.
            Although it can be removed to solve compatibility issues
        """
        header = '\033[95m'
        for attrib in self.attributes:
            header += "%-15s" % (attrib)
        header += '\033[94m%-15s' % (self.predictclass)
        header += '\033[0m'
        body = "\n"
        for row in self.values:
            for index in range(len(row)):
                if index == (len(row) - 1):
                    body += '\033[92m'
                body += "%-15s" % (row[index])
            body += '\033[0m\n'
        return header + body
