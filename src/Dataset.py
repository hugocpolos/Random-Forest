import csv
import json
from random import randint


class Dataset(object):
    """
    Creates a Dataset object from a .csv file
    Args:
        filename (str): path to the csv file
        delimiter (str): delimiter character of the csv file,
                         the default value is ';'
        predictclass (str): class to be predicted,
                            if the arg is None, then the
                            predict class will be the one at the last
                            column.
        ignore (list): list of attributes to be ignored during the dataset load

    Attributes:
        attributes (list): List with the dataset attributes
        predictclass (str): The predicted attribute of this dataset
        data (list): List of dicts, each dict refers to one dataset line and
                     have the following format:

        {attribute1:value1,..., attributeN:valueN, predict class:predict value}

        values (list): 2-dimensional list containing just the values
                       of each attribute

        numeric (dict): Stores all numerical categories and its cut value.

    """

    def __init__(self, filename, delimiter=';', predictclass=None, ignore=[],
                 metadata=None):
        super(Dataset, self).__init__()

        # Test the args
        if (type(delimiter) is not str) and (len(delimiter) is not 1):
            raise TypeError("delimiter must be a single character string.")
        if (type(predictclass) is not str) and (predictclass is not None):
            raise TypeError("predictclass must be a string.")
        if (type(ignore) is not list):
            if(type(ignore) is str):
                ignore = [ignore]
            else:
                raise TypeError(
                    "ignore must be a string or a list of strings.")
        if (type(metadata) is not str and metadata is not None):
            raise TypeError("metadata must be a string.")

        # Load the metadata
        self.numeric = {}
        if metadata is not None:
            with open(metadata) as json_file:
                d = json.load(json_file)
                for attrib in d['numeric']:
                    self.numeric[attrib] = 0
                try:
                    ignore += d['ignore']
                except KeyError:
                    ignore = []
                try:
                    predictclass = d['predictclass']
                except KeyError:
                    predictclass = None
        # Reads the csv file to memory
        _dict_ = csv.DictReader(open(filename), delimiter=delimiter)

        # Stores all attributes from the dataset
        self.attributes = _dict_.fieldnames.copy()

        # If predict class wasn't passed as argument, then
        # the default value is the last attribute.
        # If the predict class was passed, it is tested to check if
        # the class exists in the dataset.
        if predictclass is None:
            self.predictclass = self.attributes[-1]
        else:
            self.predictclass = predictclass
            if self.predictclass not in self.attributes:
                raise Exception(
                    "Class '%s' was not found in %s"
                    % (self.predictclass, filename))

        # Removes the predict class of the attribute list
        self.attributes.remove(self.predictclass)

        # Removes the ignored attributes of the attribute list
        for ignored_class in ignore:
            self.attributes.remove(ignored_class)

        # load the data into the dict format
        self.data = [dict(x) for x in _dict_]

        # remove all entries of the ignored classes
        for a in self.data:
            for ignored_class in ignore:
                del a[ignored_class]

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

    def close(self):
        del(self)
