from src.Dataset import Dataset
from src.DecisionTree import DecisionTree
from sys import argv
from src.Classifier import Classifier


def print_usage(bin_name):
    print("""Usage:
        %s dataset_filename [delimiter_char]

        dataset_filename:
            path to the dataset csv file.
        delimiter_char: optional
            character that delimites the dataset.
            The default value is ';'

        ex:
            %s data.csv ;
            %s relative/path/data.csv
            %s /absolute/path/data.csv ,
        """ % (bin_name, bin_name, bin_name, bin_name))


if __name__ == '__main__':
    if len(argv) not in [2, 3]:
        print_usage(argv[0])
        exit(0)
    else:
        delimiter = argv[2] if len(argv) is 3 else ';'

        # load the dataset to memory
        tree_forest = []
        n = 1000
        db = Dataset(argv[1], delimiter=delimiter,
                     metadata='data/buysComputer_meta.json', bootstrap_n=n)

        for training_set in db.training_set:
            tree_forest.append(DecisionTree(
                training_set, db.attributes, db.predictclass, db.numeric))

        # for tree in tree_forest:
        #     print(tree)
        print()
        print('Success: %d tree(s) have been trained' % (n))
        print('prediction for the following entry:')
        print(db.test_set[0][0])
        print(db.predictclass + ':')
        prediction = {}
        for tree in tree_forest:
            try:
                prediction[Classifier(db.test_set[0][0], tree)] += 1
            except KeyError:
                prediction[Classifier(db.test_set[0][0], tree)] = 1
        print(prediction)
