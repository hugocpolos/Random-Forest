from src.Dataset import Dataset
from src.DecisionTree import DecisionTree
from src.Forest import Forest
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
        db = Dataset(argv[1], delimiter=delimiter,
                     metadata='data/buysComputer_meta.json')

        for n in range(1, 101):
            # Load the Forest Object
            F = Forest(db, forest_length=n, seed=None)

            # Train the Forest
            F.train()
            # F.print()
            print("n = %s, hit ratio :" % (n), end=' ')

            # Test the Forest
            F.test()

        exit(0)
