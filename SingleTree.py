from src.Dataset import Dataset
from src.DecisionTree import DecisionTree
import random
from sys import argv


def print_usage(bin_name):
    print("""Usage:
        %s [Arguments]

        - Arguments:
            [--file, -f]        csv filename
            [--delimiter, -d]   csv delimiter character,
                                the default value is ';'
            [--meta, -m]        json metadata filename
            [--seed, -s]        random for pseudo-random number generation,
                                random value if a seed is not set
        """ % (bin_name))


if __name__ == '__main__':
    if len(argv) == 1:
        print_usage(argv[0])
        exit(0)
    else:
        filename = ""
        delimiter = ';'
        metadata = ""
        seed = None
        for i in range(len(argv)):
            if(argv[i] in ['--file', '-f']):
                filename = argv[i + 1]
            elif(argv[i] in ['--delimiter', '-d']):
                delimiter = argv[i + 1]
            elif(argv[i] in ['--meta', '-m']):
                metadata = argv[i + 1]
            elif(argv[i] in ['--seed', '-s']):
                seed = int(argv[i + 1])
            elif(argv[i] in ['--help', '-h']):
                print_usage(argv[0])
                exit(0)
        if seed is not None:
            random.seed(int(seed))

        # load the dataset to memory
        db = Dataset(filename, delimiter=delimiter, metadata=metadata)

        # generates a decision tree from the dataset
        tree = DecisionTree(db.data, db.attributes,
                            db.target_attribute, db.numeric, single_tree_print=True)

        # print the resultant tree on the terminal
        print(tree)
