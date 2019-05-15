from src.Dataset import Dataset
from src.DecisionTree import DecisionTree
from src.Forest import Forest
from sys import argv
from src.Bootstrap import Bootstrap
from src.CrossValidation import CrossValidation


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
            [--length, -l, -m]  forent-length, the default value is 1
            [--k-fold, -k]      number of folds for cross validation, the
                                default value is 10
            [--print, -p]       print the forest

        """ % (bin_name))


if __name__ == '__main__':
    if len(argv) is 1:
        print_usage(argv[0])
        exit(0)
    else:
        filename = ""
        delimiter = ';'
        metadata = ""
        seed = None
        n = 1
        to_print = False
        k_fold_value = 10

        for i in range(len(argv)):
            if(argv[i] in ['--file', '-f']):
                filename = argv[i + 1]
            elif(argv[i] in ['--delimiter', '-d']):
                delimiter = argv[i + 1]
            elif(argv[i] in ['--meta', '-m']):
                metadata = argv[i + 1]
            elif(argv[i] in ['--seed', '-s']):
                seed = int(argv[i + 1])
            elif(argv[i] in ['--length', '-l', '-n']):
                n = int(argv[i + 1])
            elif(argv[i] in ['--print', '-p']):
                to_print = True
            elif(argv[i] in ['k-fold', '-k']):
                k_fold_value = int(argv[i + 1])
            elif(argv[i] in ['--help', '-h']):
                print_usage(argv[0])
                exit(0)

        # load the dataset to memory
        db = Dataset(filename, delimiter=delimiter,
                     metadata=metadata)

        # Create k Folds of the dataset
        cv = CrossValidation(db.data)
        cv.generate_folds(k_fold_value)

        # generate a bootstrap for each fold
        Bootstrap_training_set = []
        for fold in cv.folds:
            bs = Bootstrap(fold)
            bs.generate_bootstrap(n)
            Bootstrap_training_set.append(bs.training_set)

        Forest_list = []
        for i in range(k_fold_value):
            F = Forest(db, n, seed)
            F.train(Bootstrap_training_set[i])
            F.show()
