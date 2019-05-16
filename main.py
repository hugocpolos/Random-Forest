from src.Dataset import Dataset
from src.Forest import Forest
from sys import argv
from src.Bootstrap import Bootstrap
from src.CrossValidation import CrossValidation
import random


def generate_fold_train_set(data, fold_index):
    ret = []

    for i in range(len(data[0])):
        dataset = []
        for j in range(len(data)):
            if(j != fold_index):
                for elem in data[j][i]:
                    dataset.append(elem)
        ret.append(dataset)
    return ret


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
            [--length, -l, -n]  forent-length, the default value is 1
            [--k-fold, -k]      number of folds for cross validation, the
                                default value is 10
            [--print, -p]       print the forest
            [--debug]           debug the program with print

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
        debug = False

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
            elif(argv[i] in ['--debug']):
                debug = True
            elif(argv[i] in ['k-fold', '-k']):
                k_fold_value = int(argv[i + 1])
            elif(argv[i] in ['--help', '-h']):
                print_usage(argv[0])
                exit(0)
        if seed is not None:
            random.seed(int(seed))

        # load the dataset to memory
        db = Dataset(filename, delimiter=delimiter,
                     metadata=metadata)
        if debug:
            print('Db loaded:')
            print("Attributes: %s\nPredict Class: %s\nNumerical Classes: %s" %
                  (db.attributes, db.predictclass, db.numeric))
        # Create k Folds of the dataset
        cv = CrossValidation(db.data)
        cv.generate_folds(k_fold_value)

        if debug:
            print('Generated %d folds Successfully' % (k_fold_value))
            print('Folds:')
            count = 1
            for fold in cv.folds:
                print('%d' % (count))
                count += 1
                print(fold)

        # generate a bootstrap for each fold
        Bootstrap_training_set = []
        for fold in cv.folds:
            bs = Bootstrap(fold)
            bs.generate_bootstrap(n)
            Bootstrap_training_set.append(bs.training_set)

        if debug:
            print('Successfully generated a bootstrap training set for each fold')
            print('Training Sets:')
            count = 1
            for t_set in Bootstrap_training_set:
                print('%d' % (count))
                count += 1
                print(t_set)

        # Create a List of Trained Forests,
        # each forest is trained with k-1 folds,
        # and trained with k=i fold
        Forest_list = []
        for i in range(k_fold_value):
            training_set = generate_fold_train_set(Bootstrap_training_set, i)
            if debug:
                print(
                    'For the fold %d as test set, generated the following training set:' % (i + 1))
                print(training_set)

            test_set = cv.folds[i]
            if debug:
                print()
                print('test set:')
                print(test_set)

            F = Forest(db, n)
            F.train(training_set)

            if debug:
                print('Successfully trained a forest')
                print('Resultant Forest:')
            if to_print:
                F.show()

            if debug:
                print('Testing the forest with the dataset')
                print(test_set)
                print('Test Results:')
            print('%d: ' % (i + 1))
            print(F.test(test_set, debug))
