from src.Dataset import Dataset
from src.Forest import Forest
from sys import argv
from src.Bootstrap import Bootstrap
from src.CrossValidation import CrossValidation
import src.Metrics as Metrics
import src.ConfusionMatrix as ConfusionMatrix
import random


def generate_train_set_from_folds_except_n(data, n, forest_lenght):
    ret = []

    for i in range(len(data)):
        if i != n:
            for sample in data[i]:
                ret.append(sample)

    bs = Bootstrap(ret)
    bs.generate_bootstrap(forest_lenght)
    return bs.training_set


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
                  (db.attributes, db.target_attribute, db.numeric))

        # Create k Folds of the dataset
        cv = CrossValidation(db.data, db.target_attribute)
        cv.generate_stratified_folds(k_fold_value)

        if debug:
            print('Generated %d folds Successfully' % (k_fold_value))
            print('Folds:')
            count = 1
            for fold in cv.folds:
                print('%d' % (count))
                count += 1
                print(fold)

        Forest_list = []
        for i in range(k_fold_value):
            training_set = generate_train_set_from_folds_except_n(cv.folds, n=i, forest_lenght=n)
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

            # F1-measure:
            (real_labels, pred_labels) = F.test(test_set, debug)
            confusion_matrix = ConfusionMatrix.create_confusion_matrix(
                real_labels, pred_labels, db.get_target_attribute_values())
            f1 = Metrics.f1measure(confusion_matrix)

            print('Confusion Matrix')
            print(confusion_matrix)
            print('F1-measure:', f1)
