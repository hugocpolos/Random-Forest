from src.Dataset import Dataset
from src.Forest import Forest
from sys import argv
from src.Bootstrap import Bootstrap
from src.CrossValidation import CrossValidation
import src.Metrics as Metrics
import src.ConfusionMatrix as ConfusionMatrix
import src.Plot as Plot
import random
import numpy as np


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
            [--ntree, -n]       forent-length, the default value is 1
            [--ntree2 -n2]      varies ntree from --ntree to --ntree2 and plot the f1measure(ments) to f1measure.png (ntree analysis)
            [--k-fold, -k]      number of folds for cross validation, the
                                default value is 10
            [--print, -p]       print the forest
            [--debug]           debug the program with print

        """ % (bin_name))

def load_dataset_into_memory():
    # load the dataset into memory
    db = Dataset(filename, delimiter=delimiter,
                    metadata=metadata)
    if debug:
        print('Db loaded:')
        print("Attributes: %s\nPredict Class: %s\nNumerical Classes: %s" %
                (db.attributes, db.predictclass, db.numeric))
    return db

def create_k_folds_from_dataset(k, dataset):
    # Create k Folds of the dataset
    cv = CrossValidation(dataset.data, dataset.predictclass)
    cv.generate_stratified_folds(k)

    if debug:
        print('Generated %d folds Successfully' % (k))
        print('Folds:')
        count = 1
        for fold in cv.folds:
            print('%d' % (count))
            count += 1
            print(fold)
    
    return cv

def train_k_forests(k, cv, database, ntree):
    forest_list = []
    for i in range(k):
        training_set = generate_train_set_from_folds_except_n(cv.folds, n=i, forest_lenght=ntree)
        if debug:
            print(
                'For the fold %d as test set, generated the following training set:' % (i + 1))
            print(training_set)

        F = Forest(database, ntree)
        F.train(training_set)

        if debug:
            print('Successfully trained a forest')
            print('Resultant Forest:')
        if to_print:
            F.show()
        
        forest_list.append(F)
    
    return forest_list

def test_forest_list(forest_list, cv, database):

    classes = db.get_target_attribute_values()
    f1measure = 0

    for i, forest in enumerate(forest_list):
        test_set = cv.folds[i]

        (real_labels, pred_labels) = forest.test(test_set, debug)
        confusion_matrix = ConfusionMatrix.create_confusion_matrix(real_labels, pred_labels, classes)
        f1measure += Metrics.f1measure(confusion_matrix)

        if debug:
            print('Testing the forest with the dataset:')
            print(test_set)
            print('Test Results:')

            print('%d: ' % (i + 1))
            print(confusion_matrix)
    
    f1measure /= len(forest_list)
    if debug:
        print('Mean F1-Measure: ', f1measure)

    return f1measure

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
        n2 = 0
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
            elif(argv[i] in ['-n2']):
                n2 = int(argv[i + 1])
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

        db = load_dataset_into_memory()
        cv = create_k_folds_from_dataset(k_fold_value, db)

        if n2 > 0:  # then ntree analysis
                    # ntree varies from -n to -n2 (argument values)
        
            f1_mean_list = []

            for i in range(n, n2+1):
                
                forest_list = train_k_forests(k_fold_value, cv, db, i)
                f1_mean = test_forest_list(forest_list, cv, db)        
                f1_mean_list.append(f1_mean)

            Plot.plot_to_png(range(n, n2+1), f1_mean_list, 'f1measure')

        else:       # then just train and test a single forest
            
            forest_list = train_k_forests(k_fold_value, cv, db, n)
            f1_mean = test_forest_list(forest_list, cv, db)
            print(f1_mean)
