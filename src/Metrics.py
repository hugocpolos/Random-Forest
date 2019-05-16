import numpy as np
import src.ConfusionMatrix as ConfusionMatrix
from src.Forest import Forest

def true_positives(confusion_matrix, class_index):
    return confusion_matrix[class_index, class_index]

def false_positives(confusion_matrix, class_index):
    return sum(confusion_matrix[:, class_index])-confusion_matrix[class_index, class_index]

def false_negatives(confusion_matrix, class_index):
    return sum(confusion_matrix[class_index, :])-confusion_matrix[class_index, class_index]

def recall(tp, fn):
    """Mesures the completeness of the model.
     tp : int 
        true positives
     fn : int
        false negatives
    """

    # print('recall(tp: ' + str(tp) + ', fn: ' + str(fn) + ')')

    return tp/(tp+fn)

def precision(tp, fp):
    """Mesures the precision of the model.
    tp : int 
        true positives
    fp : int
        false positives
    """

    # print('precision(tp: ' + str(tp) + ', fp: ' + str(fp) + ')')

    return tp/(tp+fp)

def f1measure(confusion_matrix, beta = 1):
    """Measures the F1-Score/Measure given a confusion matrix.
    confusion_matrix :  matrix that exposes predicted classes and the respective real classes
                        the method create_confusion_matrix(...) may be useful
    beta :  weights the influence between recall and precision in the result

    return: the F1-Score/Mesure 
    """

    number_of_classes = confusion_matrix.shape[0]

    recall_acu      = 0.
    precision_acu   = 0.

    for class_index in range(number_of_classes):
        tp = true_positives(confusion_matrix, class_index)
        fp = false_positives(confusion_matrix, class_index)
        fn = false_negatives(confusion_matrix, class_index)
        recall_acu      += recall(tp, fn)
        precision_acu   += precision(tp, fp)

    macro_recall      = recall_acu/number_of_classes
    macro_precision   = precision_acu/number_of_classes

    # print('macro_recall', macro_recall)
    # print('macro_precision', macro_precision)
    
    if beta == 1:
        return 2*(macro_precision * macro_recall) / (macro_precision + macro_recall)
    else:
        return (1+beta**2)*(macro_precision*macro_recall)/(((beta**2)*macro_precision) + macro_recall)

    ######################################
    # TESTS AGAINST WELL-KNOWN SOLUTIONS #
    ######################################

if __name__ == '__main__':

    from sklearn import metrics

    real_classes = ['c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1',
                        'c2', 'c2', 'c2', 'c2', 'c2',
                        'c3', 'c3', 'c3', 'c3', 'c3', 'c3', 'c3', 'c3', 'c3', 'c3']

    pred_classes_array = ['c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c2', 'c3', 'c3', 'c3',
                        'c2', 'c2', 'c2', 'c2', 'c1',
                        'c3', 'c3', 'c3', 'c3', 'c3', 'c3', 'c1', 'c1', 'c2', 'c2']

    classes = ['c1', 'c2', 'c3']

    confusion_matrix = ConfusionMatrix.create_confusion_matrix(real_classes, pred_classes_array, classes)
    sklean_confusion_matrix = metrics.confusion_matrix(real_classes, pred_classes_array, classes)

    print(confusion_matrix)
    print(sklean_confusion_matrix)

    f1measure = f1measure(confusion_matrix)
    sklearn_f1measure = metrics.precision_recall_fscore_support(real_classes, pred_classes_array, average='macro')

    print('f1measure\t', f1measure)
    print('sk fb_score\t', sklearn_f1measure[2])
