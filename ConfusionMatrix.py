import numpy as np

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
    return tp/(tp+fn)

def precision(tp, fp):
    """Mesures the precision of the model.
    tp : int 
        true positives
    fp : int
        false positives
    """

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

def create_confusion_matrix(real_classes, pred_classes, classes):
    """Creates a confusion matrix based on prediction and the respective predictions.
    real_classes : array containing the real classes (the order of the instances must match the predicted classes array)
    pred_classes : array containing the predicted classes (the order of the instances must match the real classes array)
    classes : list of the different classification values; exemple: ['high', 'average', 'low']
    """

    number_of_classes   = len(classes)
    number_of_instances = len(real_classes)

    confusion_matrix = np.zeros((number_of_classes, number_of_classes), np.uint)

    for instance_number in range(number_of_instances):

        true_class = real_classes[instance_number]
        pred_class = pred_classes_array[instance_number]

        true_index = classes.index(true_class)
        pred_index = classes.index(pred_class)

        # true_class_str = true_class + "(" + str(true_index) + ")"
        # pred_class_str = pred_class + "(" + str(pred_index) + ")"
        # print(str(instance_number) + " " + true_class_str + " " + pred_class_str)
        
        confusion_matrix[true_index, pred_index] += 1

    return confusion_matrix

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

    confusion_matrix = create_confusion_matrix(real_classes, pred_classes_array, classes)
    sklean_confusion_matrix = metrics.confusion_matrix(real_classes, pred_classes_array, classes)

    print(confusion_matrix)
    print(sklean_confusion_matrix)

    f1measure = f1measure(confusion_matrix)
    sklearn_f1measure = metrics.precision_recall_fscore_support(real_classes, pred_classes_array, average='macro')

    print('f1measure', f1measure)
    print('sk fb_score', sklearn_f1measure[2])
