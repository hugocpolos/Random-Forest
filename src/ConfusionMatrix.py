import numpy as np

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
        pred_class = pred_classes[instance_number]

        true_index = classes.index(true_class)
        pred_index = classes.index(pred_class)

        # true_class_str = true_class + "(" + str(true_index) + ")"
        # pred_class_str = pred_class + "(" + str(pred_index) + ")"
        # print(str(instance_number) + " " + true_class_str + " " + pred_class_str)
        
        confusion_matrix[true_index, pred_index] += 1

    return confusion_matrix