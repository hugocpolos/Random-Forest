import math
import copy


def best_info_gain(D, L, predict_class, numerical_attributes, single_tree_print=False):
    # Creates a copy of the dataset to be handled
    D_copy = copy.deepcopy(D)

    # Split the numerical values beetwen A-> less than the cut value
    # and B -> greater or equal to the cut value to make the best info gain
    # calculation
    if (single_tree_print):
        print('For the dataset:')
        for d in D_copy:
            print(d)

    for attrib in L:
        if attrib in numerical_attributes:
            avg_val = numerical_attributes[attrib]
            for entry in D_copy:
                try:
                    entry[attrib] = 'A' if float(
                        entry[attrib]) < avg_val else 'B'
                except Exception as e:
                    pass

    # Calculate the information gain of the dataset
    dataset_info = __Dataset_Info(D_copy, predict_class)

    # Calculates the information gain of each attribute
    # and stores in a dict.
    gain = {}
    for attribute in L:
        gain[attribute] = dataset_info - \
            __Info_class(D_copy, attribute, predict_class)
    if (single_tree_print):
        print(gain)

    # Return the attribute with the maximum info_gain
    v = list(gain.values())
    k = list(gain.keys())
    return k[v.index(max(v))]


def __Dataset_Info(D, predict_class):
    total = len(D)
    info = 0
    counter = {}
    for i in range(len(D)):
        counter[D[i][predict_class]] = 0
    for i in range(len(D)):
        counter[D[i][predict_class]] += 1

    for key, value in counter.items():
        info -= (value / total) * math.log2(value / total)

    return info


def __Info_class(D, attribute, predict_class):
    counter = {}
    total = len(D)
    info = 0

    for i in range(len(D)):
        counter[D[i][attribute]] = 0

    for i in range(len(D)):
        counter[D[i][attribute]] += 1

    for key, val in counter.items():
        sub_D = [x for x in D if (x[attribute] == key)]
        info += ((val / total) * __Dataset_Info(sub_D, predict_class))

    return info
