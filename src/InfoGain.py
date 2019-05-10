import math


def best_info_gain(D, L, predict_class):
    dataset_info = __Dataset_Info(D, predict_class)
    gain = {}
    for attribute in L:
        gain[attribute] = dataset_info - __Info_class(D, attribute, predict_class)
    # print(gain)
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
