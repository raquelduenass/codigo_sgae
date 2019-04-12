from sklearn import metrics
from collections import Counter


def show_metrics(real, predicted, soft):  # , probabilities):
    """
    Printing of the main metrics to check the performance of the system
    :param real: ground truth
    :param predicted: predictions from the CNN and silence
    :param soft: softened predictions
    """
    before_ave_accuracy, before_precision, before_recall, before_f_score = [], [], [], []
    after_ave_accuracy, after_precision, after_recall, after_f_score = [], [], [], []

    for i in range(len(real)):
        real[i] = real[i][:len(predicted[i])]

        # Accuracy before softening
        before_ave_accuracy.append(metrics.accuracy_score(real[i], predicted[i]))
        before_precision.append(metrics.precision_score(real[i], predicted[i], average='weighted'))
        before_recall.append(metrics.recall_score(real[i], predicted[i], average='weighted'))
        before_f_score.append(metrics.f1_score(real[i], predicted[i], average='weighted'))
        # metrics.precision_recall_curve(real, probabilities)

        # Accuracy after softening
        after_ave_accuracy.append(metrics.accuracy_score(real[i], soft[i]))
        after_precision.append(metrics.precision_score(real[i], soft[i], average='weighted'))
        after_recall.append(metrics.recall_score(real[i], soft[i], average='weighted'))
        after_f_score.append(metrics.f1_score(real[i], soft[i], average='weighted'))

    before_ave_accuracy = sum(before_ave_accuracy)/len(real)
    before_precision = sum(before_precision) / len(real)
    before_recall = sum(before_recall) / len(real)
    before_f_score = sum(before_f_score) / len(real)
    after_ave_accuracy = sum(after_ave_accuracy) / len(real)
    after_precision = sum(after_precision) / len(real)
    after_recall = sum(after_recall) / len(real)
    after_f_score = sum(after_f_score) / len(real)

    print('Average accuracy before softening= ', before_ave_accuracy)
    print('Precision before softening= ', before_precision)
    print('Recall before softening= ', before_recall)
    print('F-score before softening= ', before_f_score)
    print('Average accuracy after softening= ', after_ave_accuracy)
    print('Precision after softening= ', after_precision)
    print('Recall after softening= ', after_recall)
    print('F-score after softening= ', after_f_score)
    return


# TODO: Introducir probabilidades de CNN
def soft_max_prob(predicted, wind_len, param, files, probabilities):
    """

    :param predicted:
    :param wind_len:
    :param param:
    :param files:
    :param probabilities:
    :return:
    """
    ret_soft = [[]] * files

    for j in range(files):
        if not (param == 0):
            join = []
            prob_over = []
            for i in range(len(predicted[j])):
                if i < param:
                    join.append(predicted[j][i])
                    prob_over.append(probabilities[j][i])
                else:
                    sum_prob = 0
                    for k in range(int(i-param), int(i)):
                        sum_prob += probabilities[j][k]
                    prob_over.append(sum_prob)
                    join.append(predicted[j][prob_over[i].index(max(prob_over[i]))])
        else:
            join = predicted[j]
            prob_over = probabilities

        soft = []
        for i in range(len(join)):
            if i < wind_len // 2:
                soft.append(Counter(join[0:i + wind_len // 2]).most_common(1)[0][0])
            elif i > len(join) - 1 - wind_len // 2:
                soft.append(Counter(join[i - wind_len // 2:len(join) - 1]).most_common(1)[0][0])
            else:
                soft.append(Counter(join[i - wind_len // 2:i + wind_len // 2]).most_common(1)[0][0])
        ret_soft[j] = soft
    return ret_soft


def soft_max(predicted, wind_len, param, files):
    """

    :param predicted:
    :param wind_len:
    :param param:
    :param files:
    :return:
    """
    ret_soft = [[]]*files

    for j in range(files):
        if not(param == 0):
            join = []
            for i in range(len(predicted[j])):
                if i < param:
                    join.append(predicted[j][i])
                else:
                    join.append(Counter(predicted[j][int(i-param):int(i)]).most_common(1)[0][0])
        else:
            join = predicted[j]

        soft = []
        for i in range(len(join)):
            if i < wind_len//2:
                soft.append(Counter(join[0:i+wind_len//2]).most_common(1)[0][0])
            elif i > len(join)-1-wind_len//2:
                soft.append(Counter(join[i-wind_len//2:len(join)-1]).most_common(1)[0][0])
            else:
                soft.append(Counter(join[i-wind_len//2:i+wind_len//2]).most_common(1)[0][0])
        ret_soft[j] = soft
    return ret_soft


def counting(data, label):
    """
    Identifies the position and duration of the appearances of a determined class
    in a sequence of labels
    :param data: sequence of labels
    :param label: name of class to identify
    :return pos: list of the offset of every appearance of the class
    :return length: list of durations of the appearances of the class
    """
    loc = [i for i in range(len(data)) if data[i] == label]
    if not(len(loc) == 0):
        pos = [loc[0]] + [loc[i+1] for i in range(len(loc)-1) if (not (loc[i]+1 == loc[i+1]))]
        fin = [loc[i] for i in range(len(loc)-1) if (not (loc[i]+1 == loc[i+1]))]
        fin.append(loc[-1])
        length = [fin[i]-pos[i]+1 for i in range(len(pos))]
    else:
        pos = []
        length = []
    return pos, length


def join_labels(predicted, silence):
    """
    Merge of the silence labels and the predicted ones from the CNN
    :param predicted: predictions from the CNN
    :param silence: silence detections
    :return: silence: merge of the inputs
    """
    j = 0
    for i in range(len(predicted)):
        while silence[j] != '':
            j = j+1
        silence[j] = predicted[i]
        j = j+1
    return silence


def separate_labels(predicted, lengths):
    """

    :param predicted:
    :param lengths:
    :return:
    """
    labels = [[]]*len(lengths)
    for i in range(len(lengths)):
        if i == 0:
            labels[i] = predicted[0:lengths[i]]
        else:
            labels[i] = predicted[lengths[i-1]+1:lengths[i]]
    return labels


def labels_to_number(labels):
    """

    :param labels:
    :return:
    """
    dct = {'music': 0, 'music_speech': 1, 'speech': 2, 'noise': 3}
    numbers = list(map(dct.get, labels))
    return numbers
