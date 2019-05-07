from sklearn import metrics
from collections import Counter


def show_metrics(real, predicted, soft):
    """
    Printing of the main metrics to check the performance of the system
    :param real: ground truth
    :param predicted: predictions from the CNN and silence
    :param soft: softened predictions
    """

    real = real[:len(predicted)]

    # Metrics before softening
    print('Average accuracy before softening= ', metrics.accuracy_score(real, predicted))
    print('Precision before softening= ', metrics.precision_score(real, predicted, average='weighted'))
    print('Recall before softening= ', metrics.recall_score(real, predicted, average='weighted'))
    print('F-score before softening= ', metrics.f1_score(real, predicted, average='weighted'))

    # Accuracy after softening
    print('Average accuracy after softening= ', metrics.accuracy_score(real, soft))
    print('Precision after softening= ', metrics.precision_score(real, soft, average='weighted'))
    print('Recall after softening= ', metrics.recall_score(real, soft, average='weighted'))
    print('F-score after softening= ', metrics.f1_score(real, soft, average='weighted'))
    return


def show_detections(labels, separation, overlap):
    """

    :param labels:
    :param separation:
    :param overlap:
    :return:
    """
    music_pos, music_dur = counting(labels, 'M')
    print('Music detected in:')
    for i in range(len(music_pos)):
        if overlap == 0:
            print('Beginning: ', (music_pos[i] * separation) // 60, 'min ',
                  int((music_pos[i] * separation) % 60),
                  'seg - Duration: ', music_dur[i] * separation)
        else:
            print('Beginning: ', int((music_pos[i] * overlap) // 60), 'min ',
                  int((music_pos[i] * overlap) % 60),
                  'seg - Duration: ', music_dur[i] * overlap)
    return


def soft_max_prob(predicted, wind_len, param, files, probabilities):
    # TODO: Introduce probabilities of CNN
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
            # prob_over = probabilities

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


def join_labels(predicted, silence, lengths=None):
    """
    Merge of the silence labels and the predicted ones from the CNN
    :param predicted: predictions from the CNN
    :param silence: silence detections
    :param lengths:
    :return: silence: merge of the inputs
    """
    j = 0
    for i in range(len(predicted)):
        while silence[j] != '':
            j = j+1
        silence[j] = predicted[i]
        j = j+1

    if lengths is None:
        labels = silence
    else:
        labels = [[]] * len(lengths)
        for i in range(len(lengths)):
            if i == 0:
                labels[i] = predicted[0:lengths[i]]
            else:
                labels[i] = predicted[lengths[i - 1] + 1:lengths[i]]
    return labels


def labels_to_number(labels, f_output):
    """

    :param labels:
    :param f_output:
    :return:
    """
    if f_output == 'sigmoid':
        dct = {'music': [1, 0, 0], 'music_speech': [1, 1, 0],
               'speech': [0, 1, 0], 'noise': [0, 0, 1]}
    else:
        dct = {'music': 0, 'music_speech': 1, 'speech': 2, 'noise': 3}
    numbers = list(map(dct.get, labels))
    return numbers
