from sklearn import metrics
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from common_flags import FLAGS
from matplotlib.patches import Rectangle


def show_results(real, predicted, soft=None):
    """
    Printing of the main metrics to check the performance of the system
    # Arguments:
        real: ground truth
        predicted: predictions from the CNN and silence
        soft: softened predictions
    """

    real = real[:len(predicted)]

    # Metrics before softening
    print('Average accuracy before softening= ', metrics.accuracy_score(real, predicted))
    print('Precision before softening= ', metrics.precision_score(real, predicted, average='weighted'))
    print('Recall before softening= ', metrics.recall_score(real, predicted, average='weighted'))
    print('F-score before softening= ', metrics.f1_score(real, predicted, average='weighted'))

    # Accuracy after softening
    if soft is not None:
        print('Average accuracy after softening= ', metrics.accuracy_score(real, soft))
        print('Precision after softening= ', metrics.precision_score(real, soft, average='weighted'))
        print('Recall after softening= ', metrics.recall_score(real, soft, average='weighted'))
        print('F-score after softening= ', metrics.f1_score(real, soft, average='weighted'))

    # Music detection
    labels = number_to_labels(predicted)
    music_pos, music_dur = counting(labels, 'music')
    print('Music detected in:')
    for i in range(len(music_pos)):
        if FLAGS.overlap == 0:
            print('Beginning: ', (music_pos[i] * FLAGS.separation) // 60, 'min ',
                  int((music_pos[i] * FLAGS.separation) % 60),
                  'seg - Duration: ', music_dur[i] * FLAGS.separation)
        else:
            print('Beginning: ', int((music_pos[i] * FLAGS.overlap) // 60), 'min ',
                  int((music_pos[i] * FLAGS.overlap) % 60),
                  'seg - Duration: ', music_dur[i] * FLAGS.overlap)

    return


def soft_max_prob(predicted, files, probabilities):
    # TODO: Introduce probabilities of CNN
    """
    # Arguments:
        predicted:
        files:
        probabilities:
    # Return:
        ret_soft:
    """
    ret_soft = [[]] * files

    for j in range(files):
        if not (FLAGS.separation / FLAGS.overlap == 0):
            join = []
            prob_over = []
            for i in range(len(predicted[j])):
                if i < FLAGS.separation / FLAGS.overlap:
                    join.append(predicted[j][i])
                    prob_over.append(probabilities[j][i])
                else:
                    sum_prob = 0
                    for k in range(int(i-FLAGS.separation / FLAGS.overlap), int(i)):
                        sum_prob += probabilities[j][k]
                    prob_over.append(sum_prob)
                    join.append(predicted[j][prob_over[i].index(max(prob_over[i]))])
        else:
            join = predicted[j]
            # prob_over = probabilities

        soft = []
        for i in range(len(join)):
            if i < FLAGS.wind_len // 2:
                soft.append(Counter(join[0:i + FLAGS.wind_len // 2]).most_common(1)[0][0])
            elif i > len(join) - 1 - FLAGS.wind_len // 2:
                soft.append(Counter(join[i - FLAGS.wind_len // 2:len(join) - 1]).most_common(1)[0][0])
            else:
                soft.append(Counter(join[i - FLAGS.wind_len // 2:i + FLAGS.wind_len // 2]).most_common(1)[0][0])
        ret_soft[j] = soft
    return ret_soft


def soft_max(predicted, files):
    """
    # Arguments:
        predicted:
        files:
    # Return:
        ret_soft:
    """
    ret_soft = [[]]*files

    for j in range(files):
        if not(FLAGS.separation / FLAGS.overlap == 0):
            join = []
            for i in range(len(predicted[j])):
                if i < FLAGS.separation / FLAGS.overlap:
                    join.append(predicted[j][i])
                else:
                    join.append(Counter(predicted[j]
                                        [int(i-FLAGS.separation / FLAGS.overlap):int(i)]).most_common(1)[0][0])
        else:
            join = predicted[j]

        soft = []
        for i in range(len(join)):
            if i < FLAGS.wind_len//2:
                soft.append(Counter(join[0:i+FLAGS.wind_len//2]).most_common(1)[0][0])
            elif i > len(join)-1-FLAGS.wind_len//2:
                soft.append(Counter(join[i-FLAGS.wind_len//2:len(join)-1]).most_common(1)[0][0])
            else:
                soft.append(Counter(join[i-FLAGS.wind_len//2:i+FLAGS.wind_len//2]).most_common(1)[0][0])
        ret_soft[j] = soft
    return ret_soft


def counting(data, label):
    """
    Identifies the position and duration of the appearances of a determined class
    in a sequence of labels
    # Arguments:
        data: sequence of labels
        label: name of class to identify
    # Return:
        pos: list of the offset of every appearance of the class
        length: list of durations of the appearances of the class
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


def predict(probabilities, threshold):
    probabilities_fil = [[]]*len(probabilities)
    for i, item in enumerate(probabilities):
        probabilities_fil[i] = [1*(probabilities[i][j] >= threshold) for j in range(len(probabilities[i]))]

    labels = sigmoid_to_softmax(probabilities_fil)
    return labels


def separate_labels(labels, lengths):
    """
    # Arguments:
        labels: original flow of labels from all demo audio files
        lengths: duration of each demo file
    # Return:
        labels_ret: list of labels from each file
    """
    labels_ret = [[]] * len(lengths)
    for i in range(len(lengths)):
        if i == 0:
            labels_ret[i] = labels[0:lengths[i]]
        else:
            labels_ret[i] = labels[lengths[i - 1] + 1:lengths[i]]
    return labels_ret


def labels_to_number(labels):
    """
    # Arguments:
        labels: original ground truth
    # Return:
        numbers: output of the network according to its output function
    """
    if FLAGS.f_output == 'sigmoid':
        dct = {'music': [1, 0, 0], 'music_speech': [1, 1, 0],
               'speech': [0, 1, 0], 'noise': [0, 0, 1]}
    else:
        dct = {'music': 0, 'music_speech': 1, 'speech': 2, 'noise': 3}
    numbers = list(map(dct.get, labels))
    return numbers


def sigmoid_to_softmax(sigmoid):
    dct = {'music': [1, 0, 0], 'music_speech': [1, 1, 0],
           'speech': [0, 1, 0], 'noise': [0, 0, 1],
           'speech_noise': [0, 1, 1], 'music_noise': [1, 0, 1],
           'music_speech_noise': [1, 1, 1], 'silence': [0, 0, 0]}
    label_dct = {'music': [1, 0, 0, 0], 'music_speech': [0, 1, 0, 0],
                 'speech': [0, 0, 1, 0], 'noise': [0, 0, 0, 1],
                 'speech_noise': [0, 0, 1, 0], 'music_noise': [1, 0, 0, 0],
                 'music_speech_noise': [0, 1, 0, 0], 'silence': [0, 0, 0, 1]}
    labels = [list(dct.keys())[list(dct.values()).index(j)] for j in sigmoid]
    softmax = list(map(label_dct.get, labels))
    return softmax


def number_to_labels(numbers):
    """
    # Arguments:
        labels: original ground truth
    # Return:
        numbers: output of the network according to its output function
    """

    dct = {0: 'music', 1: 'music_speech', 2: 'speech', 3: 'noise'}
    real_labels = list(map(dct.get, numbers))

    return real_labels


def plot_output(labels, subplot, name):
    labels = number_to_labels(labels)
    music_pos, music_dur = counting(labels, 'music')
    music_pos_seg, music_dur_seg = [[]]*len(music_pos), [[]]*len(music_dur)

    for i in range(len(music_pos)):
        if FLAGS.overlap == 0:
            music_pos_seg[i] = music_pos[i] * FLAGS.separation
            music_dur_seg[i] = music_dur[i] * FLAGS.separation
            tot_dur = (len(labels) + 1)*FLAGS.separation
        else:
            music_pos_seg[i] = music_pos[i] * FLAGS.overlap
            music_dur_seg[i] = music_dur[i] * FLAGS.overlap
            tot_dur = len(labels)*FLAGS.overlap+FLAGS.separation

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    for x, xe, in zip(music_pos_seg, music_dur_seg):
        ax1.add_patch(Rectangle((x, 0), xe, 5))

    plt.xlim((0, tot_dur))
    plt.ylim((0, 5))

    if subplot == 1:
        plt.title('Real music locations: ' + str(name))
    else:
        plt.title('Predicted music locations: ' + str(name))
    plt.show()

    return


def real_label_process(real, n_samples, lengths):
    adapted_labels = []
    real_labels = []

    labels = [real[i].split("', '") for i in range(len(real))]
    for i in range(len(labels)):
        labels[i][0] = labels[i][0].split("['")[1]
        labels[i][-1] = labels[i][-1].split("']")[0]

    flat_real = [item for sublist in labels for item in sublist]

    if len(flat_real) == n_samples:
        adapted_labels = flat_real
    else:
        for i in range(n_samples):
            if FLAGS.overlap == 0:
                lower_index = round(i * FLAGS.separation)
                lower_solap = (lower_index + 1) - i * FLAGS.separation
                upper_solap = (i+1) * FLAGS.separation - (lower_index + 1)
            else:
                lower_index = round(i * FLAGS.overlap)
                lower_solap = (lower_index + 1) - i * FLAGS.overlap
                upper_solap = i * FLAGS.overlap + FLAGS.separation - (lower_index + 1)

            if lower_solap > upper_solap:
                index = lower_index
            else:
                index = lower_index + 1

            adapted_labels.append(labels[index])

    if FLAGS.demo_path == "../../databases/muspeak":
        for i in range(len(adapted_labels)):
            real_labels.append(number_to_labels(adapted_labels[i]))
    else:
        real_labels = adapted_labels

    real_labels = labels_to_number(real_labels)
    real_labels = separate_labels(real_labels, lengths)

    return real_labels


def predicted_label_process(probs, lengths):
    # Class correspondence
    if FLAGS.f_output == 'sigmoid':
        predicted_labels = predict(probs, FLAGS.threshold)
    else:
        predicted_labels = np.argmax(probs, axis=-1).tolist()

    predicted_labels = separate_labels(predicted_labels, lengths)

    # Temporal filtering
    if FLAGS.structure == 'simple':
        predicted_labels = soft_max(predicted_labels, len(lengths))

    return predicted_labels
