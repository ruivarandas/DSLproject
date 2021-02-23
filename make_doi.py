import os
import numpy as np
from read_data_210 import read_mit_database, generate_time, _ax_plot, normalize, argparse
import matplotlib.pylab as plt
import json

def segment_ecg(signal, timestamps, num_cycles=5, heartbeat=5):
    timestamps = timestamps[1:]
    segments, limits = [], []
    inf = 100
    ind = num_cycles-heartbeat

    for i in range(0, len(timestamps)-num_cycles):
        aux = timestamps[i]
        first = i+num_cycles-ind
        second = i+num_cycles+1-ind
        if i < len(timestamps)-num_cycles - 1 and not timestamps[first]-inf < 0:
            right = timestamps[second] - aux
            left = timestamps[first] - aux
            top = np.max(signal[timestamps[first]-inf:timestamps[second]-inf])
            bottom = np.min(signal[timestamps[first]-inf:timestamps[second]-inf])
        elif timestamps[first]-inf < 0:
            right = timestamps[second]-inf
            left = timestamps[first]-inf
            top = np.max(signal[:right])
            bottom = np.min(signal[:right])
        else:
            right = timestamps[-1-ind] - aux
            left = timestamps[first] - aux
            top = np.max(signal[timestamps[first]-inf:])
            bottom = np.min(signal[timestamps[first]-inf:])
        if i < len(timestamps)-num_cycles - 1 and not timestamps[i]-inf < 0:
            segment = signal[timestamps[i]-inf:timestamps[i+num_cycles+1]-inf]
        elif timestamps[i]-inf < 0:
            segment = signal[:timestamps[i + num_cycles + 1] - inf]
        else:
            segment = signal[timestamps[i] - inf:]
        segments.append(segment)
        limits.append([left, right, bottom, top])

        # print(right, left, top, bottom, timestamps[i+num_cycles] - aux, timestamps[i+num_cycles+1] - aux)
        # plt.figure()
        # plt.plot(segment)
        # plt.hlines([top, bottom], left, right)
        # plt.vlines([left, right], bottom, top)
        # plt.show()
        # input()

    return np.array(segments), np.array(limits)


def convert_values_to_pixels(x, y):
    """
    I made calibration lines in Excel based on the limits of the plot and the corresponding pixel positions
    :param x: int or float
    :param y: int or float
    :return:
    """
    return int(np.round(232.5*x + 375, 0)), int(np.round(-96.389*y + 227.5))


def config_labels():
    with open("config.json") as j:
        config = json.load(j)
        j.close()
    return config["labels_bin"]


def get_binary_label(la, labels_bin_list):
    if la in labels_bin_list['abnormal']:
        return 'abnormal'
    else:
        return 'normal'

if __name__ == "__main__":
    folder = r'./data/mit-bih-arrhythmia-database-1.0.0'
    parser = argparse.ArgumentParser()
    parser.add_argument("-beat")
    args = parser.parse_args()

    binary_labels = config_labels()

    folders = []
    for file in set([file.split('.')[0] for file in os.listdir(folder)]):
        try:
            folders.append(int(file))
        except ValueError:
            pass

    with open(os.path.join(f"./ROI/{args.beat}_ROI.txt"), 'w') as f:
        f.write("Patient\tFile\tbottom\ttop\tleft\tright\n")
        for file in folders:
            print(file)
            signal, annotations = read_mit_database(folder, file)
            annotations.standardize_custom_labels()
            labels = annotations.symbol
            R_peaks = annotations.sample
            fs = annotations.fs
            segmented_ecg, limits = segment_ecg(signal[:, 0], R_peaks, 5, int(args.beat))
            print(len(segmented_ecg))
            print(len(labels))
            for i, segment in enumerate(segmented_ecg):
                print(f"{i}/{len(segmented_ecg)}", end='\r')
                time = generate_time(segment, fs)
                left, right, bottom, top = limits[i]

                # Transformation based on the plot (see the other file)
                left, right = left/fs - len(time)/(2*fs) + 5, right/fs - len(time)/(2*fs) + 5
                bottom, top = (bottom-np.mean(segment))*2/np.ptp(segment), (top-np.mean(segment))*2/np.ptp(segment)

                left, bottom = convert_values_to_pixels(left, bottom)
                right, top = convert_values_to_pixels(right, top)
                f.write(f"{file}\t{i}_0\t{bottom}\t{top}\t{left}\t{right}\t{label}\n")
            print()
