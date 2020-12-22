import numpy as np
from os.path import join, exists
from os import makedirs
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from biosignalsnotebooks import generate_time


def read_mit_database(path, signal):
    import wfdb

    record = wfdb.rdrecord(join(path, str(signal)))
    annotation = wfdb.rdann(join(path, str(signal)), 'atr')

    return np.array(record.p_signal), annotation


def segment_ecg(signal, timestamps, num_cycles=5):
    segments = []
    inf = 100
    for i in range(0, len(timestamps)-num_cycles):
        if i < len(timestamps)-num_cycles - 1 and not timestamps[i]-inf < 0:
            segments.append(signal[timestamps[i]-inf:timestamps[i+num_cycles+1]-inf])
        elif timestamps[i]-inf < 0:
            segments.append(signal[:timestamps[i + num_cycles + 1] - inf])
        else:
            segments.append(signal[timestamps[i] - inf:])
    return np.array(segments)


def _ax_plot(secs=10):
    ax = plt.axes()
    ax.set_xticks(np.arange(0, 11, 0.2))
    ax.set_yticks(np.arange(-2, 3, 0.5))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_ylim(-1.8, 1.8)
    ax.set_xlim(0, secs)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
    ax.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    return ax


if __name__ == '__main__':
    folder = r'.\mit-bih-arrhythmia-database-1.0.0'
    for file in list(range(202, 210)):
        if file not in [110, 120, 204, 206, 211, 216, 218, 224, 225, 226, 227, 229]:
            print(file)
            signal, annotations = read_mit_database(folder, file)
            annotations.standardize_custom_labels()
            labels = annotations.symbol
            R_peaks = annotations.sample
            fs = annotations.fs
            segmented_ecg = segment_ecg(signal, R_peaks, 5)
            new_folder = join(r'.\Figures', str(file))
            if not exists(new_folder):
                makedirs(new_folder)
            with open(join(new_folder, str(file)+'.txt'), 'w') as f:
                f.write("Sample\tLabel\n")
            for i, segment in enumerate(segmented_ecg):
                print(f"{i}/{len(segmented_ecg)}", end='\r')
                for j in range(segment.shape[1]):
                    fig = plt.figure(figsize=(30, 4.5))
                    _ = _ax_plot()
                    time = generate_time(segment[:, j], fs)
                    _ = plt.plot(np.array(time) - len(time)/(2*fs) + 5, segment[:, j], color='black')
                    fig.savefig(join(new_folder, str(i) + '_' + str(j)))
                    plt.close(fig)

                with open(join(new_folder, str(file) + '.txt'), 'a') as f:
                    f.write(str(i) + '\t' + labels[i] + '\n')
