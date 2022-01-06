import numpy as np
from os.path import join, exists
from os import makedirs, listdir
import wfdb


def read_mit_database(path, signal):
    record = wfdb.rdrecord(join(path, str(signal)))
    annotation = wfdb.rdann(join(path, str(signal)), 'atr')

    print(record.sig_name)

    return np.array(record.p_signal), annotation


if __name__ == '__main__':
    folder = r'./data/mit-bih-arrhythmia-database-1.0.0'

    folders = []
    for file in set([file.split('.')[0] for file in listdir(folder)]):
        try:
            folders.append(int(file))
        except ValueError:
            pass

    for file in folders:
        print(file)
        signal, annotations = read_mit_database(folder, file)
