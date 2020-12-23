import numpy as np
from os import walk
from os.path import join

def get_class_balance():
    custom_labels = {"abnormal": ["A", "a", "J", "S", "V", "E", "F", "P", "/", "f", "Q"], "normal": ["N", "L", "R", "e", "j"]}
    normal, abnormal = 0, 0
    for folder, files, _ in walk('.\\Figures\\raw_figures\\'):
        for f in files:
            for folder1, files1, signals in walk(join('.\\Figures\\raw_figures\\', f)):
                for signal in signals:
                    if '.txt' in signal:
                        f_ = join(join('.\\Figures\\raw_figures\\', f), signal)
                        labels = np.loadtxt(f_, dtype=np.object)[1:, 1]
                        for label in labels:
                            if label in custom_labels['normal']:
                                normal += 1
                            elif label in custom_labels['abnormal']:
                                abnormal += 1

    print(f"normal={normal}\tabnormal={abnormal}\tunbalancement={abnormal/normal}")
    return {"normal": normal, "abnormal": abnormal}
