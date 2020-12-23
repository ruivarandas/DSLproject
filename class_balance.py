import numpy as np
from os import walk, sep
from os.path import join
import json

def configurations(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data

def get_class_balance():
    custom_labels = {"abnormal": ["A", "a", "J", "S", "V", "E", "F", "P", "/", "f", "Q"], "normal": ["N", "L", "R", "e", "j"]}
    normal, abnormal = 0, 0
    print(f'.{sep}Figures{sep}raw_figures{sep}')
    for folder, files, _ in walk(f'.{sep}Figures{sep}raw_figures{sep}'):

        for f in files:
            print("2")
            for folder1, files1, signals in walk(join(f'.{sep}Figures{sep}raw_figures{sep}', f)):
                print("3")
                for signal in signals:
                    print("4")
                    if '.txt' in signal:
                        f_ = join(join(f'.{sep}Figures{sep}raw_figures{sep}', f), signal)
                        labels = np.loadtxt(f_, dtype=np.object)[1:, 1]
                        for label in labels:
                            print("5")
                            if label in custom_labels['normal']:
                                normal += 1
                            elif label in custom_labels['abnormal']:
                                abnormal += 1
    print(normal)
    print(abnormal)
    print(f"normal={normal}\tabnormal={abnormal}\tunbalancement={abnormal/normal}")
    return {"normal": normal, "abnormal": abnormal}
