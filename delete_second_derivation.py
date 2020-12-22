from os.path import join, exists
from os import makedirs, walk, remove
from numpy import concatenate

if __name__ == '__main__':
    for folder, signal, _ in walk('.\Figures'):
        for s in signal:
            print(s)
            for _, _, segments in walk(join(folder, s)):
                for segment in segments:
                    file = join(join(folder, s), segment)
                    if '_1.png' in file:
                        # print(file)
                        remove(file)
