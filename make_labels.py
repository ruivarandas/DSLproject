from read_data_210 import *

if __name__ == "__main__":
    folder = r'.\mit-bih-arrhythmia-database-1.0.0'
    parser = argparse.ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-f")
    args = parser.parse_args()
    for file in range(int(args.i), int(args.f)):
        if file not in [110, 120, 204, 206, 211, 216, 218, 224, 225, 226, 227, 229]:
            print(file)
            signal, annotations = read_mit_database(folder, file)
            annotations.standardize_custom_labels()
            labels = annotations.symbol
            R_peaks = annotations.sample
            fs = annotations.fs
            new_folder = join(r'.\data\raw_figures', str(file))
            if not exists(new_folder):
                makedirs(new_folder)
            with open(join('./labels', str(file)+'.txt'), 'w') as f:
                f.write("Sample\tLabel\n")
            for i, segment in enumerate(labels[5:]):
                with open(join('./labels', str(file) + '.txt'), 'a') as f:
                    f.write(str(i) + '\t' + labels[i+5] + '\n')
