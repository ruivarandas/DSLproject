from json import load
from os.path import join
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import cv2


def convert_to_float(data):
    return np.array([float(val) for val in data])


def present_values(folder, params=None):
    fig, ax = plt.subplots(ncols=3, figsize=(15, 10))
    for i, beat in enumerate(['initial', 'mid', 'final']):
        path_to_results = join(folder, f"{beat}_gb_grad_cam_map_metrics.json")
        with open(path_to_results, 'r') as file:
            sal_initial = load(file)
        values = convert_to_float(sal_initial['values'])
        if params is not None:
            values = values[np.where(values != params)[0]]
        print(f"Mean value of {beat} beat: {np.nanmean(values)*100:.2f} +- {np.nanstd(values)*100:.2f}%")

        ax[i].hist(values[np.logical_not(np.isnan(values))], bins=100)
        ax[i].title.set_text(beat)
        ax[i].grid()
        # make_histogram(values[np.logical_not(np.isnan(values))])
    plt.tight_layout()
    plt.show()


def present(folder, params=None):
    fig, ax = plt.subplots(ncols=3, figsize=(15, 10))
    for i, beat in enumerate(['initial', 'mid', 'final']):
        path_to_results = join(folder, f"{beat}_gb_grad_cam_map_metrics.json")
        with open(path_to_results, 'r') as file:
            sal_initial = load(file)
        values = convert_to_float(sal_initial['values'])
        if params is not None:
            values = values[np.where(np.array(sal_initial[params[0]]) == params[1])[0]]
        print(f"Mean value of {beat} beat: {np.nanmean(values)*100:.2f} +- {np.nanstd(values)*100:.2f}%")

        ax[i].hist(values[np.logical_not(np.isnan(values))], bins=100)
        ax[i].title.set_text(beat)
        ax[i].grid()
        # make_histogram(values[np.logical_not(np.isnan(values))])
    plt.show()


def get_accuracies(folder):
    for beat in ['initial', 'mid', 'final']:
        path_to_results = join(folder, f"{beat}_gb_grad_cam_map_metrics.json")
        with open(path_to_results, 'r') as file:
            sal_initial = load(file)
            print(f"{beat} beat: {len(np.where(np.array(sal_initial['pred_results']) == 'ok')[0]) / len(sal_initial['pred_results'])}")


def maps_comparison(beat, map_name, label, folder=f"/mnt/Media/bernardo/"):
    path1 = join(folder, f"attribution_maps_no_grid/attribution_maps/{map_name}/label_{beat}_beat/{label}")
    path2 = join(folder, f"attribution_maps_with_grid/attribution_maps/{map_name}/label_{beat}_beat/{label}")
    folder1 = listdir(path1)
    folder2 = listdir(path2)
    i = 0
    for file in folder1:
        if file in folder2:
            i += 1
            print(file)
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,10))
            ax1.imshow(cv2.imread(join(path1, file)))
            ax2.imshow(cv2.imread(join(path2, file)))
            ax1.axis('off')
            ax2.axis('off')
            plt.tight_layout()
            plt.show()
            if i == 20:
                break


def make_histogram(d):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure(figsize=(5, 5))
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa', density=True)#,
                                # alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(23, 45, fr'$\mu={np.mean(d)}, b={np.std(d)}$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
