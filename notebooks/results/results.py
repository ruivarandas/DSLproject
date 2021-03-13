from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import cv2

def maps_comparison(beat, map_name, label):
    path1 =  f"/mnt/Media/bernardo/attribution_maps_with_grid/attribution_maps/{map_name}/label_{beat}_beat/{label}"
    path2 = f"/mnt/Media/bernardo/attribution_maps_no_grid/attribution_maps/{map_name}/label_{beat}_beat/{label}"
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