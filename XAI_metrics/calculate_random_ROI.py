from explainability_metrics import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-save")
    args = parser.parse_args()
    save = str(args.save)

    ratios = {'0': [], '3': [], '5': []}

    for HEARTBEAT in ["0", "3", "5"]:
        print(f"\nBEAT: {HEARTBEAT}\n")


        with open(f"./ROI/{HEARTBEAT}_ROI.txt", 'r') as f:
            rois = f.readlines()
        for roi in rois[1:]:
            roi = roi.split('\t')
            top_left = (int(roi[3]), int(roi[4]))
            bottom_right = (int(roi[2]), int(roi[5]))
            patient_file = roi[0] + '_' + roi[1]
            top_left, bottom_right = transforming_roi_points(top_left, bottom_right)
            roi_area = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])
            total_area = 224*224
            ratio = roi_area / total_area

            ratios[HEARTBEAT].append(ratio)

            # print(patient_file, ratio, roi_area, bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])

        print(f"{HEARTBEAT}: {np.mean(ratios[HEARTBEAT])} +- {np.std(ratios[HEARTBEAT])}")
