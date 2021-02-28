from notebooks.prep_test_data import *
from attr_maps_methods import *
from pathlib import Path
import json
import torch
import numpy as np
import csv
import sys

"""
ROI functions
"""
def read_rois_file_as_dict(rois_filename, test_data_path):
    dict_per_row = {}
    with open(rois_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for i, row in enumerate(reader):
            path = str(test_data_path / row['label'] / f"{row['File']}_{row['Patient']}")
            dict_per_row[path] = row
    return dict_per_row


def get_roi(sample_path, rois_dict):
    stem_split = sample_path.stem.split("_")
    roi = None
    if f"{stem_split[0]}_{stem_split[1]}" != "0_0":
        try:
            roi = rois_dict[str(sample_path.parents[1] / "normal" /  sample_path.stem)]
        except KeyError:
            roi = rois_dict[str(sample_path.parents[1] / "abnormal" /  sample_path.stem)]
    return roi


def get_roi_points(roi):
    return (int(roi['left']), int(roi["top"])), (int(roi["right"]), int(roi["bottom"]))


def transforming_roi_points(left_top, right_bottom):
    """
    parameters: rois extreme points
    """
    crop_x, crop_y = 750, 125  # each side
    y_ratio, x_ratio = 224/200, 224/1500
    top_left_cropped_resized = (int((left_top[0]-crop_x)*x_ratio), int((left_top[1]-crop_y)*y_ratio))
    delta_x_resized = (right_bottom[0] - left_top[0]) * x_ratio
    delta_y_resized = (right_bottom[1] - left_top[1]) * y_ratio
    right_cropped_resized = (int(top_left_cropped_resized[0]+delta_x_resized), int(top_left_cropped_resized[1]+delta_y_resized))
    return top_left_cropped_resized, right_cropped_resized

"""
Metrics definition
"""
def metric1(attr_map, top_left, right_bottom):
    roi_sum = np.sum(attr_map[top_left[1]:right_bottom[1], top_left[0]:right_bottom[0]])
    map_sum = np.sum(attr_map)
    if map_sum == 0:
        return 0
    return roi_sum/map_sum


"""
Attribution maps
"""

def get_maps(map_type, model, inputs):
    if map_type == "saliency_map":
        return batch_saliency(model, inputs)
    elif map_type == "grad_cam_map":
        return grad_cam_batch(model, inputs)

def prepare_attr_maps(map_type, attr_map, index):
    if map_type == "saliency_map":
        return prepare_saliency(attr_map, index)
    elif map_type == "grad_cam_map":
        return preparing_grad_cam(attr_map, index)

def imshow(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
Computing metrics
"""
def compute_metrics(model, data, batch_size, rois_dict, map_type):
    classes = data["test"].dataset.classes
    metric_values = []
    pred_verification = []
    labels_list = []
    for i, (inputs, labels) in enumerate(data['test']):

        # print(f"batch nr: {i+1}", end='\r')
        sys.stdout.write('\r' + f"batch nr: {i+1}")

        attr_map, score_max_index = get_maps(map_type, model, inputs)

        for index in range(len(attr_map)):

            map_prepared = prepare_attr_maps(map_type, attr_map, index)

            label = classes[labels[index]]

            sample_path = Path(data['test'].dataset.samples[i*batch_size+index][0])

            roi = get_roi(sample_path, rois_dict)
            if roi:
                top_left, bottom_right = get_roi_points(roi)
                if map_type == "saliency_map":
                    top_left, bottom_right = transforming_roi_points(top_left, bottom_right)

                metric_values.append(str(metric1(map_prepared, top_left, bottom_right)))

                true = labels[index]
                pred = score_max_index[index]
                if pred != true:
                    pred_res = "wrong"
                else:
                    pred_res = "ok"

                labels_list.append(label)
                pred_verification.append(pred_res)

    return metric_values, pred_verification, labels_list


def metrics_one_heartbeat(data_path, models_main_path, model_name, beat, batches, rois, map_type):
    data_prep = DataPreparation(str(data_path))
    data, size = data_prep.create_dataloaders(batches, False, 4)
    model_path = models_main_path / f"label_{beat}/{model_name}.pth"
    model = torch.load(model_path, map_location=torch.device(0))
    model.eval();
    return compute_metrics(model, data, batches, rois, map_type)


def save_results(metric_values, predictions_verification, labels_true, beat, map_type):
    res_dict = {
        "values": metric_values,
        "pred_results": predictions_verification,
        "true_labels": labels_true
    }
    with open(f"XAI_metrics/{beat}_{map_type}_metrics.json", "w") as f:
        json.dump(res_dict, f)
    f.close()


def beat_int(beat):
    d = {
        "final":5,
         "mid": 3,
        "initial": 0
    }
    return d[beat]


def get_model_name(beat):
    d = {
        "final": "resnet50_d_22_t_12_17",
        "initial": "resnet50_d_22_t_19_13",
        "mid": "resnet50_d_22_t_13_24"
    }
    return d[beat]


if __name__ == '__main__':
    for HEARTBEAT in ["initial", "final", "mid"]:
        print(f"BEAT:{HEARTBEAT}")
        for attr_map_type in ["saliency_map", "grad_cam_map"]:
            print(f"MAP: {attr_map_type}\n")
            roi_file_path = list((Path.cwd() / "ROI").glob(f"{beat_int(HEARTBEAT)}_ROI.txt"))[0]
            MODELS_PATH = Path(f"./models/")
            MODEL_NAME = get_model_name(HEARTBEAT)
            TEST_DATA_PATH = Path(f'/mnt/Media/bernardo/DSL_test_data')
            BATCH_SIZE = 16
            rois_dict = read_rois_file_as_dict(roi_file_path, TEST_DATA_PATH)
            values, prediction_results, labels = metrics_one_heartbeat(TEST_DATA_PATH, MODELS_PATH, MODEL_NAME,
                                                                       HEARTBEAT, BATCH_SIZE, rois_dict, attr_map_type)
            save_results(values, prediction_results, labels, HEARTBEAT, attr_map_type)