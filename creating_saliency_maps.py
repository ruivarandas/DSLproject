from pathlib import Path
from prep_test_data import *
import shutil
from matplotlib import pyplot as plt
import json


def create_maps_folders(main_folder, beat, labels, delete_prior):
    if delete_prior and Path(main_folder).exists():
        shutil.rmtree(main_folder)
    for label in labels:
        folder = Path(main_folder) / f"label_{beat}_beat/"
        Path(folder / label).mkdir(parents=True, exist_ok=True)
    return folder

def deprocess(image):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        transforms.ToPILImage(),
    ])
    return transform(image)


def saliency_maps(model, data, main_folder, n_batches=None):
    classes = data["test"].dataset.classes
    for i, (inputs, labels) in enumerate(data['test']):
        inputs = inputs.to(0)
        labels = labels.to(0)
        x = inputs
        x.requires_grad_();
        scores = model(x)
        score_max_index = scores.argmax(dim=1)

        score_max = scores[:, score_max_index]
        score_max.backward(torch.ones_like(score_max))
        saliency, _ = torch.max(x.grad.data.abs(), dim=1)
        for index in range(len(saliency)):
            sal = saliency[index].cpu().numpy()

            label = classes[labels[index]]
            true = labels[index]
            pred = score_max_index[index]
            pred_res = "OK"
            if pred != true:
                pred_res = "wrong"
            #
            plt.figure()
            img1 = plt.imshow(sal, cmap=plt.cm.hot, alpha=.7);
            img2 = plt.imshow(deprocess(x[index].cpu()), alpha=.4);
            plt.axis('off')

            input_filename = Path(data['test'].dataset.samples[i * len(saliency) + index][0]).stem
            plt.savefig(str(main_folder / f"{label}/{input_filename}_{pred_res}.png"))
            plt.close();

        if n_batches:
            if i + 1 == n_batches:
                break

    # return saliency, x

def create_saliency_maps_one_heartbeat(data_path, models_main_path, model_name, beat, saliency_maps_path):
    data_prep = DataPreparation(str(data_path))
    data, size = data_prep.create_dataloaders(16, False, 4)
    model_path = models_main_path / f"label_{beat}/{model_name}.pth"
    model = torch.load(model_path, map_location=torch.device(0))
    model.eval();
    return saliency_maps(model, data, saliency_maps_path, None)


def get_model_name(beat):
    d = {
        "final": "resnet50_d_22_t_12_17",
        "initial": "resnet50_d_22_t_19_13",
        "mid": "resnet50_d_22_t_13_24"
    }
    return d[beat]

with open("./config.json") as f:
    config_data = json.load(f)
    f.close()

if __name__ == '__main__':
    MODELS_PATH = Path(f"./models/")
    MAP_DIR = "./attribution_maps/saliency_maps"
    DELETE_PRIOR_DIR = False
    TEST_DATA_PATH = Path(f'./data/figures_final/test')
    NR_BATCHES = 2

    for HEARTBEAT in ["initial", "final", "mid"]:
        print(f"BEAT:{HEARTBEAT}")
        MODEL_NAME = get_model_name(HEARTBEAT)
        saliency_folder = create_maps_folders(MAP_DIR, HEARTBEAT, config_data['labels_bin'], DELETE_PRIOR_DIR)
        create_saliency_maps_one_heartbeat(TEST_DATA_PATH, MODELS_PATH, MODEL_NAME, HEARTBEAT, saliency_folder)