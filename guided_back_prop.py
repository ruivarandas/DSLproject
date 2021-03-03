from prep_test_data import *
from pathlib import Path
import json
import torch
from matplotlib import pyplot as plt
from torchray.attribution.common import Probe, get_module
from torchray.attribution.grad_cam import gradient_to_grad_cam_saliency
from torchray.attribution.guided_backprop import GuidedBackpropReLU
import shutil
import numpy as np
import cv2

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

def deprocess_image_gb(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)
class GuidedBackpropReLUModel:
    def __init__(self, model):
        self.model = model.cpu()
#         self.model.eval()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)
        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

def guided_backprop_grad_cam(model, data, main_folder, n_batches=None):
    gb_model = GuidedBackpropReLUModel(model=model)
    classes = data["test"].dataset.classes
    i = 0
    for inputs, labels in data['test']:
        print(f"{i}/{int(len(data['test'].dataset.samples)/16)}", end="\r")
        inputs = inputs#.to('cuda:0')
        labels = labels
        x = inputs
    #     break
        x.requires_grad_();
        saliency_layer = get_module(model, model.layer4)
        probe = Probe(saliency_layer, target='output')
        y = model(x)
        score_max_index = y.argmax(dim=1)
        z = y[:, score_max_index]
        z.backward(torch.ones_like(z))
        saliency = gradient_to_grad_cam_saliency(probe.data[0])

        for index in range(len(saliency)):
            plt.figure()
            heatmap = np.float32(saliency[index, 0].cpu().detach())
            img = np.array(deprocess(x[index].cpu().detach()))

            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            cam_mask = cv2.merge([heatmap, heatmap, heatmap])

            gb = gb_model(x[index].unsqueeze(0).detach().cpu(), target_category=labels[index].cpu())
            gb = gb.transpose((1, 2, 0))
            cam_gb = deprocess_image_gb(cam_mask*gb)

            plt.axis('off')

            label = classes[labels[index]]
            true = labels[index]
            pred = score_max_index[index]
            pred_res = "OK"
            if pred != true:
                pred_res = "wrong"

            input_filename = Path(data['test'].dataset.samples[i * len(saliency) + index][0]).stem
            plt.savefig(str(main_folder / f"{label}/{input_filename}_{pred_res}.png"))
            plt.close();
#         if n_batches:
#             if i + 1 == n_batches:
#                 break
        i += 1

def create_gb_grad_cam_maps_one_heartbeat(data_path, models_main_path, model_name, beat, saliency_maps_path, nr_egs=None):
    data_prep = DataPreparation(str(data_path))
    data, size = data_prep.create_dataloaders(16, False, 4)
    model_path = models_main_path / f"label_{beat}/{model_name}.pth"
    model = torch.load(model_path)
    model.eval();
    guided_backprop_grad_cam(model, data, saliency_maps_path, nr_egs)


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
    MAP_DIR = "./attribution_maps/gb_grad_cam"
    DELETE_PRIOR_DIR = False
    TEST_DATA_PATH = Path(f'./data/figures_final/test')
    NR_BATCHES = 2

    for HEARTBEAT in ["initial", "final", "mid"]:
        print(f"BEAT:{HEARTBEAT}")
        MODEL_NAME = get_model_name(HEARTBEAT)
        saliency_folder = create_maps_folders(MAP_DIR, HEARTBEAT, config_data['labels_bin'], DELETE_PRIOR_DIR)
        create_gb_grad_cam_maps_one_heartbeat(TEST_DATA_PATH, MODELS_PATH, MODEL_NAME, HEARTBEAT, saliency_folder)