import torch
import numpy as np
from torchray.attribution.grad_cam import gradient_to_grad_cam_saliency
from torchray.attribution.common import Probe, get_module
import cv2

"""
Saliency maps
"""
def batch_saliency(model, inputs):
    x = inputs.to(0)
    x.requires_grad_()
    scores = model(x)
    score_max_index = scores.argmax(dim=1)
    score_max = scores[:, score_max_index]
    score_max.backward(torch.ones_like(score_max))
    saliency, _ = torch.max(x.grad.data.abs(),dim=1)
    return saliency, score_max_index

def prepare_saliency(sal_map, index):
    return sal_map[index].cpu().numpy()

"""
Grad CAM maps
"""
def grad_cam_batch(model, inputs):
    x = inputs.to(0)
    x.requires_grad_();
    saliency_layer = get_module(model, model.layer4)
    probe = Probe(saliency_layer, target='output')
    y = model(x)
    score_max_index = y.argmax(dim=1)
    z = y[:, score_max_index]
    z.backward(torch.ones_like(z))
    return gradient_to_grad_cam_saliency(probe.data[0]), score_max_index


def preparing_grad_cam(batch_grad_cam, index):
    heatmap = np.float32(batch_grad_cam[index, 0].cpu().detach())
    return cv2.resize(heatmap, (224, 224))
    # return np.uint8(255 * heatmap)