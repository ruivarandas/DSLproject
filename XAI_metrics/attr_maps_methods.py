import torch
import numpy as np
from torchray.attribution.grad_cam import gradient_to_grad_cam_saliency
from torchray.attribution.guided_backprop import GuidedBackpropReLU
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
def grad_cam_batch(model, inputs, gb_cam=False):
    if gb_cam:
        x = inputs
    else:
        x = inputs.to(0)
    x.requires_grad_();
    saliency_layer = get_module(model, model.layer4)
    probe = Probe(saliency_layer, target='output')
    y = model(x)
    score_max_index = y.argmax(dim=1)
    z = y[:, score_max_index]
    z.backward(torch.ones_like(z))
    grad_cam = gradient_to_grad_cam_saliency(probe.data[0])
    if not gb_cam:
        return grad_cam, score_max_index
    else:
        return grad_cam, score_max_index, x


def preparing_grad_cam(batch_grad_cam, index):
    heatmap = np.float32(batch_grad_cam[index, 0].cpu().detach())
    return cv2.resize(heatmap, (224, 224))
    # return np.uint8(255 * heatmap)

"""
Guided Back propagation Grad Cam maps
"""
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


def deprocess_image_gb(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def preparing_gb_grad_cam(batch_grad_cam, index, guided_backprop_model, x, labels):
    heatmap = np.float32(batch_grad_cam[index, 0].cpu().detach())
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    cam_mask = cv2.merge([heatmap, heatmap, heatmap])

    gb = guided_backprop_model(x[index].unsqueeze(0).detach().cpu(), target_category=labels[index].cpu())
    gb = gb.transpose((1, 2, 0))
    #return cv2.cvtColor(deprocess_image_gb(cam_mask * gb), cv2.COLOR_BGR2GRAY)
    return cam_mask*gb