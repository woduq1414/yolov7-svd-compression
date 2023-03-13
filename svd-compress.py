import argparse

import torch.nn as nn

from models.common import Conv, DWConv
from utils.google_utils import attempt_download
from models.experimental import Ensemble

from models.experimental import attempt_load
import torch

from tqdm.auto import tqdm
from models.custom import MyConv2dSVD
from copy import deepcopy


def attempt_load_not_fuse(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble




def load_model(weights):
    device = torch.device('cpu')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model_not_fuse = attempt_load_not_fuse(weights, map_location=device)  # load FP32 model

    return model, model_not_fuse




def compress_model(model, model_not_fuse):
    for i in tqdm(range(len(model_not_fuse.model))):

        try:
            model.model[i].conv
        except AttributeError:
            continue

        conv = model.model[i].conv


        new_conv = MyConv2dSVD(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding)

        l1 = get_l1_norm_average(model.model[i])

        w = conv.weight.data
        w_r = w.view(w.size(0), -1).t()

        if l1 >= 1:
            u, s, v = torch.svd_lowrank(w_r, int(min(w_r.shape) * 1))
        elif l1 >= 0.05:
            u, s, v = torch.svd_lowrank(w_r, int(min(w_r.shape) * 0.9))
        elif l1 >= 0.03:
            u, s, v = torch.svd_lowrank(w_r, int(min(w_r.shape) * 0.80))
        else:
            u, s, v = torch.svd_lowrank(w_r, int(min(w_r.shape) * 0.62))

        s_r = s.reshape(1, -1)
        new_conv.weight.data = torch.nn.Parameter(torch.cat([u, s_r, v], 0))
        new_conv.bias = conv.bias

        if w_r.shape[0] * w_r.shape[1] > new_conv.weight.shape[0] * new_conv.weight.shape[1]:
            model.model[i].conv = new_conv
    model.model[-1] = model_not_fuse.model[-1]
    return model


# def compress_model(model, model_not_fuse, is_use_svd=True):
#     for i in tqdm(range(len(model.model))):
#
#         try:
#             model.model[i].conv
#         except AttributeError:
#             continue
#
#         conv = model.model[i].conv
#         if is_use_svd:
#
#
#
#             new_conv = MyConv2dSVD(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding)
#
#             l1 = get_l1_norm_average(model.model[i])
#
#             w = conv.weight.data
#             w_r = w.view(w.size(0), -1).t()
#
#             if l1 >= 1:
#                 u, s, v = torch.svd_lowrank(w_r, int(min(w_r.shape) * 1))
#             elif l1 >= 0.05:
#                 u, s, v = torch.svd_lowrank(w_r, int(min(w_r.shape) * 0.9))
#             elif l1 >= 0.03:
#                 u, s, v = torch.svd_lowrank(w_r, int(min(w_r.shape) * 0.80))
#             else:
#                 u, s, v = torch.svd_lowrank(w_r, int(min(w_r.shape) * 0.62))
#
#             s_r = s.reshape(1, -1)
#             new_conv.weight.data = torch.nn.Parameter(torch.cat([u, s_r, v], 0))
#             new_conv.bias = conv.bias
#
#             if w_r.shape[0] * w_r.shape[1] > new_conv.weight.shape[0] * new_conv.weight.shape[1]:
#                 model.model[i].conv = new_conv
#         else:
#             new_conv = MyConv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding)
#             new_conv.weight.data = conv.weight.data
#             new_conv.bias = conv.bias
#
#     model.model[-1] = model_not_fuse.model[-1]
#     return model



def save_model(model, output_path, is_half=True):
    if is_half:
        ckpt = {'model': deepcopy(model).half(),
                }
    else:
        ckpt = {'model': deepcopy(model),
                }

    torch.save(ckpt, output_path)


def get_l1_norm_average(x):
    try:
        x.conv
    except AttributeError:
        return -1

    w = x.conv.weight.data
    return torch.mean(torch.abs(w))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='svd-compress.py')
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--output-path', type=str, default='compressed.pt', help='model.pt path(s)')
    parser.add_argument('--no-half', action='store_true', help='augmented inference')
    opt = parser.parse_args()

    model, model_not_fuse = load_model(opt.weights)
    compress_model(model, model_not_fuse)
    save_model(model, opt.output_path, not opt.no_half)
