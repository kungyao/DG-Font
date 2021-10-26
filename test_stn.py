import os
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

from datasets.custom_dataset import AugmentOperator
from models.generator import Generator as Generator
from models.guidingNet import GuidingNet

IMAGE_SIZE = 64
STYLE_DIM = 128
OUTPUT_K = 83

def load_model(networks, model_path):
    load_file = model_path
    if os.path.isfile(load_file):
        print("=> loading checkpoint '{}'".format(load_file))
        checkpoint = torch.load(load_file, map_location='cpu')
        for name, net in networks.items():
            tmp_keys = next(iter(checkpoint[name + '_state_dict'].keys()))
            if 'module' in tmp_keys:
                tmp_new_dict = OrderedDict()
                for key, val in checkpoint[name + '_state_dict'].items():
                    tmp_new_dict[key[7:]] = val
                net.load_state_dict(tmp_new_dict)
                networks[name] = net
            else:
                net.load_state_dict(checkpoint[name + '_state_dict'])
                networks[name] = net
    return networks

# model_path = "./logs/stn_first_2_layer/model_6.ckpt"
model_path = "./logs/stn_first_1_layer/model_3.ckpt"
dg_networks = {}
dg_networks['C_EMA'] = GuidingNet(IMAGE_SIZE, {'cont': STYLE_DIM, 'disc': OUTPUT_K}, use_stn=True)
dg_networks['G_EMA'] = Generator(IMAGE_SIZE, STYLE_DIM, use_sn=False, use_stn=True)
# dg_networks = load_model(dg_networks, model_path)
C_EMA = dg_networks['C_EMA']
G_EMA = dg_networks['G_EMA']

augment = AugmentOperator()

def style_font(sample):
    def get_sample_mask(sample: Image.Image):
        sample_mask = sample.copy()
        sample_mask = sample_mask.convert("L")
        sample_mask = sample_mask.point(lambda p: p<200 and 255)
        return sample_mask
    params = {
        'scale' : np.random.uniform(0.707, 1.414),
        'angle' : np.random.uniform(-10, 10),
        'shear' : np.random.uniform(-0.2, 0.2),
        # round to digit zero
        'kernel_size' : (int(round(np.random.uniform(8, 21), 0)) // 2) + 1
    }
    style_sample, style_sample_mask = augment(sample, get_sample_mask(sample), params)
    return style_sample

test_path = "D:/Pytorch/DG-Font/save_folder/id_34"
test_ch_path = []
for p in os.listdir(test_path):
    test_ch_path.append(os.path.join(test_path, p))

output_folder = "./results"
for i, p in enumerate(test_ch_path):
    sample = Image.open(p).convert("RGB")
    sample = style_font(sample)
    sample = sample.resize((64, 64))
    sample.save(os.path.join(output_folder, f"{i}.png"))
    sample = F.to_tensor(sample).unsqueeze(0)
    # # GuidingNet test
    # tmp = C_EMA.features[0](sample)
    # tmp = F.to_pil_image(tmp[0])
    # tmp.save(os.path.join(output_folder, f"{i}_GuidingNet.png"))
    # Generator test
    tmp = G_EMA.cnt_encoder.stn1(sample)
    tmp = F.to_pil_image(tmp[0])
    tmp.save(os.path.join(output_folder, f"{i}_Generator.png"))
