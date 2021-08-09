import os
import yaml
import shutil
import argparse
import numpy as np
import multiprocessing
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import CelebA_loader
from network import Generator

p = argparse.ArgumentParser()
p.add_argument('--cfile', '-c', type=str, default='config')
p.add_argument('--save_file', '-s', type=str, default='generated_imgs')
p.add_argument('--gpu', '-g', type=str, default='0',
               help='# of GPU. (1 GPU: single GPU)')
p.add_argument('--resume', '-r', type=str, default='')
args = p.parse_args()

##################################
# Loading training configure
##################################
with open(args.cfile) as yml_file:
    config = yaml.safe_load(yml_file.read())['test']

batch_size = config['batch_size']
attr_idx = config['attr_idx']
img_size = config['img_size']
limit = config['limit']

print('#'*50)
print('# Batch size: {}\n'
      '# Number of generated images: {}\n'
      '# Number of attributes: {}'.format(batch_size, limit, attr_idx))
print('#'*50)

data = CelebA_loader(attribute_index=attr_idx)
data_iters = DataLoader(dataset=data, 
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=multiprocessing.cpu_count())
os.makedirs(args.save_file, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda:0')
G = nn.DataParallel(Generator(classes=len(attr_idx))).to(device)
G.load_state_dict(torch.load(args.resume))
G.eval()

for idx, (_, lbl) in tqdm(enumerate(data_iters)):
    if idx > limit:break
    z = torch.FloatTensor(batch_size, 100, 1, 1).uniform_(-1, 1).to(device)
    z = torch.cat((z, lbl.unsqueeze(2).unsqueeze(2).to(device)), dim=1)
    with torch.no_grad():
        fake = G(z)
        
    fake_img = Image.fromarray((((fake+1)/2) * 255).clamp(min=0, max=255).data.cpu().squeeze().permute(1,2,0).numpy().astype(np.uint8))
    uncondition_path = os.path.join(args.save_file, 'uncondition_imgs')
    os.makedirs(uncondition_path, exist_ok=True)
    fake_img.save(os.path.join(uncondition_path, '{:0>6}.jpg'.format(idx)))

print('Finish!!')