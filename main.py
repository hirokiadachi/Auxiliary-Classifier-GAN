import os
import yaml
import shutil
import argparse
import numpy as np
import multiprocessing
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from dataloader import CelebA_loader
from models import Generator, Discriminator

p = argparse.ArgumentParser()
p.add_argument('--cfile', '-c', type=str, default='config')
p.add_argument('--checkpoints', type=str, default='checkpoints')
p.add_argument('--gpu', '-g', type=str, default='0',
               help='# of GPU. (1 GPU: single GPU)')
args = p.parse_args()

##################################
# Loading training configure
##################################
with open(args.cfile) as yml_file:
    config = yaml.safe_load(yml_file.read())['training']

batch_size = config['batch_size']
start_epoch = config['start_epoch']
end_epoch = config['end_epoch']
lr = config['lr']
beta = config['beta']
weight_decay = config['weight_decay']
tb = config['tb']
img_size = config['img_size']
attr_idx = config['attr_idx']

print('#'*50)
print('# Batch size: {}\n'
      '# Epoch (start/end): {}/{}\n'
      '# Learning rate: {}\n'
      '# Beta1/Beta2: {}/{}\n'
      '# Weight decay: {}\n'
      '# Image size: {}\n'
      '# Number of attributes: {}'.format(batch_size, start_epoch, end_epoch, lr, beta[0], beta[1], weight_decay, img_size, attr_idx))
print('#'*50)

os.makedirs(args.checkpoints, exist_ok=True)
tb_path = os.path.join(args.checkpoints, tb)
if os.path.exists(tb_path):    shutil.rmtree(tb_path)
tb = SummaryWriter(log_dir=tb_path)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda:0')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main():
    iters = 0
    train_data = CelebA_loader(attribute_index=attr_idx)
    train_sets = DataLoader(dataset=train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=multiprocessing.cpu_count())

    G = nn.DataParallel(Generator(num_classes=len(attr_idx))).to(device)
    G.apply(weights_init)
    D = nn.DataParallel(Discriminator(num_classes=len(attr_idx))).to(device)
    D.apply(weights_init)
    G_optim = optim.Adam(G.parameters(), lr=lr, betas=(beta[0], beta[1]))
    D_optim = optim.Adam(D.parameters(), lr=lr, betas=(beta[0], beta[1]))
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(start_epoch, end_epoch):
        iters = train(epoch, train_sets, G, D, G_optim, D_optim, criterion, iters)
        test(epoch, G, train_sets)
        torch.save(G.state_dict(), os.path.join(args.checkpoints, 'gen'))
        torch.save(D.state_dict(), os.path.join(args.checkpoints, 'dis'))

def train(epoch, train_iter, gen, dis, g_opt, d_opt, criterion, iters):
    gen.train()
    dis.train()
    Tensor = torch.FloatTensor
    flag_real = torch.autograd.Variable(Tensor(batch_size).fill_(1.0), requires_grad=False).to(device)
    flag_fake = torch.autograd.Variable(Tensor(batch_size).fill_(0.0), requires_grad=False).to(device)
    
    for batch_ind, (inputs, targets) in enumerate(train_iter):
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.unsqueeze(2).unsqueeze(2)
        
        z = torch.randn(targets.size(0), 100, 1, 1).to(device)
        z = torch.cat((z, targets), dim=1)
        fake_img = gen(z)
        
        dis.zero_grad()
        dis_adv_fake, dis_pred_fake = dis(fake_img.detach())
        dis_adv_real, dis_pred_real = dis(inputs)
        loss_dis_fake = criterion(dis_adv_fake.view(-1), flag_fake[:targets.size(0)])
        loss_dis_real = criterion(dis_adv_real.view(-1), flag_real[:targets.size(0)])
        predicted_error_fake = criterion(dis_pred_fake, targets.view(-1, len(attr_idx)))
        predicted_error_real = criterion(dis_pred_real, targets.view(-1, len(attr_idx)))
        
        loss_dis_fake = (loss_dis_fake + predicted_error_fake) / 2
        loss_dis_real = (loss_dis_real + predicted_error_real) / 2

        dis_loss = loss_dis_fake + loss_dis_real
        dis_loss.backward()
        d_opt.step()
        
        gen.zero_grad()
        gen_adv, gen_pred = dis(fake_img)
        loss_gen = criterion(gen_adv.view(-1), flag_real[:targets.size(0)])
        predicted_gen = criterion(gen_pred, targets.view(-1, len(attr_idx)))
        gen_loss = (loss_gen + predicted_gen) / 2
        gen_loss.backward()
        g_opt.step()
        
        iters += 1
        if batch_ind % 100 == 0:
            print('   Epoch: %d (%d iters) | Loss (G): %f | Loss (D): %f | Pred (Real): %f | Pred (Fake): %f |'\
                % (epoch, iters, gen_loss.item(), dis_loss.item(), predicted_error_real.item(), predicted_gen.item()))
            tb.add_scalars('Total loss', 
                {'dis': dis_loss, 'gen': gen_loss}, 
                global_step=iters)
            tb.add_scalars('Classification loss', 
                {'real': predicted_error_real, 'fake': predicted_gen},
                global_step=iters)
    return iters

def test(epoch, gen, train_data):
    gen.eval()
    for idx, (_, attr) in enumerate(train_data):
        if idx > 0:    break
        rand_idx = np.random.randint(len(attr))
        attr = attr.to(device)[rand_idx:rand_idx+1].repeat(100, 1).unsqueeze(2).unsqueeze(2)
        z = torch.randn(100, 100, 1, 1).type(torch.float32).to(device)
        z = torch.cat((z, attr), dim=1)
        with torch.no_grad():
            fake_img = gen(z)
        tb.add_images('Generated images %s' % attr.data.cpu()[0], fake_img, global_step=epoch)
    
if __name__ == '__main__':
    main()