import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, img_size=128):
        super(Generator, self).__init__()
        self.features = features = 64
        self.img_size = img_size
        
        self.fc = nn.Linear(z_dim+num_classes, features*8*(img_size//16)**2)
        
        self.conv1 = nn.ConvTranspose2d(features*8, features*4, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(features*4)
        
        self.conv2 = nn.ConvTranspose2d(features*4, features*2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(features*2)
        
        self.conv3 = nn.ConvTranspose2d(features*2, features, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(features)
        
        self.conv4 = nn.ConvTranspose2d(features, 3, 4, 2, 1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        h = self.fc(x.flatten(start_dim=1)).view(x.size(0), self.features*8, self.img_size//16, self.img_size//16)
        h = self.relu(self.bn1(self.conv1(h)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.tanh(self.conv4(h))
        return h
    
class Discriminator(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, img_size=128):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.features = features = 64
        
        self.conv1 = nn.Conv2d(in_ch, features, 4, 2, 1)
        self.drop1 = nn.Dropout(0.5, inplace=False)
        
        self.conv2 = nn.Conv2d(features, features*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(features*2)
        self.drop2 = nn.Dropout(0.5, inplace=False)
        
        self.conv3 = nn.Conv2d(features*2, features*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(features*4)
        self.drop3 = nn.Dropout(0.5, inplace=False)
        
        self.conv4 = nn.Conv2d(features*4, features*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(features*8)
        self.drop4 = nn.Dropout(0.5, inplace=False)
        
        self.classifier = nn.Linear(features*8*(img_size//16)**2, num_classes+1)
        self.lrelu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        
    def forward(self, x):
        h = self.drop1(self.lrelu(self.conv1(x)))
        h = self.drop2(self.lrelu(self.bn2(self.conv2(h))))
        h = self.drop3(self.lrelu(self.bn3(self.conv3(h))))
        h = self.drop4(self.lrelu(self.bn4(self.conv4(h))))
        
        h = h.flatten(start_dim=1)
        h = self.classifier(h)
        aux = h[:, :self.num_classes]
        adv = h[:, self.num_classes:]
        return adv, aux