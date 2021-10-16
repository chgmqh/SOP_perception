# -*- coding: utf-8 -*-
"""
creator:
chgmqh@163.com
data:
20211015
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import PIL.ImageOps   
from torchvision import transforms as transforms


np.set_printoptions(suppress=True) #取消科学计数法

#搭建卷积神经网络
class AlexNet(nn.Module):
    def __init__(self, num_classes=3, init_weights=False):   
        super(AlexNet, self).__init__()
        self.features = nn.Sequential( 
            nn.Conv2d(1, 16, kernel_size=11, stride=4, padding=2),  
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2),                  
            nn.Conv2d(16, 48, kernel_size=5, padding=2),          
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                 
            nn.Conv2d(48, 128, kernel_size=3, padding=1),        
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),         
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 48, kernel_size=3, padding=1),         
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            #全链接
            nn.Linear(48 * 6 * 6, 864),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(864, 864),
            nn.ReLU(inplace=True),
            nn.Linear(864, num_classes),
            nn.Sigmoid()
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    

myconvnet = AlexNet(num_classes=3, init_weights=True)
myconvnet.load_state_dict(torch.load('D:/Project/labview/202105/Python/myconvnet_par.pkl')) #载入网络参数
myconvnet.eval()

def dataload(img_path):
#读取测试数据并进行预处理
    test_data_transforms = transforms.Compose([
    transforms.Resize((224,224)) ,
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor() ,
    transforms.Normalize(0.485, 0.229)
    ])
    img = Image.open(img_path)
    img = PIL.ImageOps.invert(img)
    img = test_data_transforms(img)
    return img


def predict(img_path):
    img_path = img_path.replace('\\','/')
    test_data = dataload(img_path).unsqueeze(0)
    output = myconvnet(test_data)
    pre_lab = torch.gt(output,0.5).int()
    pre_lab = pre_lab.numpy()[0,]
    #diff =((test_data_y - pre_lab)==0).astype(int)///////
    print("预测结果为：",pre_lab,"预测可能行为",output.data.numpy()[0,])
    return output.data.numpy()[0,].tolist()
