"""
  based on resnet18 to reproduce the architecture of FCN
    paper: https://arxiv.org/abs/1411.4038
    source code: https://github.com/shelhamer/fcn.berkeleyvision.org

  there're three things to consider:
    full conved
    upsampling (bilinear initialized)
    skip connnect (maybe crop needed)

  ps: the way of upsampling, i.g the parameters of deconvolution (kernel_size, padding) is not unique
    only the stride is commonly recognized
"""

import numpy as np
import torch
from torchvision import models

# predefined
NUM_CLASSES =61
DEVICE = "cuda:0"
RESIZE = (320,480)

def bilinear_kernel_initial(in_channels,out_channels,kernel_size):
    if kernel_size%2 == 1:
        center = kernel_size//2
    else:
        center = kernel_size//2 - 0.5

    factor = (kernel_size+1)//2
    row, columm = np.ogrid[range(kernel_size),range(kernel_size)]
    weight2D = (1 - np.abs(row-center)/factor) * (1 - np.abs(columm-center)/factor)

    weight = np.zeros((in_channels,out_channels,kernel_size,kernel_size), dtype="float32")

    # only initialize the kth dimension of the kth convolutional kernel
    weight[range(in_channels),range(out_channels),:,:] = weight2D
    return  torch.as_tensor(weight)

class FCN32s_resnet18(torch.nn.Module):
    def __init__(self):
        super(FCN32s_resnet18,self).__init__()

        for name,layer in list(models.resnet18(pretrained=True).named_children())[:-2]:
            self.add_module(name,layer)

        self.pixelwise = torch.nn.Conv2d(in_channels=512, out_channels=NUM_CLASSES, kernel_size=1, stride=1)
        # may want to initialize ::self.pixelwise.weight.data:: in your own way

        self.deconv = torch.nn.ConvTranspose2d(in_channels=NUM_CLASSES, out_channels=NUM_CLASSES, kernel_size=64, stride=32, padding=16)
        self.deconv.weight.data = bilinear_kernel_initial(in_channels=NUM_CLASSES, out_channels=NUM_CLASSES, kernel_size=64) # self.deconv.weight 数据类型为parameter 加了.data才是tensor

    def forward(self,x):
        y = self.conv1(x)
        y = self.maxpool(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        y = self.pixelwise(y)
        y = self.deconv(y)
        return y

class FCN16s_resnet18(torch.nn.Module):
        def __init__(self):
            super(FCN16s_resnet18, self).__init__()

            for name, layer in list(models.resnet18(pretrained=True).named_children())[:-2]:
                self.add_module(name, layer)

            self.pixelwise_layer3 = torch.nn.Conv2d(in_channels=256, out_channels=NUM_CLASSES, kernel_size=1,
                                                    stride=1)
            self.upsample_16x_layer3 = torch.nn.ConvTranspose2d(in_channels=NUM_CLASSES, out_channels=NUM_CLASSES, kernel_size=32, stride=16, padding=8)
            self.upsample_16x_layer3.weight.data = bilinear_kernel_initial(in_channels=NUM_CLASSES, out_channels=NUM_CLASSES, kernel_size=32)

            # layer4's upsampled was designed to the same size of layer3
            # so no cropping needed
            # otherwise, need to crop with the smaller size

            self.pixelwise_layer4 = torch.nn.Conv2d(in_channels=512, out_channels=NUM_CLASSES, kernel_size=1, stride=1)
            self.upsample_2x_layer4 = torch.nn.ConvTranspose2d(in_channels=NUM_CLASSES, out_channels=NUM_CLASSES, kernel_size=4, stride=2, padding=1)
            self.upsample_2x_layer4.weight.data = bilinear_kernel_initial(in_channels=NUM_CLASSES, out_channels=NUM_CLASSES, kernel_size=4)

        def forward(self, x):
            y = self.conv1(x)
            y = self.maxpool(y)
            y = self.layer1(y)
            y = self.layer2(y)
            y = self.layer3(y)
            score_layer3 = self.pixelwise_layer3(y)

            y = self.layer4(y)
            score_layer4 = self.pixelwise_layer4(y)
            up_score_layer4 = self.upsample_2x_layer4(score_layer4)

            fuse_layer3 = score_layer3 + up_score_layer4
            y = self.upsample_16x_layer3(fuse_layer3)
            return y

class FCN8s_resnet18(torch.nn.Module):
    def __init__(self):
        super(FCN8s_resnet18, self).__init__()

        for name, layer in list(models.resnet18(pretrained=True).named_children())[:-2]:
            self.add_module(name, layer)

        self.pixelwise_layer2 = torch.nn.Conv2d(in_channels=128, out_channels=NUM_CLASSES, kernel_size=1, stride=1)
        self.upsample_8x_layer2 = torch.nn.ConvTranspose2d(in_channels=NUM_CLASSES, out_channels=NUM_CLASSES, kernel_size=16, stride=8, padding=4)
        self.upsample_8x_layer2.weight.data = bilinear_kernel_initial(in_channels=NUM_CLASSES, out_channels=NUM_CLASSES, kernel_size=16)

        self.pixelwise_layer3 = torch.nn.Conv2d(in_channels=256, out_channels=NUM_CLASSES, kernel_size=1, stride=1)
        self.upsample_2x_layer3 = torch.nn.ConvTranspose2d(in_channels=NUM_CLASSES, out_channels=NUM_CLASSES, kernel_size=4, stride=2, padding=1)
        self.upsample_2x_layer3.weight.data = bilinear_kernel_initial(in_channels=NUM_CLASSES,out_channels=NUM_CLASSES, kernel_size=4)

        self.pixelwise_layer4 = torch.nn.Conv2d(in_channels=512, out_channels=NUM_CLASSES, kernel_size=1, stride=1)
        self.upsample_2x_layer4 = torch.nn.ConvTranspose2d(in_channels=NUM_CLASSES, out_channels=NUM_CLASSES, kernel_size=4, stride=2, padding=1)
        self.upsample_2x_layer4.weight.data = bilinear_kernel_initial(in_channels=NUM_CLASSES, out_channels=NUM_CLASSES, kernel_size=4)

    def forward(self, x):
        y = self.conv1(x)
        y = self.maxpool(y)
        y = self.layer1(y)
        y = self.layer2(y)
        score_layer2 = self.pixelwise_layer2(y)

        y = self.layer3(y)
        score_layer3 = self.pixelwise_layer3(y)

        y = self.layer4(y)
        score_layer4 = self.pixelwise_layer4(y)
        up_score_layer4 = self.upsample_2x_layer4(score_layer4)

        fuse_layer3 = score_layer3 + up_score_layer4
        up_score_layer3 = self.upsample_2x_layer3(fuse_layer3)

        fuse_layer2 = score_layer2 + up_score_layer3
        y = self.upsample_8x_layer2(fuse_layer2)
        return y

model = FCN8s_resnet18().to(DEVICE)
# model = FCN16s_resnet18().to(DEVICE)
# model = FCN32s_resnet18().to(DEVICE)

# test if the output size == the input size
print(model)
output= torch.zeros((64, 3, *RESIZE)).to(DEVICE)
print(output.size())

# appendix the resnet architect and size of every step
"""
ResNet18 (
(conv1): k=7, s=2, p=3, 64    ----floor((H+1)/2)  320->160  note！pytorch ignored the left
          (maxpool): k=3, s=2, p=1,     ----(H+3)/4  160->80
          (layer1): Sequential(       
            (0): BasicBlock(           
              (conv1): k=3, s=1, p=1, 64
              (conv2): k=3, s=1, p=1, 64
            (1): BasicBlock(
              (conv1): k=3, s=1, p=1, 64
              (conv2): k=3, s=1, p=1, 64   ----same
            )
          )
          (layer2): Sequential(     
            (0): BasicBlock(
              (conv1): k=3, s=2, p=1, 128  ----(H+7)/8  80->40
              (conv2): k=3, s=1, p=1, 128
              (downsample): Sequential(
                (0): k=1, s=1, Conv2d(64->128)

              )
            )
            (1): BasicBlock(  
              (conv1):  k=3, s=1, p=1, 128
              (conv2):  k=3, s=1, p=1, 128  ----same
            )
          )
          (layer3): Sequential(
            (0): BasicBlock(
              (conv1): k=3, s=2, p=1, 256   ----(H+15)/16  40->20
              (conv2): k=3, s=1, p=1, 256

              (downsample): Sequential(
                (0): k=1, s=1, Conv2d(128->258)
              )
            )
            (1): BasicBlock(
              (conv1): k=3, s=1, p=1, 256
              (conv2): k=3, s=1, p=1, 256   -----same
            )
          )
          (layer4): Sequential(
            (0): BasicBlock(
              (conv1): k=3, s=2, p=1, 512   ----(H+31)/32  20->10
              (conv2): k=3, s=1, p=1, 512
              (downsample): Sequential(
                (0): k=1, s=1, Conv2d(256->512)

              )
            )
            (1): BasicBlock(
              (conv1): k=3, s=1, p=1, 512
              (conv2): k=3, s=1, p=1, 512   ----same
            )
          )
          (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
          (fc): Linear(in_features=512, out_features=1000, bias=True)
        )
        
"""
