import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
import cv2
from PIL import Image
import os 
import copy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
import torchvision
import torchvision.models as models

# transform is used for data augmentation
transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.70, rotate_limit=0, p=.75),
                #A.HorizontalFlip(p = 0.5),
                #A.VerticalFlip(p = 0.5),
            ])

# make Dataset class
class MakeDataset(Dataset):
    def __init__(self, root_image1, root_mask1, transform):
        images_list = [pp for pp in os.listdir(root_image1)]
        masks_list = [pp.replace('.jpg', '_mask.jpg') for pp in images_list]

        self.path_images = [os.path.join(root_image1, fn) for fn in images_list]
        self.path_masks = [os.path.join(root_mask1, fn) for fn in masks_list]

        # Here is to load all the images and masks in the dataset according 
        # to the corresponding relationship. You can of course also do it via a csv file.

        self.transform = transform

    def __len__(self):
        return len(self.path_images)
    
    def __getitem__(self, idx):
        path_image = self.path_images[idx]
        path_mask = self.path_masks[idx]

        image = np.array(Image.open(path_image))
        mask = np.array(Image.open(path_mask).convert('L'))
        # cause the masks I have used are images with 3 channels, 
        # i use here .convert('L') to transform them to with 1 channels.

        if self.transform:
            augmentation  = self.transform(image = image, mask = mask)
            image = augmentation['image']
            mask = augmentation['mask']

        image = to_tensor(image)
        mask = to_tensor(mask)

        # make sure that the size an number of channels of the images and masks after process here
        # are consistent with the input and the output of the model
        # I want the input of model is image with 3 channels and 1024*1024
        # and the output of model is image with 1 channel and 1024*1024
        # so I let all images after __getitem__() to consistent with it.

        return image, mask



def GetTrainVal(batch_size):

    root_image = ""
    root_mask = ""

    root_image_val = ""
    root_mask_val = ""
    # Fill in with the directory paths for the training and validation sets in your dataset.

    # instantiate dataset
    train_ds = MakeDataset(root_image, root_mask, transform)
    val_ds = MakeDataset(root_image_val, root_mask_val, transform)

    print('The total number of images in the train dataset: ', len(train_ds))
    print('The total number of images in the validation dataset: ', len(val_ds))

    # make dataloader, shuffle should be true
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=True)

    # see the infos of datas
    for img, mask in train_dl:
        print('every batch of train image: ',img.shape, img.dtype)
        # torch.Size([ , 3, 1024, 1024]) torch.float32
        print('every batch of train mask: ',mask.shape, mask.dtype)
        # torch.Size([ , 1, 1024, 1024]) torch.float32
        break
    
    for img, mask in val_dl:
        print('every batch of val image: ',img.shape, img.dtype)
        # torch.Size( , 3, 1024, 1024]) torch.float32
        print('every batch of val mask: ',mask.shape, mask.dtype)
        # torch.Size([ , 1, 1024, 1024]) torch.float32
        break

    return train_dl, val_dl


#---------------------------------------------------------------------------------------------------------------#
# model DenseNetSegmentation
# the model from https://github.com/asagar60/TableNet-pytorch
class DenseNet(nn.Module):
    def __init__(self, requires_grad = True):
        super(DenseNet, self).__init__()
        denseNet = torchvision.models.densenet121(pretrained=True).features
        self.densenet_out_1 = torch.nn.Sequential()
        self.densenet_out_2 = torch.nn.Sequential()
        self.densenet_out_3 = torch.nn.Sequential()

        for x in range(8):
            self.densenet_out_1.add_module(str(x), denseNet[x])
        for x in range(8,10):
            self.densenet_out_2.add_module(str(x), denseNet[x])
        
        self.densenet_out_3.add_module(str(10), denseNet[10])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        
        out_1 = self.densenet_out_1(x)
        out_2 = self.densenet_out_2(out_1)
        out_3 = self.densenet_out_3(out_2)
        return out_1, out_2, out_3

class Decoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(Decoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
                        in_channels = 256,
                        out_channels = 256,
                        kernel_size = kernels[0], 
                        stride = strides[0])
        self.upsample_1_table = nn.ConvTranspose2d(
                        in_channels = 256,
                        out_channels=128,
                        kernel_size = kernels[1],
                        stride = strides[1])
        self.upsample_2_table = nn.ConvTranspose2d(
                        in_channels = 128 + channels[0],
                        out_channels = 256,
                        kernel_size = kernels[2],
                        stride = strides[2])
        self.upsample_3_table = nn.ConvTranspose2d(
                        in_channels = 256 + channels[1],
                        out_channels = 1,
                        kernel_size = kernels[3],
                        stride = strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x) 
        out = self.upsample_1_table(x)
        out = torch.cat((out, pool_4_out), dim=1)
        out = self.upsample_2_table(out)
        out = torch.cat((out, pool_3_out), dim=1)
        out = self.upsample_3_table(out)
        return out

class DenseNetSegmentation(nn.Module):
    def __init__(self, use_pretrained_model = True, basemodel_requires_grad = True):
        super(DenseNetSegmentation, self).__init__()
        
        self.base_model = DenseNet(pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
        self.pool_channels = [512, 256]
        self.in_channels = 1024
        self.kernels = [(1,1), (1,1), (2,2),(16,16)]
        self.strides = [(1,1), (1,1), (2,2),(16,16)]
        
        #common layer
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8))

        self._decoder = Decoder(self.pool_channels, self.kernels, self.strides)


    def forward(self, x):

        pool_3_out, pool_4_out, pool_5_out = self.base_model(x)
        conv_out = self.conv6(pool_5_out) 
        out = self._decoder(conv_out, pool_3_out, pool_4_out) #torch.Size([ , 1, 1024, 1024])
        return out
#---------------------------------------------------------------------------------------------------------------#
# model U-Net
class conv_block(nn.Module):

    def __init__(self, input_channels, output_channels, down=True):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride = 1, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace = True),

                                   nn.Conv2d(output_channels, output_channels, kernel_size=3, stride = 1, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace = True)
                                  )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		        nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, 32)
        self.Conv2 = conv_block(32, 64)
        self.Conv3 = conv_block(64, 128)

        self.Conv4 = conv_block(128, 256)
        self.Conv5 = conv_block(256, 512)

        self.Up5 = up_conv(512, 256)
        self.Up_conv5 = conv_block(512, 256)

        self.Up4 = up_conv(256, 128)
        self.Up_conv4 = conv_block(256, 128)

        self.Up3 = up_conv(128, 64)
        self.Up_conv3 = conv_block(128, 64)

        self.Up2 = up_conv(64, 32)
        self.Up_conv2 = conv_block(64, 32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self,x):

        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1
#---------------------------------------------------------------------------------------------------------------#

device = 'cuda' # 'cuda' means GPU
model = DenseNetSegmentation().to(device)
# or
# model = U_Net().to(device)

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

num_epochs = 100
batch_size = 4

train_dl, val_dl = GetTrainVal(batch_size)

# helper functions
bce_loss = nn.BCEWithLogitsLoss() # sigmoid + BCE
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay = 3e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-12)

epoch_train_loss_list = []
epoch_val_loss_list = []

print('--------------------------------start--------------------------------------')
for epoch in range(1,num_epochs+1):
    print('---------------------------------------------------------------------------')
    print('epoch %s' % epoch)

    current_lr = get_lr(optimizer)
    
    running_loss = []
    val_loss = []

    # Training
    model.train()
    for i,(image, truth) in enumerate(train_dl):
        predictions = model(image.cuda())

        loss = bce_loss(predictions, truth.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss)

    # Validation
    model.eval()
    with torch.no_grad():
        for image1, truth1 in val_dl:
            predictions = model(image.cuda())

            loss = bce_loss(predictions, truth.cuda())
            val_loss.append(loss)
    
    epoch_train_loss = sum(running_loss) / len(running_loss)
    epoch_val_loss = sum(val_loss) / len(val_loss)
    scheduler.step(epoch_val_loss) # LR Scheduler
    
    epoch_train_loss_list.append(epoch_train_loss.cpu().detach().numpy())
    epoch_val_loss_list.append(epoch_val_loss.cpu().detach().numpy())
  
    print(f"==>train_loss: {epoch_train_loss} ==>val_loss: {epoch_val_loss} ==>learn-rate: {current_lr} ")

    if epoch/5 in range(1, 81):
        # Save the model every five epochs
        torch.save(model, 'DenseNetSegmentation_%s.pkl' %(epoch))
        print('save done')

    epoch += 1
