import os
from uuid import uuid4
from glob import glob
from PIL import Image

from torch import cat, save, load, set_grad_enabled, randn, sqrt, nn
from torch.nn.functional import pad
from torch.utils.data import Dataset

import torchvision.transforms.functional as transform


class ImageListDataset(Dataset):
    
    def __init__(self, 
                 data_path, 
                 std: float=0.1, 
                 size: int=100,
                 mode='L'):
        super().__init__()
        self.data_path = data_path
        self.std = std
        self.size = size
        self.mode = mode

        self.images = glob(self.data_path + '/**/*.jpg', recursive=True)
        self.images += glob(self.data_path + '/**/*.png', recursive=True)

        dataset_mean = 0
        dataset_sq_mean = 0
        count = len(self.images)*size*size
        for img_path in self.images:
            img = self.__load_image(img_path)
            dataset_mean += img.sum(axis = [1, 2])/count
            dataset_sq_mean += (img ** 2).sum(axis = [1, 2])/count

        self.dataset_mean = dataset_mean
        self.dataset_std = sqrt(dataset_sq_mean - dataset_mean**2)

    def __load_image(self,img_path):
        img = Image.open(img_path)
        img = img.convert(self.mode)

        img = transform.to_tensor(img)
        img = transform.resize(img,self.size)
        img = transform.center_crop(img,self.size)

        return img

    def __getitem__(self, key):
        img = self.__load_image(self.images[key])
        img = transform.normalize(img,self.dataset_mean,self.dataset_std)

        t_img = img.clone()
        t_img += randn(img.size())*self.std
        
        return t_img, img
    
    def __len__(self):
        return len(self.images)
    
def pad_concat_channels(x1,x2,dim=1):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    
    x1 = cat([x1, x2], dim=dim)

    return x1


class ConvBatchRelu(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels,
            kernel_size, padding=kernel_size//2))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.seq = nn.Sequential(*layers)

    def forward(self,x):
        return self.seq(x)

class UNetEncoder(nn.Module):
    def __init__(self, channels=[1,64,128,256,512,1024], kernel_size=3, double_conv=True):
        super().__init__()
        
        self.enc_blocks = nn.ModuleList()

        for i in range(len(channels)-1):
            if double_conv:
                self.enc_blocks.append(nn.Sequential(
                    ConvBatchRelu(channels[i], channels[i+1], kernel_size),
                    ConvBatchRelu(channels[i+1], channels[i+1], kernel_size)))
            else:
                self.enc_blocks.append(ConvBatchRelu(channels[i], channels[i+1], kernel_size))
            
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features

class UNetDecoder(nn.Module):
    def __init__(self, channels=[1024,512,256,128,64], kernel_size=3, double_conv=True):
        super().__init__()
        self.channels = channels

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(len(channels)-1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2))
            if double_conv:
                self.dec_blocks.append(nn.Sequential(
                    ConvBatchRelu(2*channels[i+1], channels[i+1], kernel_size),
                    ConvBatchRelu(channels[i+1], channels[i+1], kernel_size)))
            else:
                self.dec_blocks.append(ConvBatchRelu(2*channels[i+1], channels[i+1], kernel_size))
        
    def forward(self, encoder_features):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](encoder_features[-1-i])
            x = pad_concat_channels(x, encoder_features[-2-i], dim=1)
            x = self.dec_blocks[i](x)
        return x
    


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = str(uuid4())

    def train_test_epoch(self, dataloader, optimizer, criterion, grad_enabled=True):
        running_loss = 0.0

        self.train(grad_enabled)
        set_grad_enabled(grad_enabled)

        for inputs, targets in dataloader:
            if grad_enabled:
                optimizer.zero_grad()

            outputs = self(inputs)
            loss = criterion(outputs, targets)
            
            if grad_enabled:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        return running_loss/len(dataloader)

    def fit(self, 
            train_dataloader, 
            test_dataloader, 
            optimizer, 
            criterion,
            n_epochs = 100,
            patience = 10,
            use_best_model = True):

        best_loss = float('inf')
        best_epoch = 0
        n_its_since_improved = 0

        if use_best_model:
            os.makedirs(os.path.join('runs',self.id),exist_ok=True)
            save(self, os.path.join('runs',self.id,f'training_checkpoint_epoch_{best_epoch}.pt'))

        for epoch in range(n_epochs):
            train_loss = self.train_test_epoch(train_dataloader, optimizer, criterion)

            test_loss = self.train_test_epoch(test_dataloader, optimizer, criterion, grad_enabled=False)

            print(f'epoch: {epoch}. Train loss: {train_loss}. Test loss: {test_loss}')

            if test_loss<best_loss:
                os.remove(os.path.join('runs',self.id,f'training_checkpoint_epoch_{best_epoch}.pt'))
                best_epoch = epoch
                best_loss = test_loss
                n_its_since_improved = 0
                print(f'    Saving checkpoint at epoch: {best_epoch}')
                save(self, os.path.join('runs',self.id,f'training_checkpoint_epoch_{best_epoch}.pt'))
            else:
                n_its_since_improved += 1
                if n_its_since_improved > patience:
                    print(f'    Have not improved in {n_its_since_improved} iterations. Returning.')
                    break        

        print(f'Loading checkpoint at epoch: {best_epoch}')
        print(f'Test loss at best epoch: {best_loss}')
        self = load(os.path.join('runs',self.id,f'training_checkpoint_epoch_{best_epoch}.pt'))


class UNet(Net):
    def __init__(self, channels=[1,64,128,256,512,1024], n_classes=1, kernel_size=3, double_conv=True):
        super().__init__()

        self.encoder = UNetEncoder(channels, kernel_size, double_conv)
        self.decoder = UNetDecoder(channels[:0:-1], kernel_size, double_conv)
        self.head = nn.Conv2d(channels[1],n_classes,1)

    def forward(self,x):
        features = self.encoder(x)
        out = self.decoder(features)
        out = self.head(out)
        return out


class ConvNet(Net):
    def __init__(self, channels=[1,2,1], kernel_size=3):
        super().__init__(self)
        layers = []
        for i in range(len(channels)-1):
            layers.append(ConvBatchRelu(channels[i], channels[i+1, kernel_size]))

        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)