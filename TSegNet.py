import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _PixelShuffler(nn.Module):
    def forward(self, input):
        batch_size, c, h, w = input.size()
        rh, rw = (2, 2)
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)
        out = input.view(batch_size, rh, rw, oc, h, w)
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous()
        out = out.view(batch_size, oc, oh, ow)  # channel first
        return out

class _UpScale(nn.Sequential):
    def __init__(self, input_features, output_features):
        super(_UpScale, self).__init__()
        self.add_module('conv2_', nn.Conv2d(input_features, output_features * 4, kernel_size=3, padding=1, padding_mode='reflect'))
        self.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))
        self.add_module('pixelshuffler', _PixelShuffler())    

class SAE_RESNEXT_ENCODER(nn.Module):
    def __init__(self):
        super(SAE_RESNEXT_ENCODER, self).__init__()
        self.resnext50 = models.resnext50_32x4d(pretrained=True)
        self.conv1 = self.resnext50.conv1
        self.bn1 = self.resnext50.bn1
        self.relu = self.resnext50.relu
        self.maxpool = self.resnext50.maxpool
        self.L1 = self.resnext50.layer1
        self.L2 = self.resnext50.layer2
        self.L3 = self.resnext50.layer3
        self.L4 = self.resnext50.layer4
        # self.fc1 = nn.Linear(24576, 1024*4*3)
        # self.fc2 = nn.Linear(1024*4*3, 1024*4*3)
        self.fc_conv = nn.Conv2d(2048, 1024, 1)
        self.fc2_conv = nn.Conv2d(1024, 1024, 1)
        self.upscale = _UpScale(1024, 512)   
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1_out = self.L1(x)
        l2_out = self.L2(l1_out)
        l3_out = self.L3(l2_out)
        l4_out = self.L4(l3_out)
        # l4_out = l4_out.flatten(start_dim=1)
        # fc1_out = self.fc1(l4_out)
        fc2_out = self.fc2_conv(self.fc_conv(l4_out))
        ups_out = self.upscale(fc2_out)
        return ups_out

class _DownScale(nn.Sequential):
    def __init__(self, input_features, output_features):
        super(_DownScale, self).__init__()
        self.add_module('conv2_', nn.Conv2d(input_features, output_features * 4, kernel_size=5, stride=2, padding=2))
        self.add_module('pixelshuffler', _PixelShuffler())

class _ResBlock(nn.Module):
    def __init__(self, input_features):
        super(_ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_features, input_features, 3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_features, input_features, 3, padding=1, padding_mode='reflect')
        self.leak1 = nn.LeakyReLU(0.2, inplace=True)
        self.leak2 = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leak1(out)
        out = self.conv2(out)
        out = out+identity
        out = self.leak2(out)
        return out

class _DecoderBlock(nn.Module):
    def __init__(self, input_features, output_features):
        super(_DecoderBlock, self).__init__()
        self.upscale = _UpScale(input_features, output_features)
        self.res = _ResBlock(output_features)
        self.downscale = _DownScale(output_features, output_features)
        self.leak = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.upscale(x)
        x = self.res(x)
        identity = x
        out = self.downscale(x)
        out = out+identity
        out = self.leak(out)
        return out

class SEG_DECODER(nn.Module):
    def __init__(self):
        super(SEG_DECODER, self).__init__()
        self.db1 = _DecoderBlock(512, 256)
        self.db2 = _DecoderBlock(256, 128)
        self.db3 = _DecoderBlock(128, 64)
        self.db4 = _DecoderBlock(64, 32)
        self.conv = nn.Conv2d(32, 1, 1)
    
    def forward(self, x):
        x = self.db1(x)
        x = self.db2(x)
        x = self.db3(x)
        x = self.db4(x)
        x = self.conv(x)
        return x

class TSegNet(nn.Module):
    def __init__(self):
        super(TSegNet, self).__init__()
        self.encoder = SAE_RESNEXT_ENCODER()
        self.decoderA = SEG_DECODER()
        self.activ = nn.Sigmoid()
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        identity = x
        x = self.encoder(x)
        x = self.decoderA(x)
        x = F.upsample(x, size=identity.shape[2:], mode='bilinear')
        x = self.activ(x)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        return x
    
    def loss(self, out, mask):
        cost = self.criterion(out, mask)
        return cost

if __name__ == "__main__":
    from mask_process import Mask
    maskObj = Mask('train.txt', 16)
    model = TSegNet()
    while True:
        imgs, masks = maskObj.next()
        imgs = torch.tensor(imgs).float()
        masks = torch.tensor(masks).float()
        out = model(imgs)
        cost = model.loss(out, masks)
        print(cost.item())
