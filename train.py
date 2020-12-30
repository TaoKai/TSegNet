import torch
import torch.optim as optim
import torch.nn as nn
from codecs import open
import numpy as np
import cv2
from mask_process import Mask
from TSegNet import TSegNet
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def testGen(data, model):
    model.eval()
    indices = data.indices.copy()
    random.shuffle(indices)
    indices = indices[:64]
    imgPaths = data.imgPaths
    paths = []
    for i in indices:
        paths.append(imgPaths[i])
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        imgs.append(img)
    imgs = np.array(imgs, dtype=np.float32)
    imgs = torch.tensor(imgs).float()/255
    imgs = imgs.permute(0, 3, 1, 2).to(device)
    out = model(imgs)
    shp = out.shape[1:]
    out = out.reshape(-1, shp[0], shp[1], 1)
    imgs = imgs.permute(0, 2, 3, 1)*out*255
    imgs = imgs.reshape(-1, shp[1], 3)
    img_out = imgs.detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite('test_out.jpg', img_out)
    print('save test pics.')

def train(epoch, batch_size):
    model_path = 'TSEG_model.pth'
    model = TSegNet()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-6, betas=(0.5, 0.999))
    trainData = Mask('train.txt', batch_size)
    for i in range(epoch):
        model.train()
        for j in range(trainData.train_len//batch_size+1):
            imgs, masks = trainData.next()
            imgs = torch.tensor(imgs).float().to(device)
            masks = torch.tensor(masks).float().to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = model.loss(out, masks)
            loss.backward()
            optimizer.step()
            print('epoch', i, 'step', j, 'loss', loss.item())
        torch.save(model.state_dict(), model_path)
        model.eval()
        testGen(trainData, model)

train(5000, 96)
