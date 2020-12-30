import os, sys
import cv2
import numpy as np
from dlib_util import test_68points, get_det_pred, show_68points
from skimage import transform as trans
from codecs import open
import random

det, pred = get_det_pred()
landmarks_2D = np.array([
[ 0.000213256,  0.106454  ], #17
[ 0.0752622,    0.038915  ], #18
[ 0.18113,      0.0187482 ], #19
[ 0.29077,      0.0344891 ], #20
[ 0.393397,     0.0773906 ], #21
[ 0.586856,     0.0773906 ], #22
[ 0.689483,     0.0344891 ], #23
[ 0.799124,     0.0187482 ], #24
[ 0.904991,     0.038915  ], #25
[ 0.98004,      0.106454  ], #26
[ 0.490127,     0.203352  ], #27
[ 0.490127,     0.307009  ], #28
[ 0.490127,     0.409805  ], #29
[ 0.490127,     0.515625  ], #30
[ 0.36688,      0.587326  ], #31
[ 0.426036,     0.609345  ], #32
[ 0.490127,     0.628106  ], #33
[ 0.554217,     0.609345  ], #34
[ 0.613373,     0.587326  ], #35
[ 0.121737,     0.216423  ], #36
[ 0.187122,     0.178758  ], #37
[ 0.265825,     0.179852  ], #38
[ 0.334606,     0.231733  ], #39
[ 0.260918,     0.245099  ], #40
[ 0.182743,     0.244077  ], #41
[ 0.645647,     0.231733  ], #42
[ 0.714428,     0.179852  ], #43
[ 0.793132,     0.178758  ], #44
[ 0.858516,     0.216423  ], #45
[ 0.79751,      0.244077  ], #46
[ 0.719335,     0.245099  ], #47
[ 0.254149,     0.780233  ], #48
[ 0.726104,     0.780233  ], #54
], dtype=np.float32)

class Mask(object):
    def __init__(self, training_path, batch_size):
        self.training_path = training_path
        self.batch_size = batch_size
        self.cur = 0
        self.imgPaths, self.maskPaths, self.train_len = self.getFiles()
        self.indices = [i for i in range(self.train_len)]
        random.shuffle(self.indices)
    
    def next(self):
        if self.cur+self.batch_size<=self.train_len:
            inds = self.indices[self.cur:self.cur+self.batch_size]
            imgs = []
            masks = []
            for i in inds:
                flip = False
                if random.random()>0.7:
                    flip = True
                ip = self.imgPaths[i]
                mp = self.maskPaths[i]
                img = cv2.imread(ip, cv2.IMREAD_COLOR)
                img = cv2.flip(img, 1) if flip else img
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                rand_hsv = np.random.rand(3)*np.array([40, 60, 60])-np.array([20, 30, 30])
                img += rand_hsv
                img = img.clip(0, 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
                img = img.transpose([2, 0, 1]).astype(np.float32)/255
                mask = cv2.imread(mp, cv2.IMREAD_COLOR)
                mask = cv2.flip(mask, 1) if flip else mask
                mask = mask[:,:,0].astype(np.float32)/255
                imgs.append(img)
                masks.append(mask)
            self.cur += self.batch_size
            return np.array(imgs, dtype=np.float32), np.array(masks, dtype=np.float32)
        else:
            self.cur = 0
            random.shuffle(self.indices)
            return self.next()
    
    def getFiles(self):
        lines = open(self.training_path, 'r', 'utf-8').read().strip().split('\n')
        imgs = []
        masks = []
        for l in lines:
            ip, mp = l.split(' ')
            imgs.append(ip)
            masks.append(mp)
        return imgs, masks, len(lines)

def getFileList(path):
    files = [path+'/'+p for p in os.listdir(path)]
    return files

def getAlignMatrix(dic):
    points = []
    for _, v in dic.items():
        points += v
    points = points[17:49]+points[54:55]
    points = np.array(points, dtype=np.float32)
    src = landmarks_2D*np.array([80, 80])
    tform = trans.SimilarityTransform()
    tform.estimate(points, src)
    M = tform.params[0:3,:]
    return M

def getPolyPoints(dic):
    face = dic['face']
    brows = dic['lbrow']+dic['rbrow']
    for b in brows:
        b[1] -= 5
    face = dic['face']
    # for f in face:
    #     f[1] -= 3
    face.reverse()
    key_points = face+brows
    key_points = np.array(key_points, dtype=np.int32)
    return key_points

def drawMask(points, img):
    shp = img.shape
    mask = np.zeros(shp, dtype=np.uint8)
    cv2.fillPoly(mask, [points], (255, 255, 255))
    mask = mask.astype(np.float32)/255
    return mask

def getMaskPair(img):
    dic = test_68points(img, det, pred)
    if dic is None:
        return None, None
    points = getPolyPoints(dic)
    mask = drawMask(points, img)
    M = getAlignMatrix(dic)
    mask = cv2.warpAffine(mask, M[:2], (80, 80), borderValue=0.0)*255
    img = cv2.warpAffine(img, M[:2], (80, 80), borderValue=0.0)
    mask = mask.astype(np.uint8)
    return img, mask

def generateTrainingPairs(path):
    name = path.split('/')[-1]
    cnt = 0
    files = getFileList(path)
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        img, mask = getMaskPair(img)
        ipath = 'pairs/'+name+str(cnt)+'_img.jpg'
        mpath = 'pairs/'+name+str(cnt)+'_mask.jpg'
        cv2.imwrite(ipath, img)
        cv2.imwrite(mpath, mask)
        ff = open('train.txt', 'a', 'utf-8')
        ff.write(ipath+' '+mpath+'\n')
        print(cnt, mpath)
        cnt += 1

def generateByList():
    fList = [
        'faces/baby',
        'faces/xiaoying'
    ]
    for fl in fList:
        generateTrainingPairs(fl)

def test():
    path = 'faces/xiaoying'
    files = getFileList(path)
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        img, mask = getMaskPair(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        rand_hsv = np.random.rand(3)*np.array([40, 60, 60])-np.array([20, 30, 30])
        img += rand_hsv
        img = img.clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        mask = np.concatenate([img, mask], axis=1)
        cv2.imshow('', mask)
        cv2.waitKey(10)

if __name__ == "__main__":
    maskObj = Mask('train.txt', 16)
    while True:
        imgs, masks = maskObj.next()
        print(maskObj.cur, imgs.shape, masks.shape)