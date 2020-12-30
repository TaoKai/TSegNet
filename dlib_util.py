import dlib
import os, sys
import cv2

def get_det_pred():
    model_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    return detector, predictor

def test_68points(img, detector, predictor):
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        x0 = d.left()
        y0 = d.top()
        x1 = d.right()
        y1 = d.bottom()
        shape = predictor(img, d)
        face = []
        for j in range(0, 17):
            x = shape.part(j).x
            y = shape.part(j).y
            face.append([x, y])
            # cv2.circle(img, (x, y), 3, (0,0,255), 2)
        lbrow = []
        for j in range(17, 22):
            x = shape.part(j).x
            y = shape.part(j).y
            lbrow.append([x, y])
            # cv2.circle(img, (x, y), 3, (0,255,0), 2)
        rbrow = []
        for j in range(22, 27):
            x = shape.part(j).x
            y = shape.part(j).y
            rbrow.append([x, y])
            # cv2.circle(img, (x, y), 3, (255,0,0), 2)
        nose = []
        for j in range(27, 36):
            x = shape.part(j).x
            y = shape.part(j).y
            nose.append([x, y])
            # cv2.circle(img, (x, y), 3, (255,0,255), 2)
        leye = []
        for j in range(36, 42):
            x = shape.part(j).x
            y = shape.part(j).y
            leye.append([x, y])
            # cv2.circle(img, (x, y), 3, (255,255,0), 2)
        reye = []
        for j in range(42, 48):
            x = shape.part(j).x
            y = shape.part(j).y
            reye.append([x, y])
            # cv2.circle(img, (x, y), 3, (0,255,255), 2)
        omouth = []
        for j in range(48, 60):
            x = shape.part(j).x
            y = shape.part(j).y
            omouth.append([x, y])
            # cv2.circle(img, (x, y), 3, (0,100,100), 2)
        imouth = []
        for j in range(60, 68):
            x = shape.part(j).x
            y = shape.part(j).y
            imouth.append([x, y])
            # cv2.circle(img, (x, y), 3, (100,100,255), 2)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        dic = {
            'face':face,
            'lbrow':lbrow,
            'rbrow':rbrow,
            'nose':nose,
            'leye':leye,
            'reye':reye,
            'omouth':omouth,
            'imouth':imouth,
        }
        return dic

def show_68points(img, detector, predictor):
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        x0 = d.left()
        y0 = d.top()
        x1 = d.right()
        y1 = d.bottom()
        shape = predictor(img, d)
        face = []
        for j in range(0, 17):
            x = shape.part(j).x
            y = shape.part(j).y
            face.append([x, y])
            cv2.circle(img, (x, y), 3, (0,0,255), 2)
        lbrow = []
        for j in range(17, 22):
            x = shape.part(j).x
            y = shape.part(j).y
            lbrow.append([x, y])
            cv2.circle(img, (x, y), 3, (0,255,0), 2)
        rbrow = []
        for j in range(22, 27):
            x = shape.part(j).x
            y = shape.part(j).y
            rbrow.append([x, y])
            cv2.circle(img, (x, y), 3, (255,0,0), 2)
        nose = []
        for j in range(27, 36):
            x = shape.part(j).x
            y = shape.part(j).y
            nose.append([x, y])
            cv2.circle(img, (x, y), 3, (255,0,255), 2)
        leye = []
        for j in range(36, 42):
            x = shape.part(j).x
            y = shape.part(j).y
            leye.append([x, y])
            cv2.circle(img, (x, y), 3, (255,255,0), 2)
        reye = []
        for j in range(42, 48):
            x = shape.part(j).x
            y = shape.part(j).y
            reye.append([x, y])
            cv2.circle(img, (x, y), 3, (0,255,255), 2)
        omouth = []
        for j in range(48, 60):
            x = shape.part(j).x
            y = shape.part(j).y
            omouth.append([x, y])
            cv2.circle(img, (x, y), 3, (0,100,100), 2)
        imouth = []
        for j in range(60, 68):
            x = shape.part(j).x
            y = shape.part(j).y
            imouth.append([x, y])
            cv2.circle(img, (x, y), 3, (100,100,255), 2)
        cv2.imshow('img', img)
        cv2.waitKey(1)
        dic = {
            'face':face,
            'lbrow':lbrow,
            'rbrow':rbrow,
            'nose':nose,
            'leye':leye,
            'reye':reye,
            'omouth':omouth,
            'imouth':imouth,
        }
        return dic

if __name__ == "__main__":
    img_path = 'faces/jinzhengen/'
    pics = [img_path+fp for fp in os.listdir(img_path)]
    for p in pics:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        det, pred = get_det_pred()
        dic = show_68points(img, det, pred)