import cv2
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pathlib import Path
import sys
import os

IMG_SIZE = 32

transform_test = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Resize((IMG_SIZE, IMG_SIZE)),
     transforms.Normalize((0.5), (0.5))
     ])

transform2 = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5)),
     #transforms.Resize((IMG_SIZE, IMG_SIZE)),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, one_c):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    pts = [((float(one_c[0])), float(one_c[1])),
            ((float(one_c[2])), float(one_c[3])),
            ((float(one_c[4])), float(one_c[5])),
            ((float(one_c[6])), float(one_c[7]))]

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.LazyLinear(84)
        self.fc3 = nn.LazyLinear(2)

    def forward(self, x):
        #x = torch.mean(x, dim=1, keepdim=True)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

itera = 0

PATH = './cifar_net_at.pth'
net = Net()
net.load_state_dict(torch.load(PATH))


coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 01840_20210319_111448727.jpg

source_dir = 'bmw_01_select/'
filenames = os.listdir(source_dir)

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#video = cv2.VideoWriter('video_at1.avi', fourcc, 1, (1280, 720))

for filename in filenames:
    filepath = source_dir + filename
    print(filepath)

    one_img_bgr = cv2.imread(filepath)
    one_img_rgb = cv2.cvtColor(one_img_bgr, cv2.COLOR_BGR2RGB)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # self.device = "cpu"
    model.eval().to(device)
    #print("device", device)
    #print("model", model)
    transformRCNN = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor = transformRCNN(one_img_rgb).to(device)
    #print(tensor.shape)
    #add batch size
    tensor = tensor.unsqueeze(0)
    outputsRCNN = model(tensor)
    pred_classes = [coco_names[i] for i in outputsRCNN[0]['labels'].cpu().numpy()]
    pred_scores = outputsRCNN[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputsRCNN[0]['boxes'].detach().cpu().numpy()

    #print("pred_bboxes", pred_bboxes)
    #print("pred_scores", pred_scores)
    #print("pred_classes", pred_classes)

    for i, rect in enumerate(pred_bboxes):
        if pred_scores[i] < 0.5: continue
        #print(rect, pred_scores[i], pred_classes[i])
        #cv2.rectangle(one_img_bgr, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), 3)
        pks = []
        pks.append(int(rect[0]))
        pks.append(int(rect[1]))
        pks.append(int(rect[2]))
        pks.append(int(rect[1]))
        pks.append(int(rect[0]))
        pks.append(int(rect[3]))
        pks.append(int(rect[2]))
        pks.append(int(rect[3]))

        #if pred_classes[i] == 'car':
            #cv2.rectangle(one_img_bgr, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), 3)

        __test = four_point_transform(one_img_bgr, pks)
        #__gray = cv2.cvtColor(__test, cv2.COLOR_BGR2GRAY)
        __final = cv2.resize(__test, (200, 200))
        tensor = transform2(__final)
        #output = net(tensor)
        #_, predicted = torch.max(output, 1)
        #string_pred = str(predicted)[8]
        #print(string_pred)
        if pred_classes[i] == 'car':
            # save it
            __test = cv2.resize(__test, (80, 80))
            gray = cv2.cvtColor(__test, cv2.COLOR_BGR2GRAY)
            name = 'cars/car' + str(itera) + '.jpg'
            cv2.imwrite(name, gray)
            itera += 1

        #if string_pred == str(1) and pred_classes[i] == 'car':
            #cv2.rectangle(one_img_bgr, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 0, 255), 3)

        #cv2.imshow("test", __final)
        #cv2.waitKey()

    resized_ = cv2.resize(one_img_bgr, (1280, 720))

    #video.write(resized_)
    #cv2.imshow("result", resized_)
    #cv2.waitKey()


cv2.waitKey()
#video.release()
