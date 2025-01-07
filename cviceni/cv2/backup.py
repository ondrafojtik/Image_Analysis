
#!/usr/bin/python

import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
#from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib
matplotlib.use("WebAgg")


import numpy as np
from skimage.data import lfw_subset
from skimage.feature import hog

import os


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

def show_image(image):
    _image = image
    _image = cv2.resize(_image, (1280, 720))
    cv2.imshow('test', _image)

def draw_rectangle(image, top_left, bottom_right):
    color = (0, 0, 255)
    thickness = 2
    image = cv2.rectangle(image, top_left, bottom_right, color, thickness)

def draw_line(image, p1, p2):
    color = (0, 0, 255)
    thickness = 2
    image = cv2.line(image, p1, p2, color, thickness)

def draw_rectangle_pkm(image, coords):
    p1 = (int(coords[0]), int(coords[1]))
    p2 = (int(coords[2]), int(coords[3]))
    draw_line(image, p1, p2)

    p1 = (int(coords[2]), int(coords[3]))
    p2 = (int(coords[4]), int(coords[5]))
    draw_line(image, p1, p2)

    p1 = (int(coords[4]), int(coords[5]))
    p2 = (int(coords[6]), int(coords[7]))
    draw_line(image, p1, p2)

    p1 = (int(coords[6]), int(coords[7]))
    p2 = (int(coords[0]), int(coords[1]))
    draw_line(image, p1, p2)


def hog_extract(img):
    return hog(img, orientations=8,
               pixels_per_cell=(16, 16),
               cells_per_block=(2, 2),
               visualize=False, channel_axis=None)

def main(argv):
    #cv2.namedWindow("img", 0)

    #images = lfw_subset()
    #cv2.imshow("img", images[0])

    images_pos = []
    images_neg = []

    source_dir_pos = 'bmw/pos_bmw/'
    source_dir_neg = 'bmw/neg_bmw/'

    pos_dir_size = 0
    neg_dir_size = 0

    filenames = os.listdir(source_dir_pos)
    for filename in filenames:
        filepath = source_dir_pos + filename
        img__ = cv2.imread(filepath, 0)
        img_resized = cv2.resize(img__, (80, 80))
        images_pos.append(img_resized)
        pos_dir_size += 1

    filenames = os.listdir(source_dir_neg)
    for filename in filenames:
        filepath = source_dir_neg + filename
        img__ = cv2.imread(filepath, 0)
        #img_resized = cv2.resize(img__, (80, 80))
        images_neg.append(img__)
        neg_dir_size += 1

    print(pos_dir_size)
    print(neg_dir_size)


    #cv2.imshow('t', images_neg[1000])
    #cv2.waitKey()

    #resized_img = cv.resize(images[120], (80, 80)
    #hf = hog_extract(cv.resize(images[120], 80, 80))
    hf = hog_extract(images_neg[2]) #pocet priznaku obrazu
    print("ff")
    print(len(hf))

    train_faces = images_pos
    print(len(train_faces))
    train_faces_lab = [1] * len(train_faces)
    print(train_faces_lab)

    train_neg = images_neg
    print(len(train_neg))
    train_neg_lab = [0] * len(train_neg)
    print(train_neg_lab)

    train_all = np.concatenate((train_faces, train_neg))
    train_lab = np.concatenate((train_faces_lab, train_neg_lab))

    feature_list = []

    print("TRAIN START")

    for i, img in enumerate(train_all):
        res_img = cv2.resize(img, (80, 80))
        X = hog_extract(res_img)
        #print(f'feature dim: {len(X)}')
        feature_list.append(X)
        #cv.imshow("res_img", res_img)
        #cv.waitKey()

    print("TRAIN END")

    #FIT
    clf = svm.SVC(kernel="linear")
    clf.fit(feature_list, train_lab)


    print("TEST START")

    # 1. HOG

    img_85 = cv2.resize(images_pos[1000], (80, 80))
    X = hog_extract(img_85)

    # 2. Predict

    print("TEST END")
    result = clf.predict(X.reshape(1, -1))
    print(f"result: {result}")

    #cv2.imshow("img_85", img_85)
    #cv2.waitKey()

    #cv2.namedWindow("img_parking", 0) #resizable window

    #######################################################
    #######################################################
    #######################################################
    #######################################################

    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)


    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()

    __test_image = cv2.imread(test_images[0], 1)
    __test_image = cv2.imread('bmw_01_select/000897_20210319_111409434.jpg', 1)

    #cv2.imshow("img", __test_image)

    base_rec_size_arr = [50, 100, 150, 200, 250, 300, 350, 400]
    step_arr = [10, 25, 50, 100]

    base_rec_size = 250
    step = 25

    top_left = [0, 0]
    bot_right = [base_rec_size, base_rec_size]

    while 1:
        if bot_right[1] > __test_image.shape[0]:
            break

        pks = []
        #levy horni
        pks.append(top_left[0])
        pks.append(top_left[1])

        #pravy horni
        pks.append(bot_right[0])
        pks.append(top_left[1])

        #pravy dolni
        pks.append(bot_right[0])
        pks.append(bot_right[1])

        #levy dolni
        pks.append(top_left[0])
        pks.append(bot_right[1])

        cropped = four_point_transform(__test_image, pks)
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cropped_gray = cv2.resize(cropped_gray, (80, 80))
        #cv2.imshow('test', cropped_gray)

        X = hog_extract(cropped_gray)

        # 2. Predict

        #print("TEST END")
        result = clf.predict(X.reshape(1, -1))
        #print(f"result: {result}")

        if result == 1:
            draw_rectangle(__test_image, top_left, bot_right)
            cv2.imshow('t', cropped_gray)
            cv2.waitKey()


        #cv2.waitKey()



        if (bot_right[0]+step) > __test_image.shape[1]:
            top_left[0] = 0
            bot_right[0] = base_rec_size
            top_left[1] += step
            bot_right[1] += step
        else:
            top_left[0] += step
            bot_right[0] += step


    result_resized = cv2.resize(__test_image, (1280, 720))
    cv2.imshow("result", result_resized)
    cv2.waitKey()


'''
def main(argv):
    #cv2.namedWindow("img_parking", 0) #resizable window

    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)


    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()

    __test_image = cv2.imread(test_images[0], 1)

    cv2.imshow("img", __test_image)


    base_rec_size = 80
    step = 80

    top_left = [0, 0]
    bot_right = [base_rec_size, base_rec_size]

    while 1:
        if bot_right[1] > __test_image.shape[0]:
            break

        pks = []
        #levy horni
        pks.append(top_left[0])
        pks.append(top_left[1])

        #pravy horni
        pks.append(bot_right[0])
        pks.append(top_left[1])

        #pravy dolni
        pks.append(bot_right[0])
        pks.append(bot_right[1])

        #levy dolni
        pks.append(top_left[0])
        pks.append(bot_right[1])

        draw_rectangle(__test_image, top_left, bot_right)
        cropped = four_point_transform(__test_image, pks)
        #cv2.imshow('test', cropped)
        #cv2.waitKey()



        if (bot_right[0]+step) > __test_image.shape[1]:
            top_left[0] = 0
            bot_right[0] = base_rec_size
            top_left[1] += step
            bot_right[1] += step
        else:
            top_left[0] += step
            bot_right[0] += step


    cv2.imshow("tt", __test_image)
    cv2.waitKey()
'''

if __name__ == "__main__":
   main(sys.argv[1:])
