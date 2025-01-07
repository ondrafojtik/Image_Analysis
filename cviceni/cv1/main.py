#!/usr/bin/python

import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob

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
    #print(pkm_coordinates)
    #print("********************************************************")

    __test_image = cv2.imread(test_images[3], 1)

    edge_for_full = cv2.Canny(__test_image, 200, 255, 10)
    white_for_full = 0
    for i in range(edge_for_full.shape[0]):
            for j in range(edge_for_full.shape[1]):
                k = edge_for_full[i,j]
                if edge_for_full[i, j] != 0:
                    white_for_full += 1


    for pks in pkm_coordinates:
        #draw_rectangle_pkm(__test_image, pks)
        __test = four_point_transform(__test_image, pks)

        edge = cv2.Canny(__test, 200, 255, 10)

        #show_image(edge)
        #cv2.waitKey()

        # calc amount of black pixels
        white_pixels = 0
        for i in range(edge.shape[0]):
            for j in range(edge.shape[1]):
                k = edge[i,j]
                if edge[i, j] != 0:
                    white_pixels += 1


        th_rand = np.random.random()
        if white_pixels > (edge.shape[0] * edge.shape[1]) * 0.05:
        #if white_pixels > white_for_full * 0.01:
            cv2.circle(__test_image, (int(pks[0]), int(pks[1])), 20, (0, 0, 255), -1)
        else:
            cv2.circle(__test_image, (int(pks[0]), int(pks[1])), 20, (0, 255, 0), -1)

        #show_image(__test)
        #cv2.waitKey()
    edge2 = cv2.Canny(__test_image, 200, 255, 10)
    show_image(edge2)
    cv2.waitKey()
    show_image(__test_image)
    #show_image(__test)
    cv2.waitKey()

if __name__ == "__main__":
   main(sys.argv[1:])
