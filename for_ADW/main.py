import cv2
import numpy as np
from copy import deepcopy


def getImageFromCamera():

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # frame = process(frame)
        cv2.imshow('output', frame)
        key = cv2.waitKey(24)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("test.png", frame)
            break
        if key == 27:  # exit on ESC
            break

def balance(image):
    KERNEL_SIZE = 20
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE)) / (KERNEL_SIZE * KERNEL_SIZE)
    blurred = cv2.filter2D(image, -1, kernel=kernel)
    image = image / blurred
    image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image.astype(np.uint8)

    cv2.imshow('balance', image)

    return image

def process(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 31)
    cv2.imshow('threshold',image)

    # image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.bitwise_not(image)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    image = cv2.dilate(image, kernel, iterations=2)
    image = cv2.erode(image, kernel, iterations=2)
    image = cv2.dilate(image, kernel, iterations=1)
    #image = cv2.erode(image, kernel, iterations=4)

    cv2.imshow('morphologie',image)

    return image


def findContour(image_bgr):
    # image_bgr = balance(image_bgr)
    image_binar = process(image_bgr)
    contours, _ = cv2.findContours(image_binar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big_contours = []
    # remove bad contours
    for c in contours:
        if cv2.contourArea(c) > 200:
            big_contours.append(c)
    
    boundRect = [None]*len(big_contours)
    for i, c in enumerate(big_contours):
        polyline = cv2.approxPolyDP(c, 5, True)
        boundRect[i] = cv2.boundingRect(polyline)
    
    # draw bounding boxes
    color = (0, 0, 255)
    color_image = np.ones(image.shape, np.uint8)*255
    for x, y, w, h in boundRect:
        # cv2.rectangle(image, (x,y),(x+w,y+h), color, thickness)
        triangle = np.array(
            [[x, y], [x, y+h], [x+w, y+h], [x+w, y]])
        cv2.fillConvexPoly(color_image, triangle, color)


    image_add = cv2.addWeighted(image_bgr, 0.5, color_image, 0.9, 0)

    return image_add

#getImageFromCamera()

image = cv2.imread('for_ADW/test_bild/test1.png', 1)
image = cv2.resize(image, (512,512))
cv2.imshow('original', image)

image = findContour(image)
cv2.imshow('result',image)
cv2.waitKey()