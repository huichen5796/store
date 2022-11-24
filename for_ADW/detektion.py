import cv2
import numpy as np
# use plt to show image is better than cv2, I thought
import matplotlib.pyplot as plt


def Hough(image, min_long):
    '''
    Lines Detektion durch Hough Transform

    input:
    - image: the input image schould be a gray image.
    - min_long: to controll the min long of lines.

    '''
    edges = cv2.Canny(
        image, 50, 250, apertureSize=3)  # apertureSize is the size of kernel

    lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 50,
                            minLineLength=min_long, maxLineGap=1)
    # make a whiteboard to show the lines
    whiteboard = np.ones((image.shape[0], image.shape[1]))

    for line in lines:
        x1, y1, x2, y2 = line[0]

        cv2.line(whiteboard, (x1, y1), (x2, y2), 0, 1)

    # cv2.imshow('Hough', whiteboard)
    plt.subplot(222)
    plt.title('HOUGH')
    plt.imshow(whiteboard, 'gray')


def LSD(image, min_long):
    '''
    Lines Detektion mit LSD

    input:
    - image
    - min_long

    '''
    lsd = cv2.createLineSegmentDetector(0, scale=1)
    dlines = lsd.detect(image)[0]
    # make a whiteboard to show the lines
    whiteboard = np.ones((image.shape[0], image.shape[1]))
    for dline in dlines:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))

        # für LSD Detektor gibt es kein Parameter zum Filtern der Länge der linien
        # so machen wir selbst.
        long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
        if long >= min_long*min_long:
            cv2.line(whiteboard, (x0, y0), (x1, y1), 0, 1, cv2.LINE_AA)
    # cv2.imshow('LSD', whiteboard)
    plt.subplot(223)
    plt.title('LSD')
    plt.imshow(whiteboard, 'gray')


def FLD(image, min_long):

    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(image)
    whiteboard = np.ones((image.shape[0], image.shape[1]))
    for dline in dlines:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))

        long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
        if long >= min_long*min_long:
            cv2.line(whiteboard, (x0, y0), (x1, y1), 0, 1, cv2.LINE_AA)
    # cv2.imshow('FLD', whiteboard)
    plt.subplot(224)
    plt.title('FLD')
    plt.imshow(whiteboard, 'gray')


def Hough_Circle(image):
    detected_circles = cv2.HoughCircles(image,
                                        cv2.HOUGH_GRADIENT, 1, 400, param1=50,
                                        param2=30, minRadius=0, maxRadius=1000)

    whiteboard = np.ones((image.shape[0], image.shape[1]))

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(whiteboard, (a, b), r, 0, 2)

        plt.subplot(121)
        plt.imshow(image, 'gray')
        plt.title('Original Image')
        plt.subplot(122)
        plt.imshow(whiteboard, 'gray')
        plt.title('Get Circle')
        plt.show()


def main():
    image_path = 'for_ADW/test_bild/test3.png'
    # parameter 0 means read the image as a gray
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (1024, 1024))
    image = cv2.adaptiveThreshold(image,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 21)
    image = cv2.GaussianBlur(image, (11,11), 0)

    min_long = 200  # do not filter
    # cv2.imshow('Origenal Image', image)
    plt.subplot(221)
    plt.title('Original Image')
    plt.imshow(image, 'gray')

    # Hough(image, min_long)
    LSD(image, min_long)
    # FLD(image, min_long)
    # cv2.waitKey()

    plt.show()

    Hough_Circle(image)

main()


