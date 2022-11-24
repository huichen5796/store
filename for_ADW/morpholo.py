import cv2
import matplotlib.pyplot as plt

image = cv2.imread('test_bild/test3.png', 0)
# image = cv2.resize(image, (1024, 1024))
image = cv2.adaptiveThreshold(image,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 21)
# image = cv2.GaussianBlur(image, (11,11), 0)

image = cv2.bitwise_not(image)
plt.imshow(image, 'gray')
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
image = cv2.dilate(image,kernel,iterations = 1)
plt.imshow(image, 'gray')
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
image = cv2.erode(image,kernel,iterations = 2)
plt.imshow(image, 'gray')
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
image = cv2.dilate(image,kernel,iterations = 1)

plt.imshow(image, 'gray')
plt.show()