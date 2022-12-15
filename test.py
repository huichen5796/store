'''
quelle: https://github.com/yghstill/deepLearning_OCR/blob/master/reco_chars.py

'''

import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'D:\\for_tesseract\\tesseract.exe'

path_test_image = 'Testbild_1.jpg'  # path
image_color = cv2.imread(path_test_image)

new_shape = (image_color.shape[1] // 3, image_color.shape[0] // 3)
image_color = cv2.resize(image_color, new_shape)

cv2.imshow('original', image_color)

image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray', image)

adaptive_threshold = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 27)

cv2.imshow('binar', adaptive_threshold)

adaptive_threshold = 255 - adaptive_threshold

cv2.imshow('inver', adaptive_threshold)

def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges

def median_split_ranges(peek_ranges):
    new_peek_ranges = []
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    for i, peek_range in enumerate(peek_ranges):
        num_char = int(round(widthes[i]/median_w, 0))
        if num_char > 1:
            char_w = float(widthes[i] / num_char)
            for i in range(num_char):
                start_point = peek_range[0] + int(i * char_w)
                end_point = peek_range[0] + int((i + 1) * char_w)
                new_peek_ranges.append((start_point, end_point))
        else:
            new_peek_ranges.append(peek_range)
    return new_peek_ranges

def OCR_Tesseract(image):
    '''
    OCR of a image

    - input: image
    - output: str

    '''
    result = pytesseract.image_to_string(
        image, lang='eng', config='--psm 7')

    return result

# Try to find text lines and chars
horizontal_sum = np.sum(adaptive_threshold, axis=1) # Sums pixel values horizontally

peek_ranges = extract_peek_ranges_from_array(horizontal_sum) 

print(peek_ranges)

for i, peek_range in enumerate(peek_ranges):
    x = 0
    y = peek_range[0]
    # w = vertical_range[1] - x
    h = peek_range[1] - y
    char_img = adaptive_threshold[y:y+h+1, x:-1]
    char_img = image = cv2.bitwise_not(char_img)
    cv2.imshow('roi%i'%i, char_img)
    print(OCR_Tesseract(char_img))

cv2.waitKey()
