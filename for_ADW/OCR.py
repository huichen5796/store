#import easyocr
import cv2
import pytesseract
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = 'D:\\for_tesseract\\tesseract.exe'


def Get_ROI(image, zone):
    '''
    get the region of interest in the image

    input:
    - image
    - zone: the location of the ROI

    hier sollte beobachtet werden, dass der Ursprungspunkt des Koordinatensystems des Bilds 
    befindet sich in der oberen linken Ecke. 
    Ich beschreibe die ROI mit Koordinate der oberen linken Ecke der ROI sowie der Weite und Höhe.
    also:
    zone = [x,y,w,h] --> x -- Abszisse der oberen linken Ecke
                         y -- Ordinate der oberen linken Ecke
                         w -- Weite der ROI 
                         h -- Höhe der ROI 

    '''

    x, y, w, h = zone[0], zone[1], zone[2], zone[3]

    ROI_zone = image[(y):(y+h), (x):(x+w)]

    cv2.imshow('ROI', ROI_zone)
    cv2.waitKey()

    return ROI_zone


def OCR_Tesseract(image):
    '''
    OCR of a image

    - input: image
    - output: str

    '''

    result = pytesseract.image_to_string(
        image, lang='deu', config='--psm 7')

    return result


def OCR_Easyocr(image):

    # need to run only once to load model into memory
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image)

    return result


image = cv2.imread('for_ADW/test_bild/test4.png', 1)

print('result of tesseract: ')
print(OCR_Tesseract(image))
plt.imshow(image)
plt.axis('off')
plt.show()
#print('result of easzocr: ')
#print('--------------------------------')
#print(OCR_Easyocr(image))
