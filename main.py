from turtle import width
import cv2 as cv
import numpy as np
import easyocr

WIDTH = 800
HEIGHT = 600
DELAYTIME = 5000

font = cv.FONT_HERSHEY_SIMPLEX

def centertext(text, img):
    textsize = cv.getTextSize(text, font, 2, 2)[0]
    textX = (img.shape[1] - textsize[0]) / 2
    textY = (img.shape[0] + textsize[1]) / 2
    return textX,textY

def showtext(text):
    blank = np.zeros((150,500), dtype='uint8')
    x,y = centertext(text,blank)
    cv.putText(blank, text, (int(x), int(y)), font, 2, (255, 255, 255), 2)
    cv.imshow(text, blank)
    cv.waitKey(DELAYTIME)

img = cv.imread('Cars276.png')
blank = np.zeros(img.shape, dtype='uint8')

img = cv.resize(img, (WIDTH,HEIGHT))
img2 = img.copy()
cv.imshow('orginal image', img)
cv.waitKey(DELAYTIME)

showtext('GrayScale')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
cv.waitKey(DELAYTIME)

showtext('Threshold')

threshold, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY )
cv.imshow('Simple Thresholded', thresh)
cv.waitKey(DELAYTIME)

showtext('GaussianBlur')

blur = cv.GaussianBlur(thresh, (9,9), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)
cv.waitKey(DELAYTIME)

showtext('Edge Detection')

canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)
cv.waitKey(DELAYTIME)

contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
print(f'{len(contours)} contour(s) found!')
print(f'{len(hierarchies)} hierarchie(s) found!')

newcontours = []
for contour in contours:
    area = cv.contourArea(contour)
    if area > 1000:
        newcontours.append(contour)

smallestY = HEIGHT
smallestX = WIDTH
largestY = 0
largestX = 0

for point in newcontours[0]:
    if point[0][0] > largestX:
        largestX = point[0][0]
    if point[0][1] > largestY:
        largestY = point[0][1]
    if point[0][0] < smallestX:
        smallestX = point[0][0]
    if point[0][1] < smallestY:
        smallestY = point[0][1]

showtext('Contours')

blank = np.zeros(img.shape, dtype='uint8')
cv.drawContours(blank, newcontours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)
cv.waitKey(DELAYTIME)

showtext('Plate Detected')

print((smallestX,smallestY), (largestX,largestY))
cv.rectangle(img2, (smallestX,smallestY), (largestX,largestY), (0,255,0), thickness=3)
cv.imshow('Rectangle', img2)
cv.waitKey(DELAYTIME)

showtext('Crop Plate')

cropped = img[ smallestY:largestY,smallestX:largestX]
cv.imshow('Cropped', cropped)
cv.waitKey(DELAYTIME)

showtext('OCR on Plate')

reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
result = reader.readtext(cropped)
print(result)

cv.putText(img2, result[0][1], (result[0][0][0][0]+smallestX,result[0][0][0][1]+smallestY+(largestY - smallestY +50)), cv.FONT_HERSHEY_TRIPLEX, 2.0, (255,150,0), 2)
cv.imshow('Text', img2)
cv.waitKey(DELAYTIME)

cv.waitKey(0)