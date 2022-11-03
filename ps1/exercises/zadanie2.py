import cv2 as cv
import numpy as np
from nviImage import NviImage

def run():
    img = NviImage.fromFile('public/square.jpg')
    cv.imshow('X', img.cv_image)
    cv.waitKey(0)
    img.to_file('public/saved.jpg')
    cv.waitKey(0)
