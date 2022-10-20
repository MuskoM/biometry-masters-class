import cv2 as cv
import numpy as np
from nviImage import NviImage



def run():
    img = NviImage.fromFile('public/bw.svg')
    cv.imshow('X', img.cv_image)
    cv.waitKey(0)