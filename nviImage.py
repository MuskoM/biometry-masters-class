import typing as t
from xmlrpc.client import boolean

import cv2 as cv
from PIL import Image
from PySide6 import QtGui
import numpy as np
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

class NviImage:

    @classmethod
    def _load_vector_image_from_file(cls, file: str):
        drawing = svg2rlg(file)
        pil_drawing = renderPM.drawToPIL(drawing, dpi=72)
        return pil_drawing
    
    @classmethod
    def _load_rasterized_image_from_file(cls, file: str):
        return Image.open(file)

    @classmethod
    def fromFile(cls, name: str):
        compatible_types = [
            ("jpg", 'rasterized'),
            ("jpeg", 'rasterized'),
            ("tiff", 'rasterized'),
            ("png", 'rasterized'),
            ("bmp ", 'rasterized'),
            ("svg", 'vector'),
        ]
        _, extension = name.split('.')
        
        use_type = ''
        for ext, type in compatible_types:
            if extension==ext:
                use_type = type
        
        load_strategy = {
            'vector': cls._load_vector_image_from_file,
            'rasterized': cls._load_rasterized_image_from_file
        }

        converted_image = load_strategy[use_type](name)
        print(converted_image.size)
        return NviImage(converted_image)

    def __init__(self, image: Image = None):
        self._image: Image = image
        self._width = self.image.size[0]
        self._height = self.image.size[1]

    @property
    def shape(self):
        return (self._width, self._height)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def image(self):
        return self._image

    @property
    def cv_image(self) -> cv.Mat:
        arr_img = np.array(self._image)
        return cv.cvtColor(arr_img, cv.COLOR_RGB2BGR)
    

def convert_cv_qt(cv_img, width, height):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(width, height)
    return QtGui.QPixmap.fromImage(p)

def check_boundaries(point:t.Iterable, boundaries: t.Iterable) -> boolean:
    if len(point) != len(boundaries):
        raise ValueError("Point and boundaries are of different dimention")
    
    for ax, bound in zip(point, boundaries):
        if bound[0] <= ax <= bound[1]:
            continue
        else:
            return False
    
    return True

if __name__ == '__main__':
    x = check_boundaries((1,3),((0,2),(1,5)))
    print(x)