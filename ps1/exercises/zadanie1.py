from pprint import pprint
import typing as t
import sys

import cv2 as cv
from PySide6 import QtCore, QtWidgets, QtGui
from PIL import Image
from PIL import ImageQt

from nviImage import NviImage, convert_cv_qt, check_boundaries
from nviWidgets import ImageViewer, MetaDataValues, RGBPicker

class MyWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()

        # Utils
        self.display_width = 640
        self.display_height = 480
        self.loaded_image: NviImage = None
        self.displayed_image: cv.Mat = None

        # Helper widgets
        self.openFileWidget = QtWidgets.QFileDialog()
        
        # Shown widgets
        self.info = MetaDataValues()
        self.picker = RGBPicker()
        self.scrollArea = QtWidgets.QScrollArea()
        self.image = ImageViewer()
        self.open_file_btn = QtWidgets.QPushButton('Open file')
        self.save_file_btn = QtWidgets.QPushButton('Save')
        
        # Layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.info)
        self.layout.addWidget(self.picker)
        self.layout.addWidget(self.scrollArea)
        self.layout.addWidget(self.open_file_btn)
        self.layout.addWidget(self.save_file_btn)

        # Setup
        self.scrollArea.setBackgroundRole(QtGui.QPalette.ColorRole.Highlight)
        self.scrollArea.setWidget(self.image)
        self.scrollArea.setVisible(False)
        self.image.setVisible(False)
        self.image.setMouseTracking(False)
        self.info.setVisible(False)
        self.picker.setVisible(False)

        # Actions
        self.open_file_btn.clicked.connect(self.openFile)
        self.save_file_btn.clicked.connect(self.saveFile)
        self.image.moved.connect(self.showPostition)
        self.image.clicked.connect(self.changeColor)
        

    @QtCore.Slot(tuple)
    def showPostition(self, pos):
        if check_boundaries(pos,((0,self.loaded_image.cv_image.shape[1]),(0,self.loaded_image.cv_image.shape[0]))):
            self.info.set_position(pos[0], pos[1])
            self.info.set_color_values(*self.loaded_image.cv_image[int(pos[1]),int(pos[0])])

    @QtCore.Slot(tuple)
    def changeColor(self, pos):
        if check_boundaries(pos,((0,self.loaded_image.cv_image.shape[1]),(0,self.loaded_image.cv_image.shape[0]))):
            self.displayed_image[int(pos[1]),int(pos[0])] = [*self.picker.get_bgr_values()]
            qt_image = convert_cv_qt(self.displayed_image, self.display_width,self.display_height)
            self.image.setPixmap(qt_image)

    @QtCore.Slot()
    def saveFile(self):
        file_name, type = self.openFileWidget.getSaveFileName(filter="Images (*.jpeg *.tiff *.png *.bmp *.jpg *.svg)")
        self.loaded_image.image = ImageQt.fromqpixmap(self.image.pixmap())
        self.loaded_image.to_file(file_name)


    @QtCore.Slot()
    def openFile(self):
        file_name, type = self.openFileWidget.getOpenFileName(filter="Images (*.jpeg *.tiff *.png *.bmp *.jpg *.svg)")
        try: 
            self.loaded_image = NviImage.fromFile(file_name)
            self.display_height = self.loaded_image.height
            self.display_width = self.loaded_image.width
            self.displayed_image = self.loaded_image.cv_image
            qt_image = convert_cv_qt(self.displayed_image,self.display_width,self.display_height)
            self.image.setPixmap(qt_image)
        except Exception as e:
            self.image.setText(f"Failed to read file {e}")
        self.image.setVisible(True)
        self.info.setVisible(True)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setVisible(True)
        self.image.setMouseTracking(True)
        self.picker.setVisible(True)


def run():
    app = QtWidgets.QApplication([])
    widget = MyWidget()
    widget.resize(800,600)
    widget.show()
    sys.exit(app.exec())