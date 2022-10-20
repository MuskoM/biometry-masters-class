import typing as t
import sys

import cv2 as cv
from PySide6 import QtCore, QtWidgets, QtGui
from PIL import Image

from nviImage import NviImage, convert_cv_qt, check_boundaries

class ImageViewer(QtWidgets.QLabel):
    moved = QtCore.Signal(tuple)

    def __init__(self):
        super().__init__()
    
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> tuple:
        self.moved.emit(event.position().toTuple()) 

class MyWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()

        # Utils
        self.display_width = 640
        self.display_height = 480
        self.loaded_image: NviImage = None

        # Helper widgets
        self.openFileWidget = QtWidgets.QFileDialog()
        
        # Shown widgets
        self.position = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.image = ImageViewer()
        self.open_file_btn = QtWidgets.QPushButton('Open file')
        self.save_file_btn = QtWidgets.QPushButton('Save')
        
        # Layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.position)
        self.layout.addWidget(self.image)
        self.layout.addWidget(self.open_file_btn)
        self.layout.addWidget(self.save_file_btn)

        # Setup
        self.image.setVisible(False)
        self.image.setMouseTracking(False)
        self.position.setVisible(False)

        # Actions
        self.open_file_btn.clicked.connect(self.openFile)
        self.save_file_btn.clicked.connect(self.saveFile)
        self.image.moved.connect(self.showPostition)
        

    @QtCore.Slot(tuple)
    def showPostition(self, pos):
        if check_boundaries(pos,((0,self.loaded_image.cv_image.shape[1]),(0,self.loaded_image.cv_image.shape[0]))):
            text_RGB = str(self.loaded_image.cv_image[int(pos[1]),int(pos[0])])
            text_POS = str(f'X:{pos[0]}, Y:{pos[1]}')
            self.position.setText(f'{text_POS} {text_RGB}')

    @QtCore.Slot()
    def saveFile(self):
        cv.imwrite(self.loaded_image)


    @QtCore.Slot()
    def openFile(self):
        file_name, type = self.openFileWidget.getOpenFileName(filter="Images (*.jpeg *.tiff *.png *.bmp *.jpg *.svg)")
        try: 
            self.loaded_image = NviImage.fromFile(file_name)
            self.display_height = self.loaded_image.height
            self.display_width = self.loaded_image.width
            qt_image = convert_cv_qt(self.loaded_image.cv_image,self.display_width,self.display_height)
            self.image.setPixmap(qt_image)
        except Exception as e:
            self.image.setText(f"Failed to read file {e}")
        self.image.setVisible(True)
        self.position.setVisible(True)
        self.image.setMouseTracking(True)


def run():
    app = QtWidgets.QApplication([])
    widget = MyWidget()
    widget.resize(800,600)
    widget.show()
    sys.exit(app.exec())