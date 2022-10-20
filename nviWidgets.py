from PySide6 import QtCore, QtWidgets, QtGui

class ImageViewer(QtWidgets.QLabel):
    moved = QtCore.Signal(tuple)
    clicked = QtCore.Signal(tuple)

    def __init__(self):
        super().__init__()
    
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> tuple:
        self.moved.emit(event.position().toTuple()) 

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self.clicked.emit(event.position().toTuple())

class RGBPicker(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
    
        self.r_val_input = QtWidgets.QLineEdit()
        self.r_val_input.setPlaceholderText("Red channel value")
        self.g_val_input = QtWidgets.QLineEdit()
        self.g_val_input.setPlaceholderText("Greeen channel value")
        self.b_val_input = QtWidgets.QLineEdit()
        self.b_val_input.setPlaceholderText("Blue channel value")

        self.layout = QtWidgets.QHBoxLayout()

        self.setLayout(self.layout)

        self.layout.addWidget(self.r_val_input)
        self.layout.addWidget(self.g_val_input)
        self.layout.addWidget(self.b_val_input)


class MetaDataValues(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.x_value = QtWidgets.QLabel("X")
        self.y_value = QtWidgets.QLabel("Y")
        self.r_value = QtWidgets.QLabel("R")
        self.g_value = QtWidgets.QLabel("G")
        self.b_value = QtWidgets.QLabel("B")
        
        self.layout = QtWidgets.QHBoxLayout()

        self.setLayout(self.layout)

        self.layout.addWidget(self.x_value)
        self.layout.addWidget(self.y_value)
        self.layout.addWidget(self.r_value)
        self.layout.addWidget(self.g_value)
        self.layout.addWidget(self.b_value)
    
    def set_position(self, x,y):
        self.x_value.setText(f'X: {x}')
        self.y_value.setText(f'Y: {y}')

    def set_color_values(self, r, g, b):
        self.r_value.setText(f'R: {r}')
        self.g_value.setText(f'G: {g}')
        self.b_value.setText(f'G: {b}')