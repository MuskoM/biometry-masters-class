from PySide6 import QtCore, QtWidgets, QtGui

class ImageViewer(QtWidgets.QGraphicsView):
    moved = QtCore.Signal(tuple)
    clicked = QtCore.Signal(tuple)

    def __init__(self):
        super().__init__()
    
    def print_position(self, event):
        print(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self.moved.emit(event.position().toTuple())
        return super().mouseMoveEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            self.centerOn(event.position())
            self.scale(1.25, 1.25)
        else:
            self.centerOn(event.position())
            self.scale(0.8, 0.8)

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
    
    def get_bgr_values(self):
        return self.b_val_input.text(), self.g_val_input.text(), self.r_val_input.text()

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