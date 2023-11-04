import sys
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QLabel , QPushButton 
from PyQt5.QtGui import QPixmap
from model import CovidMmodel
from datamodule import CovidDataModule
import matplotlib.pylab as plt
import torch
import argparse
import cv2
from PIL import Image
import torchvision.transforms as T
import numpy as np



class ImageClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()

        # Load the pre-trained model
        self.model = CovidMmodel.load_from_checkpoint('epoch=7-step=1360.ckpt')
        self.model.eval()

        self.label = QLabel(self)
        self.label.resize(400, 300)
        self.label.move(50, 50)
        self.label2 = QLabel(self)
        self.label2.resize(450, 50)
        self.label2.move(50, 450)
        

        self.load_button = self.create_button("Load Image", self.load_image, (50, 400))

    def create_button(self, text, callback, position):
        button = QPushButton(text, self)
        button.clicked.connect(callback)
        button.move(*position)
        return button

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)", options=options)
        if file_name:
            # Load and preprocess the image
            image = Image.open(file_name).convert("RGB")
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = transform(image).unsqueeze(0)
            # image = transform(image).unsqueeze(0)

            # Classify the image
            with torch.no_grad():
                prediction = torch.softmax(self.model.forward(image),dim=-1)
            class_names = ['Bệnh nhân mắc covid-19', 'Bệnh nhân mắc bệnh viêm phổi không do covid-19','Bệnh nhân khỏe mạnh']
            result ="Chẩn đoán của mô hình học máy : " + class_names[np.argmax(prediction)]
            # Display the result
            self.label2.setText(result)

        pixmap = QPixmap(file_name)
        self.label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ImageClassifierGUI()
    gui.show()
    sys.exit(app.exec_())
