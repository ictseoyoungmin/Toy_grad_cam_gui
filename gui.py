import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, 
    QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, 
    QFileDialog, QLineEdit,
)
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2

from gradcam_pipeline import GradCAMPipeline

class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.ori_image = None
        self.initUI()
        self.gradcam = GradCAMPipeline()
        # self.upload_image()
        
    def initUI(self):
        self.setGeometry(500, 500, 1240, 500)
        self.setWindowTitle('Grad CAM Output GUI')
        
        # 이미지 업로드 버튼
        self.upload_btn = QPushButton('이미지 업로드', self)
        self.upload_btn.clicked.connect(self.upload_image)
        
        # 결과 버튼
        self.result_btn = QPushButton('결과', self)
        self.result_btn.clicked.connect(self.show_result)
        
        # 이미지 출력을 위한 QLabel 위젯
        self.original_image_label = QLabel(self)
        self.original_image_label.setFixedSize(400, 400)
        self.original_image_label.setStyleSheet('border: 1px solid black')
        
        self.gradcam_image_label = QLabel(self)
        self.gradcam_image_label.setFixedSize(400, 400)
        self.gradcam_image_label.setStyleSheet('border: 1px solid black')
        
        # 저장된 모델 경로 입력 위젯
        self.model_load_text = QLineEdit()
        self.model_load_text.setFixedSize(600,30)
        self.model_load_text.setStyleSheet('border: 1px solid black')
        # model path upload 버튼
        self.model_load_btn = QPushButton('모델 선택',self)
        self.model_load_btn.clicked.connect(self.upload_model)
        
                
        # 레이아웃 설정
        vbox = QVBoxLayout()
        hbox_btn = QHBoxLayout()
        hbox_img = QHBoxLayout()
        hbox_model_path = QHBoxLayout() # pytorch model.pth address
        
        
        hbox_btn.addWidget(self.upload_btn)
        hbox_btn.addWidget(self.result_btn)
        
        hbox_img.addWidget(self.original_image_label)
        hbox_img.addWidget(self.gradcam_image_label)
        
        hbox_model_path.addWidget(self.model_load_text)
        hbox_model_path.addWidget(self.model_load_text)
        
        vbox.addLayout(hbox_btn)
        vbox.addLayout(hbox_img)
        vbox.addLayout(hbox_model_path)
        self.setLayout(vbox)

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, '이미지 업로드', '', 'Images (*.png *.xpm *.jpg *.jpeg)')
        self.file_path = file_path
        if file_path:
            ori_image = cv2.resize(cv2.cvtColor(cv2.imread(filename=file_path),cv2.COLOR_BGR2RGB),(224,224))
            self.ori_image = ori_image
            self.plot_qimage(ori_image,self.original_image_label)

    def show_result(self):
        if self.file_path is None:
            return None
        result = self.gradcam.run_pipeline(self.file_path)
        result = (result * 255).astype(np.uint8)
        gc_jet = cv2.applyColorMap(result, cv2.COLORMAP_JET)
        
        alpha = 0.6
        gradcam_jet_resized = cv2.resize(cv2.cvtColor(gc_jet,cv2.COLOR_BGR2RGB), (self.ori_image.shape[1], self.ori_image.shape[0]))
        blended = cv2.addWeighted(self.ori_image, 1- alpha, gradcam_jet_resized, alpha, 0)

        self.plot_qimage(blended,self.gradcam_image_label)
    
    def upload_model(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, '모델 업로드', '', 'CNN model (*.pth *.pt *.plk )')
        
    @staticmethod
    def plot_qimage(image,label):
        # 이미지를 QImage로 변환
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # QImage를 QPixmap으로 변환
        pixmap = QPixmap.fromImage(q_image)

        # QLabel에 QPixmap 설정
        label.setPixmap(pixmap)
        label.setScaledContents(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())