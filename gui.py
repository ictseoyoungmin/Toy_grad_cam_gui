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

from event_handler import *


class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.file_path = None   # 업로드 이미지 경로
        self.ori_image = None   # rgb 224 resized img:np.ndarray
        self.model_path = None  # 모델 경로

        self.initUI()
        self.gradcam = GradCAMPipeline()
        # self.upload_image()
        
    def initUI(self):
        self.setGeometry(500, 500, 1240, 500)
        self.setWindowTitle('Grad CAM Output GUI')
        
        # 이미지 업로드 버튼
        self.upload_btn = QPushButton('이미지 업로드', self)
        self.upload_btn.clicked.connect(lambda : upload_image(self))
        
        # 결과 버튼
        self.result_btn = QPushButton('결과', self)
        self.result_btn.clicked.connect(lambda : show_result(self))
        
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
        self.model_load_btn.setFixedWidth(100)
        self.model_load_btn.clicked.connect(lambda : upload_model(self))
        # submit 버튼
        self.model_submit_btn = QPushButton('적용')
        self.model_submit_btn.setFixedWidth(100)
        self.model_submit_btn.clicked.connect(lambda : apply_model(self))
        # 초기화 버튼
        self.model_clear_btn = QPushButton('초기화')
        self.model_clear_btn.setFixedWidth(100)
        self.model_clear_btn.clicked.connect(lambda : clear_text(self))
        # 현재 모델 표시 텍스트
        self.model_name_text = QLineEdit()
        self.model_name_text.setText("current model : ")
        self.model_name_text.setEnabled(False)
        self.model_name_text.setStyleSheet("border: none")
        
        # 레이아웃 설정
        # 전체
        vbox = QVBoxLayout()
        # 상단 버튼
        hbox_btn = QHBoxLayout()
        # 이미지 배치
        hbox_img = QHBoxLayout()
        # 모델 업로드 텍스트
        vbox_model = QVBoxLayout()
        hbox_model_path = QHBoxLayout() # pytorch model.pth address
        
        
        hbox_btn.addWidget(self.upload_btn)
        hbox_btn.addWidget(self.result_btn)
        
        hbox_img.addWidget(self.original_image_label)
        hbox_img.addWidget(self.gradcam_image_label)
        
        hbox_model_path.addWidget(self.model_load_btn)
        hbox_model_path.addWidget(self.model_load_text)
        hbox_model_path.addWidget(self.model_submit_btn)
        hbox_model_path.addWidget(self.model_clear_btn)

        vbox_model.addLayout(hbox_model_path)
        vbox_model.addWidget(self.model_name_text)
        
        vbox.addLayout(hbox_btn)
        vbox.addLayout(hbox_img)
        vbox.addLayout(vbox_model)
        self.setLayout(vbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())