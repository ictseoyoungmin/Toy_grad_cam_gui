import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication, QWidget, 
    QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, 
    QLineEdit,QMessageBox,QFrame
)

from event_handler import *


class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.file_path = None   # 업로드 이미지 경로
        self.ori_image = None   # rgb 224 resized img:np.ndarray
        self.model_path = None  # 모델 경로

        self.initUI()
        self.gradcam_pipeline = GradCAMPipeline()
        # self.upload_image()
        
    def initUI(self):
        # TODO
        # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        
        self.setGeometry(500, 500, 1240, 500)
        self.setWindowTitle('Grad CAM Output GUI')
        self.setStyleSheet("""
            QWidget {
                background-color: #262626;
                color: #FFFFFF;
            }

            QLabel {
                border: 1px solid #FFFFFF;
            }

            QPushButton {
                background-color: #373737;
                color: #FFFFFF;
                border: 1px solid #FFFFFF;
                padding: 5px;
            }

            QPushButton:hover {
                background-color: #4D4D4D;
            }

            QLineEdit {
                background-color: #373737;
                color: #FFFFFF;
                border: 1px solid #FFFFFF;
            }

            QFrame#sep {
                background-color: #FFFFFF;
                border-right: 20px solid #FFFFFF;
                margin: 0px 10px;
            }


        """)
            #         QVBoxLayout#contents {
            #     border-right: 1px solid #FFFFFF;
            #     padding-right: 20px;
            # }
        # TODO : title bar
        
        # body 
        qbody_frame = QHBoxLayout(self) 
        
        # main contents layout
        vmain_frame = QVBoxLayout()
        vmain_frame.setContentsMargins(0, 0, 0, 0)
        vmain_frame.setObjectName('contents')
        
        # flugin : 외부 라이브러리 grad-cam, transformers 예정
        vflugin_frame = QVBoxLayout()
        vflugin_frame.setContentsMargins(0,0,0,0)
        vflugin_frame.setObjectName('flugin')

        self.select_gradcam_btn = QPushButton('gradcam 선택',self)
        self.select_gradcam_btn.clicked.connect(lambda : upload_image(self))
        
        # title_bar = TitleBar(self)
        # vmain_frame.addWidget(title_bar)
        
        # 이미지 업로드 버튼
        self.upload_btn = QPushButton('이미지 업로드', self)
        self.upload_btn.clicked.connect(lambda : upload_image(self))
        
        # 결과 버튼
        self.result_btn = QPushButton('결과', self)
        self.result_btn.clicked.connect(lambda : show_result(self))
        
        # 이미지 출력을 위한 QLabel 위젯
        self.original_image_label = QLabel(self)
        self.original_image_label.setFixedSize(500, 500)
        
        self.gradcam_image_label = QLabel(self)
        self.gradcam_image_label.setFixedSize(500, 500)
        
        # 저장된 모델 경로 입력 위젯
        self.model_load_text = QLineEdit()
        self.model_load_text.setFixedSize(600,30)
        # model path upload 버튼
        self.model_load_btn = QPushButton('모델 선택')
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
        self.model_name_text.setStyleSheet("color: #FFFFFF;border: none")
        
        # 레이아웃 설정
        # 전체
        vbox = QVBoxLayout()
        vbox.setContentsMargins(20, 20, 20, 20)
        # 상단 버튼
        hbox_btn = QHBoxLayout()
        # 이미지 배치
        hbox_img = QHBoxLayout()
        hbox_img.setContentsMargins(20, 20, 20, 50)
        # gradcam , result        
        vbox_gradcam = QVBoxLayout()
        vbox_ori_img = QVBoxLayout()
        # grad cam result text 출력
        self.gc_result_cls = QLabel()
        self.gc_result_cls.setFixedSize(500,50)
        self.gc_result_cls.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.load_json_btn = QPushButton('Json 불러오기 [x]')
        self.load_json_btn.clicked.connect(lambda : upload_json(self))
        self.load_json_btn.setFixedSize(500,50)
        # self.original_image_label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        # self.gradcam_image_label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        # 모델 업로드 텍스트
        vbox_model = QVBoxLayout()
        hbox_model_path = QHBoxLayout() # pytorch model.pth address
        hbox_model_path.setContentsMargins(20, 20, 20, 50)
        
        hbox_btn.addWidget(self.upload_btn)
        hbox_btn.addWidget(self.result_btn)
        
        vbox_ori_img.addWidget(self.original_image_label)
        vbox_ori_img.addWidget(self.load_json_btn)
        vbox_gradcam.addWidget(self.gradcam_image_label)
        vbox_gradcam.addWidget(self.gc_result_cls)
        
        hbox_img.addLayout(vbox_ori_img)
        hbox_img.addLayout(vbox_gradcam)
        
        hbox_model_path.addWidget(self.model_load_btn)
        hbox_model_path.addWidget(self.model_load_text)
        hbox_model_path.addWidget(self.model_submit_btn)
        hbox_model_path.addWidget(self.model_clear_btn)

        vbox_model.addLayout(hbox_model_path)
        vbox_model.addWidget(self.model_name_text)
        
        vbox.addLayout(hbox_btn)
        vbox.addLayout(hbox_img)
        vbox.addLayout(vbox_model)
        vmain_frame.addLayout(vbox)

        # flugin layout
        vflugin_frame.addWidget(self.select_gradcam_btn)

        # Separator line
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.VLine)
        self.separator.setFrameShadow(QFrame.Raised)
        self.separator.setObjectName('sep')

        qbody_frame.addLayout(vmain_frame)
        qbody_frame.addWidget(self.separator)
        qbody_frame.addLayout(vflugin_frame)
        self.setLayout(qbody_frame)
        self.style().polish(self)
        
        # msg box
        self.msg_box = QMessageBox()
        self.msg_box.setGeometry(500, 500, 1240, 500)
        self.msg_box.setObjectName('msg_box')
        self.msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #262626;
                color: #FFFFFF;
                border: None;
            }

            QMessageBox QLabel {
                color: #FFFFFF; 
            }

            QMessageBox QPushButton {
                background-color: #373737;
                color: #FFFFFF;
                border: 1px solid #FFFFFF;
                padding: 5px;
            }

            QMessageBox QPushButton:hover {
                background-color: #4D4D4D;
            }
            """)

# TODO
class TitleBar(QWidget):
    qss = """
        QWidget#ww {
            background-color: #C8BFE7;
            color: #FFFFFF;
            padding: 12px;
        }
        QLabel {
            background-color: #373737;
            color: #FFFFFF;
            padding: 12px;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("windowTitle")
        # self.setStyleSheet(self.qss)
        # self.setStyleSheet("background-color: #373737;border: none")
        
        # 레이아웃과 타이틀바 위젯 생성
        layout_container = QWidget()
        # layout_container.setStyleSheet("background-color: #C8BFE7;")
        layout_container.setObjectName('ww')
        vbox = QVBoxLayout(layout_container)
        
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(0, 0, 0, 0)

        # 타이틀 레이블 생성
        self.title_label = QLabel("Grad CAM Output GUI")
        hbox.addWidget(self.title_label)
        hbox.addStretch()

        # 닫기 버튼 생성
        self.close_button = QPushButton(" X ")
        hbox.addWidget(self.close_button)

        # 닫기 버튼 클릭 시 이벤트 연결
        self.close_button.clicked.connect(self.parentWidget().close)
        self.mouse_pressed = False
        self.old_pos = None

        vbox.addLayout(hbox)
        self.setLayout(vbox)
        
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_pressed = True
            self.old_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_pressed = False

    def mouseMoveEvent(self, event):
        if self.mouse_pressed and self.old_pos:
            delta = event.pos() - self.old_pos
            self.parentWidget().move(self.parentWidget().pos() + delta)

      
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())