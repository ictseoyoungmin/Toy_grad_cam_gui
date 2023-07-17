import os
import cv2
import numpy as np
from PyQt5 .QtWidgets import ( QFileDialog)
from PyQt5.QtGui import QPixmap, QImage

from gradcam_pipeline import GradCAMPipeline

def upload_image(gui):
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(gui, '이미지 업로드', '', 'Images (*.png *.xpm *.jpg *.jpeg)')
    gui.file_path = file_path
    if file_path:
        ori_image = cv2.resize(cv2.cvtColor(cv2.imread(filename=file_path),cv2.COLOR_BGR2RGB),(224,224))
        gui.ori_image = ori_image
        plot_qimage(ori_image,gui.original_image_label)

def show_result(gui):
    if checkModelPath(gui):
        result,prob,cls = gui.gradcam_pipeline.run(gui.file_path)
        result = (result * 255).astype(np.uint8)
        gc_jet = cv2.applyColorMap(result, cv2.COLORMAP_JET)
        
        alpha = 0.6
        gradcam_jet_resized = cv2.resize(cv2.cvtColor(gc_jet,cv2.COLOR_BGR2RGB), (gui.ori_image.shape[1], gui.ori_image.shape[0]))
        blended = cv2.addWeighted(gui.ori_image, 1- alpha, gradcam_jet_resized, alpha, 0)

        plot_qimage(blended,gui.gradcam_image_label)
        plot_text_prob(gui.gc_result_cls,prob,cls)

def plot_text_prob(qlabel,prob,cls):
    qlabel.setText(f'{prob*100:.1f}% {cls} class')

def upload_model(gui):
    file_dialog = QFileDialog()
    # TODO: tensorflow model 포함
    file_path, _ = file_dialog.getOpenFileName(gui, '모델 업로드', '', 'CNN model (*.pth *.pt *.plk )')
    if file_path:
        setModelPath(gui,file_path)

def upload_json(gui):
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(gui, 'json 업로드', '', 'Json(*.json)')
    if file_path:
        setJson(gui,file_path)
        
def apply_model(gui):
    if gui.model_load_text.isModified():
        setModelPath(gui,gui.model_load_text.text())
        
    if checkModelPath(gui) :
        appliedText(gui)
        applyModel(gui)
        
def clear_text(gui):
    if not gui.model_load_text.isEnabled():
        gui.model_load_text.setDisabled(False)
        gui.model_load_text.setText('')
        clearModelPath(gui)
    if gui.model_load_text.text():
        gui.model_load_text.setText('')
        
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
    
###################

def applyModel(gui):
    try:
        gui.gradcam_pipeline.load_model(gui.model_path)
    except:
        gui.msg_box.information(gui.msg_box, 'Error', 'Failed load model')
        clear_text(gui)
        
def appliedText(gui):
    gui.model_load_text.setDisabled(True)
    model_name = str(gui.model_path).split('\\')[-1] # Windows
    model_name = str(gui.model_path).split('/')[-1]  # Linux        
    gui.model_name_text.setText('current model: ' + model_name)
    gui.model_load_text.setStyleSheet("""
        QLineEdit {
            color: #030303;
        }
    """)

def setModelPath(gui,file_path):
    gui.model_load_text.setText(file_path)
    gui.model_path = file_path
    gui.model_load_text.setStyleSheet("""
        QLineEdit {
            color: #FFFFFF;
        }
    """)
    
def setJson(gui,file_path):
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)

    class_dict = {}
    for key, value in data.items():
        class_dict[key] = value[1]

    gui.gradcam_pipeline.class_dict = class_dict
    gui.load_json_btn.setText('Json 불러오기 [o]')
    
def clearModelPath(gui):
    gui.model_name_text.setText('current model : ')
    gui.model_path = None
    
def checkModelPath(gui):
    if gui.model_path is None:
        gui.msg_box.information(gui.msg_box, 'Error', 'Not exist selected model')
        return False
    elif not os.path.isfile(gui.model_path):
        gui.msg_box.information(gui.msg_box, 'Error', 'Invalid file path')
        return False
    else:
        return True
    