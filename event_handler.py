import os
import cv2
import numpy as np
from PyQt5 .QtWidgets import ( QFileDialog,QMessageBox)
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
    if gui.file_path is None:
        return None
    result = gui.gradcam.run_pipeline(gui.file_path)
    result = (result * 255).astype(np.uint8)
    gc_jet = cv2.applyColorMap(result, cv2.COLORMAP_JET)
    
    alpha = 0.6
    gradcam_jet_resized = cv2.resize(cv2.cvtColor(gc_jet,cv2.COLOR_BGR2RGB), (gui.ori_image.shape[1], gui.ori_image.shape[0]))
    blended = cv2.addWeighted(gui.ori_image, 1- alpha, gradcam_jet_resized, alpha, 0)

    plot_qimage(blended,gui.gradcam_image_label)

def setModelPath(gui,file_path):
    gui.model_load_text.setText(file_path)
    gui.model_path = file_path

def upload_model(gui):
    file_dialog = QFileDialog()
    # TODO: tensorflow model 포함
    file_path, _ = file_dialog.getOpenFileName(gui, '모델 업로드', '', 'CNN model (*.pth *.pt *.plk )')
    if file_path:
        setModelPath(gui,file_path)

def checkModelPath(gui):
    if gui.model_path is None:
        QMessageBox.information(gui, 'Error', 'Not exist file')
        return False
    elif not os.path.isfile(gui.model_path):
        QMessageBox.information(gui, 'Error', 'Invalid file path')
        return False
    else:
        return True

def apply_model(gui):
    if gui.model_load_text.isModified():
        setModelPath(gui,gui.model_load_text.text())
        
    if checkModelPath(gui) :
        gui.model_load_text.setDisabled(True)
        model_name = str(gui.model_path).split('\\')[-1] # windows
        model_name = str(gui.model_path).split('/')[-1]  # linux        
        gui.model_name_text.setText('current model : '+ model_name)

def clearModelPath(gui):
    gui.model_name_text.setText('current model : ')
    gui.model_path = None
    
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