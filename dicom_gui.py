import sys
import cv2, pydicom, scipy
from numpy.core.fromnumeric import shape
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets
from PyQt5 import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from keras.models import Model, load_model
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input



class Pydicom(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        self.dcmpaths = []
        self.dcmData = []
        self.dcmImg = []
        
        self.imageMatrix = []
        self.axImg = []
        self.crImg = []
        self.sgImg = []
        
    def initUI(self):
        
        self.statusBar()
        
        # 위젯 생성
        
        self.menu = self.menuBar()   # 메뉴 생성
        self.menu_flie = self.menu.addMenu("File")        # 그룹 생성
        self.menu_edit = self.menu.addMenu("Edit")        # 그룹 생성
        
        self.file_road = QAction("Load", self)       # 메뉴 객체 생성
        self.file_road.setStatusTip("파일을 불러옵니다.")
        self.file_road.triggered.connect(self.load_dicom)
        
        self.file_save = QMenu("Save as", self)    # 메뉴 객체 생성
        self.file_save.setStatusTip("파일을 저장합니다.")
        
        self.edit_valLo = QAction("Value && Location", self)     # 메뉴 객체 생성
        self.edit_valLo.setStatusTip("이미지의 단면, 명암비를 조절합니다.")
        self.edit_valLo.triggered.connect(self.slider_in)
        
        self.edit_tumor = QAction("Tumor", self)      # 메뉴 객체 생성
        self.edit_tumor.setStatusTip("종양여부를 확인합니다.")
        self.edit_tumor.triggered.connect(self.tumor_start)
        
        self.sg_save = QAction("Sagittal", self)    # Save as 메뉴 추가
        self.sg_save.triggered.connect(self.sg_img_save)
        
        self.ax_save = QAction("Axial", self)       # Save as 메뉴 추가
        self.ax_save.triggered.connect(self.ax_img_save)
        
        self.cr_save = QAction("Coronal", self)     # Save as 메뉴 추가
        self.cr_save.triggered.connect(self.cr_img_save)
        
        self.file_save.addAction(self.sg_save)
        self.file_save.addAction(self.ax_save)
        self.file_save.addAction(self.cr_save)
        
        self.menu_flie.addAction(self.file_road)        # 메뉴 등록
        self.menu_flie.addMenu(self.file_save)          # 메뉴 등록
        
        self.menu_edit.addAction(self.edit_valLo)       # 메뉴 등록
        self.menu_edit.addAction(self.edit_tumor)       # 메뉴 등록
        
        
        self.mainWidget = MainWidget()                  # 메인 위젯 객체 생성
        
        self.setCentralWidget(self.mainWidget)          # 메인 윈도우에 메인 위젯 설정 
        self.setWindowTitle("DICOM viewer")             # 윈도우 이름 설정
        self.resize(1200, 700)                          # 윈도우 사이즈 설정
        self.show()
        
    def closeEvent(self, QCloseEvent):
        
        # 종료 확인창 생성
        ans = QMessageBox.question(self, "종료 확인", "종료하시겠습니까?", 
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ans == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()
            
    def load_dicom(self):
        self.resize(1200, 700)                  # 사이즈 재설정
        self.mainWidget.changeLayout2()         # 레이아웃 변경
        
        # 파일 불러오기
        self.dcmpaths,_ = QFileDialog.getOpenFileNames(self, "불러올 DICOM파일을 선택해 주세요")
        
        if not self.dcmpaths or self.dcmpaths == '':
            print("SisDICOM >> ------------- Files are not seleted!!!")
            return
    
        print("SisDICOM >> The first selected DICOM file name: {}".format(self.dcmpaths[0]))
        
        self.mainWidget.changeLayout0()         # 화면 초기화
        
        # 변수 초기화
        dcmFiles = []
        self.dcmImg = []
        self.dcmData = []
        
        self.axImg = []
        self.crImg = []
        self.sgImg = []
        
        self.numAx = 0
        self.numCr = 0
        self.numSg = 0
        
        self.idxAx = 0
        self.idxCr = 0
        self.idxSg = 0
        
        # DICOM 파일 읽기 및 오류 확인
        try:
            for dcmpath in self.dcmpaths:
                dcmFiles.append(pydicom.read_file(dcmpath))
        except FileExistsError as err:
            print("SisDICOM >> ------------- Error occurs during DICOM data reading!!", err)
        else:
            print("SisDICOM >> ------------- {} DICOM images are read!!!".format(len(dcmFiles)))
            
        [self.dcmImg.append(dcmFile.pixel_array) for dcmFile in dcmFiles]
        self.dcmData = dcmFiles[0].data_element
        
        # 축 길이 구하기
        size_z = len(dcmFiles)
        size_x, size_y = self.dcmImg[0].shape
        
        # DICOM 파일 설정 불러오기
        iop = dcmFiles[0].ImageOrientationPatient
        iop_round = [round(x) for x in iop]
        plane = np.cross(iop_round[0:3], iop_round[3:6])
        plane = [abs(x) for x in plane]
        
        #sagital
        if plane[0] == 1: 
            
            self.imageMatrix = np.array(self.dcmImg)
            
            # 이미지의 축 구하기
            self.numAx = size_y;
            self.numCr = size_x;
            self.numSg = size_z;
            
            self.idxAx = int(size_y/2)
            self.idxCr = int(size_x/2)
            self.idxSg = int(size_z/2)
                       
            # 픽셀 데이터 numpy 변환
            for y in range(size_y):
                self.crImg.append(np.rot90(self.imageMatrix[:,:,y], -1))
                
            for x in range(size_x):
                self.axImg.append(np.rot90(self.imageMatrix[:,x,:], -1)) 
            
            self.axImg = np.array(self.axImg)
            self.crImg = np.array(self.crImg)
            self.sgImg = self.imageMatrix
            
            
            
            # 이미지 정보
            fname = dcmFiles[0].PatientID
            size = "{0}x{1}x{2}".format(size_z, size_x, size_y)
            ori = "Sagital, " + dcmFiles[0].PatientPosition
            fov = "{0}x{1}".format(size_x, size_y)
            
            self.mainWidget.getInfo(fname, size, ori, fov)
            
            # 이미지 정보 넘기기
            self.mainWidget.getAxImg(self.axImg)
            self.mainWidget.getCrImg(self.crImg)
            self.mainWidget.getSgImg(self.sgImg)
            
            # 변수 넘기기
            self.mainWidget.getNum(self.numAx, self.numCr, self.numSg)
            
            # 이미지 그리기
            self.mainWidget.doImage1()
            self.mainWidget.doImage2()
            self.mainWidget.doImage3()
        
        #coronal    
        elif plane[1] == 1: 
            
            self.imageMatrix = np.array(self.dcmImg)
            
            # 이미지의 축 구하기
            self.numAx = size_y;
            self.numCr = size_z;
            self.numSg = size_x;
            
            self.idxAx = int(size_y/2)
            self.idxCr = int(size_z/2)
            self.idxSg = int(size_x/2)
            
            # 픽셀 데이터 numpy 변환            
            for y in range(size_y):
                self.sgImg.append(np.rot90(self.imageMatrix[:,:,y], -1))
                
            for x in range(size_x):
                self.axImg.append(np.rot90(self.imageMatrix[:,x,:])) 
            
            self.crImg = self.imageMatrix
            self.sgImg = np.array(self.sgImg)
            self.axImg = np.array(self.axImg)
            
            
            
            # 이미지 정보
            fname = dcmFiles[0].PatientID
            size = "{0}x{1}x{2}".format(size_z, size_x, size_y)
            ori = "Coronal, " + dcmFiles[0].PatientPosition
            fov = "{0}x{1}".format(size_x, size_y)
            
            self.mainWidget.getInfo(fname, size, ori, fov)
            
            # 이미지 정보 넘기기
            self.mainWidget.getAxImg(self.axImg)
            self.mainWidget.getCrImg(self.crImg)
            self.mainWidget.getSgImg(self.sgImg)
            
            # 변수 넘기기
            self.mainWidget.getNum(self.numAx, self.numCr, self.numSg)
            
            # 이미지 그리기
            self.mainWidget.doImage1()
            self.mainWidget.doImage2()
            self.mainWidget.doImage3()
        
        #Axial    
        elif plane[2] == 1: 
            
            self.imageMatrix = np.array(self.dcmImg)
            
            # 이미지의 축 구하기
            self.numSg = size_x;
            self.numAx = size_z;
            self.numCr = size_y;
            
            self.idxSg = int(size_x/2)
            self.idxAx = int(size_z/2)
            self.idxCr = int(size_y/2)
            
            # 픽셀 데이터 numpy 변환           
            for y in range(size_y):
                self.sgImg.append(np.rot90(self.imageMatrix[:,:,y], -2))
                
            for x in range(size_x):
                self.crImg.append(np.rot90(self.imageMatrix[:,x,:], -2)) 
            
            self.crImg = np.array(self.crImg)
            self.sgImg = np.array(self.sgImg)
            self.axImg = self.imageMatrix
            
            
            # 이미지 정보
            fname = dcmFiles[0].PatientID
            size = "{0}x{1}x{2}".format(size_z, size_x, size_y)
            ori = "Axial, " + dcmFiles[0].PatientPosition
            fov = "{0}x{1}".format(size_x, size_y)
            
            self.mainWidget.getInfo(fname, size, ori, fov)
            
            # 이미지 정보 넘기기
            self.mainWidget.getAxImg(self.axImg)
            self.mainWidget.getCrImg(self.crImg)
            self.mainWidget.getSgImg(self.sgImg)
            
            # 변수 넘기기
            self.mainWidget.getNum(self.numAx, self.numCr, self.numSg)
            
            # 이미지 그리기
            self.mainWidget.doImage1()
            self.mainWidget.doImage2()
            self.mainWidget.doImage3()
                
    def slider_in(self):
        if not self.dcmpaths or self.dcmpaths == '':
            print("None Data")
        else:
            self.resize(1200, 700)
            self.mainWidget.changeLayout2()
            # 슬라이더 생성
            self.mainWidget.slider_init()
        
    def tumor_start(self):
        # 사이즈 재설정 및 레이아웃 변경
        self.resize(1000, 700)
        self.mainWidget.changeLayout1()    
            
    def sg_img_save(self):
        # Sagittal 이미지 저장
        self.mainWidget.sg_save()
        
    def ax_img_save(self):
        # Axial 이미지 저장
        self.mainWidget.ax_save()
        
    def cr_img_save(self):
        # Cronal 이미지 저장
        self.mainWidget.cr_save()
        
        
class MainWidget(QWidget): 
    def __init__(self):
        super().__init__() 
        self.initUI()
        
        # AI 불러오기
        self.sg_model = load_model('model_sagital.h5')
        self.ax_model = load_model('model_axial.h5')
        self.cr_model = load_model('model_coronal.h5')
        
        self.axImg = []
        self.crImg = []
        self.sgImg = []
        
        self.axImg_copy = []
        self.crImg_copy = []
        self.sgImg_copy = []
        
        self.numAx = 0
        self.numCr = 0
        self.numSg = 0
        
        self.contAx = 256
        self.contSg = 256
        self.contCr = 256
        
    def initUI(self):
        
        self.main_layout = QHBoxLayout()
        
        #==================================Info Layout========================================
        
        # 레이블의 폭과 높이 값
        lblWidth = 330
        lblHeight = 40
        lblHalfWidth = 150
        
        # 정보 창의 메인 layout
        self.info_box = QVBoxLayout()
        self.info_box.setAlignment(Qt.AlignTop)
        
        # 파일 이름을 띄워주는 레이블
        self.lbl_name = QLabel("File name", self)
        self.lbl_name.setFixedSize(lblWidth, lblHeight)
        self.lbl_name.setStyleSheet("border-style: hidden;"
                                    "font: bold 12px;"
                                    "border-width: 2px;")
        
        self.lbl_fname = QLabel("None", self)
        self.lbl_fname.setFixedSize(lblWidth, lblHeight)
        self.lbl_fname.setStyleSheet("background-color: #FFFFFF;"
                                    "padding: 10px;"
                                    "border-style: solid;"
                                    "border-width: 2px;"
                                    "border-color: #D6DCE5;"
                                    "border-radius: 5px")
        
        # 파일 이름 layout
        self.name_box = QVBoxLayout()
        self.name_box.addWidget(self.lbl_name) 
        self.name_box.addWidget(self.lbl_fname)
        self.info_box.addLayout(self.name_box)
        
        
        # 각각의 파일 정보를 넣어줄 layout
        self.in1 = QHBoxLayout()
        self.in2 = QHBoxLayout() 
        self.in3 = QHBoxLayout()
        
        # 파일 정보을 띄워주는 레이블
        self.lbl_img_info = QLabel("Image Information", self)
        self.lbl_img_info.setFixedSize(lblWidth, lblHeight)
        self.lbl_img_info.setStyleSheet("border-style: hidden;"
                                        "font: bold 12px;"
                                        "border-width: 2px;")
        
        self.lbl_szie = QLabel("Size:", self)
        self.lbl_szie.setFixedSize(lblHalfWidth, lblHeight)
        self.lbl_szie.setStyleSheet("border-style: hidden;"
                                    "padding: 0px;"
                                    "border-width: 2px;")
        
        self.lbl_szie_info = QLabel("None", self)
        self.lbl_szie_info.setAlignment(Qt.AlignRight)
        self.lbl_szie_info.setFixedSize(lblHalfWidth, lblHeight)
        self.lbl_szie_info.setStyleSheet("border-style: hidden;"
                                        "padding: 10px;"
                                        "border-width: 2px;")
        self.in1.addWidget(self.lbl_szie)
        self.in1.addWidget(self.lbl_szie_info)
        
        self.lbl_ori = QLabel("Orientation:", self)
        self.lbl_ori.setFixedSize(lblHalfWidth, lblHeight)
        self.lbl_ori.setStyleSheet( "border-style: hidden;"
                                    "padding: 0px;"
                                    "border-width: 2px;")
        
        self.lbl_ori_info = QLabel("None", self)
        self.lbl_ori_info.setAlignment(Qt.AlignRight)
        self.lbl_ori_info.setFixedSize(lblHalfWidth, lblHeight)
        self.lbl_ori_info.setStyleSheet("border-style: hidden;"
                                        "padding: 10px;"
                                        "border-width: 2px;")
        self.in2.addWidget(self.lbl_ori)
        self.in2.addWidget(self.lbl_ori_info)
        
        self.lbl_FOV = QLabel("FOV:", self)
        self.lbl_FOV.setFixedSize(lblHalfWidth, lblHeight)
        self.lbl_FOV.setStyleSheet( "border-style: hidden;"
                                    "padding: 0px;"
                                    "border-width: 2px;")
        
        self.lbl_FOV_info = QLabel("None", self)
        self.lbl_FOV_info.setAlignment(Qt.AlignRight)
        self.lbl_FOV_info.setFixedSize(lblHalfWidth, lblHeight)
        self.lbl_FOV_info.setStyleSheet("border-style: hidden;"
                                        "padding: 10px;"
                                        "border-width: 2px;")
        
        self.in3.addWidget(self.lbl_FOV)
        self.in3.addWidget(self.lbl_FOV_info)
        
        # 파일 정보의 메인 layout
        self.img_info_box = QVBoxLayout()
        self.img_info_box2 = QVBoxLayout()
        
        self.img_info_box2.addLayout(self.in1)
        self.img_info_box2.addLayout(self.in2)
        self.img_info_box2.addLayout(self.in3)
        
        self.infoFrame = QtWidgets.QFrame()
        self.infoFrame.setLayout(self.img_info_box2)
        self.infoFrame.resize(lblWidth, lblHeight*3)
        self.infoFrame.setStyleSheet("background-color: #FFFFFF;"
                                    "padding: 0px;"
                                    "border-style: solid;"
                                    "border-width: 2px;"
                                    "border-color: #D6DCE5;"
                                    "border-radius: 5px")
        
        self.img_info_box.addWidget(self.lbl_img_info)
        self.img_info_box.addWidget(self.infoFrame)
        
        # 상단 layout과 구별을 위한 마진
        self.img_info_box.setContentsMargins(0,20,0,0)
        self.info_box.addLayout(self.img_info_box)
        
        # 정보 창의 메인 layout
        self.tumor_box = QVBoxLayout()
        self.tumor_box.setAlignment(Qt.AlignTop)
        
        # tumor 정보를 띄워주는 레이블
        self.lbl_tumor = QLabel("Tumor", self)
        self.lbl_tumor.setFixedSize(lblWidth, lblHeight)
        self.lbl_tumor.setStyleSheet("border-style: hidden;"
                                    "font: bold 12px;"
                                    "border-width: 2px;")
        
        self.lbl_state = QLabel("None", self)
        self.lbl_state.setStyleSheet("background-color: #FFFFFF;"
                                     "padding: 0px;"
                                     "border-style: solid;"
                                     "border-width: 0px;"
                                     "border-radius: 0px")
        
        # tumor 콤보 박스 생성
        self.tumorCombo = QtWidgets.QComboBox(self)
        self.tumorCombo.addItem("sagital")
        self.tumorCombo.addItem("axial")
        self.tumorCombo.addItem("coronal")
        self.tumorCombo.resize(50, 30)
        self.tumorCombo.setStyleSheet("background-color: #FFFFFF;"
                                     "padding: 0px;"
                                     "border-style: solid;"
                                     "border-width: 2px;"
                                     "border-color: #D6DCE5;"
                                     "border-radius: 0px")
        
        self.tumorLabelLayout = QHBoxLayout(self)
        self.tumorLabelLayout.addWidget(self.lbl_state)
        self.tumorLabelLayout.addWidget(self.tumorCombo)
        
        
        self.tumorLabelFrame = QtWidgets.QFrame(self)
        self.tumorLabelFrame.setLayout(self.tumorLabelLayout)
        self.tumorLabelFrame.setStyleSheet("background-color: #FFFFFF;"
                                     "padding: 0px;"
                                     "border-style: solid;"
                                     "border-width: 2px;"
                                     "border-color: #D6DCE5;"
                                     "border-radius: 5px")
        
        
        self.tumor_box.addWidget(self.lbl_tumor)
        self.tumor_box.addWidget(self.tumorLabelFrame)
        
        self.infoTumorFrame = QtWidgets.QFrame(self)
        self.infoTumorFrame.setLayout(self.tumor_box)
        
        # 상단 layout과 구별을 위한 마진
        self.tumor_box.setContentsMargins(0,20,0,0)
        self.info_box.addWidget(self.infoTumorFrame)
        
        self.infoTumorFrame.hide()
        
        
        #==================================Image Layout========================================
        
        self.img_grid = QGridLayout()
        
        # 캔버스 레이아웃 생성
        self.canvas_layout1 = QVBoxLayout()
        self.canvas_layout2 = QVBoxLayout()
        self.canvas_layout3 = QVBoxLayout()
        
        # matplotlib figure 설정
        self.fig1 = plt.Figure()
        self.fig2 = plt.Figure()
        self.fig3 = plt.Figure()
        
        self.fig1.set_facecolor("black")
        self.fig2.set_facecolor("black")
        self.fig3.set_facecolor("black")
        
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas3 = FigureCanvas(self.fig3)
        
        # location slider 생성
        self.slider1 = QSlider(Qt.Horizontal, self)
        self.slider2 = QSlider(Qt.Horizontal, self)
        self.slider3 = QSlider(Qt.Horizontal, self)
        
        # value slider 생성
        self.slider1_1 = QSlider(Qt.Horizontal, self)
        self.slider2_1 = QSlider(Qt.Horizontal, self)
        self.slider3_1 = QSlider(Qt.Horizontal, self)
        
        self.canvas_layout1.addWidget(self.canvas1)
        self.canvas_layout1.addWidget(self.slider1)
        self.canvas_layout1.addWidget(self.slider1_1)
        
        self.canvas_layout2.addWidget(self.canvas2)
        self.canvas_layout2.addWidget(self.slider2)
        self.canvas_layout2.addWidget(self.slider2_1)
        
        self.canvas_layout3.addWidget(self.canvas3)
        self.canvas_layout3.addWidget(self.slider3)
        self.canvas_layout3.addWidget(self.slider3_1)
        
        self.img_grid.addLayout(self.canvas_layout1, 0, 0)
        self.img_grid.addLayout(self.canvas_layout2, 0, 1)
        self.img_grid.addLayout(self.canvas_layout3, 1, 0)
        
        # location slider 이벤트
        self.slider1.valueChanged.connect(self.sliderChange1)
        self.slider2.valueChanged.connect(self.sliderChange2)
        self.slider3.valueChanged.connect(self.sliderChange3)
        
        # value slider 이벤트
        self.slider1_1.valueChanged.connect(self.sliderChange1_1)
        self.slider2_1.valueChanged.connect(self.sliderChange2_1)
        self.slider3_1.valueChanged.connect(self.sliderChange3_1)
        
        self.slider1.setVisible(False)
        self.slider2.setVisible(False)
        self.slider3.setVisible(False)
        
        self.slider1_1.setVisible(False)
        self.slider2_1.setVisible(False)
        self.slider3_1.setVisible(False)
        
        
        self.tumor_canvas = QVBoxLayout()
        
        self.figTumor = plt.Figure()
        self.figTumor.set_facecolor("black")
        
        self.canvas4 = FigureCanvas(self.figTumor)
        
        self.tumor_canvas.addWidget(self.canvas4)
        
        self.gridFrame = QtWidgets.QFrame()
        self.gridFrame.setLayout(self.img_grid)
        
        self.imageFrame = QtWidgets.QFrame()
        self.imageFrame.setLayout(self.tumor_canvas)
        
        self.main_layout.addLayout(self.info_box)
        self.main_layout.addWidget(self.gridFrame)
        self.main_layout.addWidget(self.imageFrame)
        
        # AI 선택 콤보 박스
        self.tumorCombo.activated[str].connect(self.comboChange)
        
        self.imageFrame.hide()
        
        self.setLayout(self.main_layout)
        self.show()

    def sg_save(self):
        
        # Sagittal 이미지 저장
        if len(self.sgImg_copy) != 0 :
            fileSave = QFileDialog.getSaveFileName(self, 'Save Image', '.jpg')
            
            if not fileSave or fileSave == '':
                print("SisDICOM >> ------------- Files are not seleted!!!")
            else:
                cv2.imwrite(fileSave[0], self.sgImg_copy[self.idxSg])

    def ax_save(self):
        
        # Axial 이미지 저장
        if len(self.axImg_copy) != 0 :
            fileSave = QFileDialog.getSaveFileName(self, 'Save Image', '.jpg')
            
            if not fileSave or fileSave == '':
                print("SisDICOM >> ------------- Files are not seleted!!!")
            else:
                cv2.imwrite(fileSave[0], self.axImg_copy[self.idxAx])
        
    def cr_save(self):
        
        # Coronal 이미지 저장
        if len(self.crImg_copy) != 0 :
            fileSave = QFileDialog.getSaveFileName(self, 'Save Image', '.jpg')
            
            if not fileSave or fileSave == '':
                print("SisDICOM >> ------------- Files are not seleted!!!")
            else:
                cv2.imwrite(fileSave[0], self.crImg_copy[self.idxCr])

    def getAxImg(self, axImg):
        # Axial 이미지 불러오기
        self.axImg = axImg
        self.axImg_copy = axImg
        
    def getCrImg(self, crImg):
        # Coronal 이미지 불러오기
        self.crImg = crImg
        self.crImg_copy = crImg

    def getSgImg(self, sgImg):
        # Sagittal 이미지 불러오기
        self.sgImg = sgImg
        self.sgImg_copy = sgImg
    
    def getNum(self, numAx, numCr, numSg):
        # 이미지 축 불러오기
        self.numAx = numAx
        self.numCr = numCr
        self.numSg = numSg
        self.idxAx = int(numAx/2)
        self.idxSg = int(numSg/2)
        self.idxCr = int(numCr/2)
    
    def doImage1(self):
        
        # 캔버스 초기화
        self.fig1.clear()
        
        # Coronal 이미지 그리기
        ax = self.fig1.add_subplot(111)
        ax.imshow(self.crImg_copy[self.idxCr], cmap = 'gray')
        
        # 선 그리기
        ax.plot([0, self.numSg], [self.numAx/2, self.numAx/2], 'y')
        ax.plot([self.numSg/2, self.numSg/2], [0, self.numAx], 'y')
        
        # 텍스트 표기
        ax.text(0, self.numAx/2 - 10, 'L', fontsize = 12, color = 'yellow')
        ax.text(self.numSg/2 + 10, 10, 'S', fontsize = 12, color = 'yellow')
        ax.axis('off')
        
        self.canvas1.draw()
        
    def doImage2(self):
        
        # 캔버스 초기화
        self.fig2.clear()
        
        # Sagittal 이미지 그리기
        ax = self.fig2.add_subplot(111)
        ax.imshow(self.sgImg_copy[self.idxSg], cmap = 'gray')
        
        # 선 그리기
        ax.plot([0, self.numCr], [self.numAx/2, self.numAx/2], 'y')
        ax.plot([self.numCr/2, self.numCr/2], [0, self.numAx], 'y')
        
        # 텍스트 표기
        ax.text(0, self.numAx/2 - 10, 'A', fontsize = 12, color = 'yellow')
        ax.text(self.numCr/2 + 10, 10, 'S', fontsize = 12, color = 'yellow')
        ax.axis('off')
        
        self.canvas2.draw()
        
    def doImage3(self):
        
        # 캔버스 초기화
        self.fig3.clear()
        
        # Axial 이미지 그리기
        ax = self.fig3.add_subplot(111)
        ax.imshow(self.axImg_copy[self.idxAx], cmap = 'gray')
        
        # 선 그리기
        ax.plot([0, self.numSg], [self.numCr/2, self.numCr/2], 'y')
        ax.plot([self.numSg/2, self.numSg/2], [0, self.numCr], 'y')
        
        # 텍스트 표기
        ax.text(0, self.numCr/2 - 10, 'L', fontsize = 12, color = 'yellow')
        ax.text(self.numSg/2 + 10, 10, 'A', fontsize = 12, color = 'yellow')
        ax.axis('off')
        
        self.canvas3.draw()
    
    def doTumorImage(self, position):
        
        # 캔버스 초기화
        self.figTumor.clear()
        ax = self.figTumor.add_subplot(111)
        
        if len(self.sgImg) != 0 or len(self.axImg) != 0 or len(self.crImg) != 0 :
            
            # Sagittal
            if position == 0:
                
                last_weight = self.sg_model.layers[-1].get_weights()[0] # (1280, 2)
                
                # 새로운 아웃풋 모델 생성
                new_model = Model(  inputs=self.sg_model.input,
                                    outputs=(self.sg_model.layers[-3].output, 
                                            self.sg_model.layers[-1].output))
                
                # 이미지 전처리
                res = cv2.resize(self.sgImg_copy[self.idxSg], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                les = np.repeat(res, 3)
                les = les.reshape(224,224,3)
                test_input = preprocess_input(np.expand_dims(les.copy(), axis=0)) # 사진 전처리
                
                # AI 예측
                last_conv_output, pred = new_model.predict(test_input)
                last_conv_output = np.squeeze(last_conv_output) # (7,7, 1280)
                feature_activation_maps = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) 
                # (7,7,1280) -> (224,224,1280)
                
                pred_class = np.argmax(pred)
                predicted_class_weights = last_weight[:, pred_class] # (1280, 1)
                
                final_output = np.dot(feature_activation_maps.reshape((224*224,1280)), predicted_class_weights).reshape(224,224)
                # (224*224, 1280) dot_product(1280, 1) = (224*224, 1)
                
                # 예측값 표기
                self.lbl_state.setText('%.2f%% Tumor' % (pred[0][1]*100))
                
                ax.imshow(res, cmap = 'gray')   # 원래 이미지
                ax.imshow(final_output, cmap='jet', alpha=0.5)  # 예측 이미지
                ax.axis('off')
            
            # Axial    
            elif position == 1:
                
                last_weight = self.ax_model.layers[-1].get_weights()[0] # (1280, 2)
                
                # 새로운 아웃풋 모델 생성
                new_model = Model(  inputs=self.ax_model.input,
                                    outputs=(self.ax_model.layers[-3].output, 
                                            self.ax_model.layers[-1].output))
                
                # 이미지 전처리
                res = cv2.resize(self.axImg_copy[self.idxAx], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                les = np.repeat(res, 3)
                les = les.reshape(224,224,3)
                test_input = preprocess_input(np.expand_dims(les.copy(), axis=0)) # 사진 전처리
                
                # AI 예측
                last_conv_output, pred = new_model.predict(test_input)
                last_conv_output = np.squeeze(last_conv_output) # (7,7, 1280)
                feature_activation_maps = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) 
                # (7,7,1280) -> (224,224,1280)
                
                pred_class = np.argmax(pred)
                predicted_class_weights = last_weight[:, pred_class] # (1280, 1)
                
                final_output = np.dot(feature_activation_maps.reshape((224*224,1280)), predicted_class_weights).reshape(224,224)
                # (224*224, 1280) dot_product(1280, 1) = (224*224, 1)
                
                self.lbl_state.setText('%.2f%% Tumor' % (pred[0][1]*100))
                
                ax.imshow(res, cmap = 'gray')   # 원래 이미지
                ax.imshow(final_output, cmap='jet', alpha=0.5)  # 예측 이미지
                ax.axis('off')
            
            # Coronal    
            elif position == 2:
                
                last_weight = self.cr_model.layers[-1].get_weights()[0] # (1280, 2)
                
                # 새로운 아웃풋 모델 생성
                new_model = Model(  inputs=self.cr_model.input,
                                    outputs=(self.cr_model.layers[-3].output, 
                                            self.cr_model.layers[-1].output))
                
                # 이미지 전처리
                res = cv2.resize(self.crImg_copy[self.idxCr], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                les = np.repeat(res, 3)
                les = les.reshape(224,224,3)
                test_input = preprocess_input(np.expand_dims(les.copy(), axis=0)) # 사진 전처리
                
                # AI 예측
                last_conv_output, pred = new_model.predict(test_input)
                last_conv_output = np.squeeze(last_conv_output) # (7,7, 1280)
                feature_activation_maps = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) 
                # (7,7,1280) -> (224,224,1280)
                
                pred_class = np.argmax(pred)
                predicted_class_weights = last_weight[:, pred_class] # (1280, 1)
                
                final_output = np.dot(feature_activation_maps.reshape((224*224,1280)), predicted_class_weights).reshape(224,224)
                # (224*224, 1280) dot_product(1280, 1) = (224*224, 1)
                
                self.lbl_state.setText('%.2f%% Tumor' % (pred[0][1]*100))
                
                ax.imshow(res, cmap = 'gray')   # 원래 이미지
                ax.imshow(final_output, cmap='jet', alpha=0.5)  # 예측 이미지
                ax.axis('off')
        
            self.canvas4.draw()
    
    def comboChange(self):
        # 콤보박스 이벤트
        self.doTumorImage(self.tumorCombo.currentIndex())
        
    def slider_init(self):
        
        # slider 초기 설정
        self.slider1.setRange(0, self.numCr-1)
        self.slider1.setValue(self.idxCr)
        
        self.slider2.setRange(0, self.numSg-1)
        self.slider2.setValue(self.idxSg)
        
        self.slider3.setRange(0, self.numAx-1)
        self.slider3.setValue(self.idxAx)
        
        self.slider1_1.setRange(0, 511)
        self.slider1_1.setValue(256)
        
        self.slider2_1.setRange(0, 511)
        self.slider2_1.setValue(256)
        
        self.slider3_1.setRange(0, 511)
        self.slider3_1.setValue(256)
        
        self.slider1.setVisible(True)
        self.slider2.setVisible(True)
        self.slider3.setVisible(True)
        self.slider1_1.setVisible(True)
        self.slider2_1.setVisible(True)
        self.slider3_1.setVisible(True)
                                                                                                                                                   
    def sliderChange1(self):
        # Coronal location 이벤트
        self.idxCr = self.slider1.value()
        self.doImage1()
        
    def sliderChange2(self):
        # Sagittal location 이벤트
        self.idxSg = self.slider2.value()
        self.doImage2()
    
    def sliderChange3(self):
        # Axial location 이벤트
        self.idxAx = self.slider3.value()
        self.doImage3()
     
    def sliderChange1_1(self):
        # Coronal value 이벤트
        self.contCr = self.slider1_1.value()
        self.crImg_copy = self.saturate_contrast1(self.crImg, self.contCr/256)
        
        self.doImage1()
        
    def sliderChange2_1(self):
        # Sagittal value 이벤트
        self.contSg = self.slider2_1.value()
        self.sgImg_copy = self.saturate_contrast1(self.sgImg, self.contSg/256)
        
        self.doImage2()
    
    def sliderChange3_1(self):
        # Axial value 이벤트
        self.contAx = self.slider3_1.value()
        self.axImg_copy = self.saturate_contrast1(self.axImg, self.contAx/256)
        
        self.doImage3()
        
    def getInfo(self, fname, size, ori, fov):
        # DICOM 파일 정보 불러오기
        self.lbl_fname.setText(fname)
        self.lbl_szie_info.setText(size)
        self.lbl_ori_info.setText(ori)
        self.lbl_FOV_info.setText(fov)
        
    def saturate_contrast1(self, p, num):
        # 이미지 밝기 변화
        pic = p.copy()
        pic = pic.astype('uint16')
        pic = np.clip(pic*num, 0, 255)
        return pic
    
    def saturate_contrast2(self, p, num):
        # 이미지 대조도 변화
        pic = p.copy()
        pic = pic.astype('uint16')
        pic = np.clip((1+num)*pic - 128*num, 0, 255)
        return pic
    
    def changeLayout0(self):
        # basic 레이아웃으로 전환
        self.fig1.clear()
        self.fig2.clear()
        self.fig3.clear()
        self.slider1.hide()
        self.slider2.hide()
        self.slider3.hide()
        self.slider1_1.hide()
        self.slider2_1.hide()
        self.slider3_1.hide()
    
    def changeLayout1(self):
        # AI 레이아웃으로 전환
        self.gridFrame.hide()
        self.imageFrame.show()
        self.infoTumorFrame.show()
        
    def changeLayout2(self):
        # value 레이아웃으로 전환
        self.imageFrame.hide()
        self.gridFrame.show()
        self.infoTumorFrame.hide()
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Pydicom()
    sys.exit(app.exec_())