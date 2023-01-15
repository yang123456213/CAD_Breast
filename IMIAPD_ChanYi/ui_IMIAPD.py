# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_IMIAPD.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

import cv2
import numpy as np
from PyQt5.QtCore import QRectF, Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import QMessageBox, QFileDialog
from PyQt5.Qt import QThread, QMutex, pyqtSignal
from PyQt5.QtGui import QCursor, QImage, QPixmap
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVtkWidget
from PyQt5.QtWidgets import QMenu, QGraphicsPixmapItem, QGraphicsView, QAction, QGraphicsScene
from radiomics import featureextractor
from PIL import Image
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QWidget
from deep_for_seg.predict import predict
import SimpleITK as sitk
from sklearn.externals import joblib
# import joblib
import pickle

def extra_Radiomics_features(image, mask):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    # B = featureextractor.getImageTypes()
    # print(B)
    # A = featureextractor.getFeatureClasses().keys()
    extractor.enableImageTypes(Original={}, Gradient={}, LBP2D={}, LoG={},
                               Wavelet={}, Exponential={}, Square={}, SquareRoot={}, Logarithm={})
    extractor.enableAllFeatures()
    extractor.enableFeaturesByName(firstorder=[])
    extractor.enableFeaturesByName(shape2D=[])
    extractor.enableFeaturesByName(glcm=['Autocorrelation',
                                          'ClusterProminence',
                                          'ClusterShade',
                                          'ClusterTendency',
                                          'Contrast',
                                          'Correlation',
                                          'DifferenceAverage',
                                          'DifferenceEntropy',
                                          'DifferenceVariance',
                                          'Id',
                                          'Idm',
                                          'Idmn',
                                          'Idn',
                                          'Imc1',
                                          'Imc2',
                                          'InverseVariance',
                                          'JointAverage',
                                          'JointEnergy',
                                          'JointEntropy',
                                          'MCC',
                                          'MaximumProbability',
                                          'SumEntropy',
                                          'SumSquares',
                                          ])
    extractor.enableFeaturesByName(glrlm=[])
    extractor.enableFeaturesByName(glszm=[])
    extractor.enableFeaturesByName(ngtdm=[])
    extractor.enableFeaturesByName(gldm=[])

    sitk_image = sitk.GetImageFromArray(image)
    sitk_mask = sitk.GetImageFromArray(mask)

    featureVector = extractor.execute(sitk_image, sitk_mask)

    radiomics_features_names = []
    radiomics_features_values = []
    for featureName in featureVector.keys():
        if 'diagnostics' not in featureName:
            radiomics_features_names.append(featureName)  # 特征名称
            item = featureVector[featureName]  # 特征值
            radiomics_features_values.append(item)

    return radiomics_features_names, radiomics_features_values


class PrintImage(QWidget):
    def __init__(self, pixmap, parent=None):
        QWidget.__init__(self, parent=parent)
        self.pixmap = pixmap

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)

class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent=parent)
        self._zoom = 0
        self._empty = True
        self._photo = QGraphicsPixmapItem()
        self._scene = QGraphicsScene(self)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignCenter)  # 居中显示

    def contextMenuEvent(self, event):
        if not self.has_photo():
            return
        menu = QMenu()
        save_action = QAction('另存为', self)
        save_action.triggered.connect(self.save_current)  # 传递额外值
        menu.addAction(save_action)
        menu.exec(QCursor.pos())

    def save_current(self):
        file_name = QFileDialog.getSaveFileName(self, '另存为', './', 'Image files(*.jpg *.gif *.png)')[0]
        print(file_name)
        if file_name:
            self._photo.pixmap().save(file_name)

    def get_image(self):
        if self.has_photo():
            return self._photo.pixmap().toImage()

    def has_photo(self):
        return not self._empty

    def change_image(self, img):
        self.update_image(img)
        self.fitInView()

    # def img_to_pixmap(self, img, resize_flag):
    #     A = np.max(img)
    #     B = np.min(img)
    #     img_mask = img > 0
    #     img = 255 - (img-B*1.0/(A-B)) * 255
    #     img[img_mask == 0] = 0
    #     img = np.array(img, dtype=np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # bgr -> rgb
    #     h, w, c = img.shape
    #     img = QImage(img, w, h, 3 * w, QImage.Format_RGB888)
    #     return QPixmap.fromImage(img)

    def img_to_pixmap(self, img, resize_flag):
        A = np.max(img)
        img = (img * 1.0 / A) * 255
        img = np.array(img, dtype=np.uint8)
        h, w, c = img.shape
        img = QImage(img, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(img)

    def img_to_pixmap_mask(self, img, resize_flag):
        A = np.max(img)
        B = np.min(img)
        img_mask = img > 0
        img = img_mask * 255
        img[img_mask == 0] = 0
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # bgr -> rgb
        h, w, c = img.shape
        img = QImage(img, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(img)

    def img_to_pixmap_rectangle(self, img, resize_flag):
        h, w, c = img.shape
        img = QImage(img, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(img)

    def update_image_rectangle(self, img, resize_flag=False):
        self._empty = False
        self._photo.setPixmap(self.img_to_pixmap_rectangle(img, resize_flag))
        self.fitInView()

    def update_image_mask(self, img, resize_flag=False):
        self._empty = False
        self._photo.setPixmap(self.img_to_pixmap_mask(img, resize_flag))
        self.fitInView()

    def update_image(self, img, resize_flag=False):
        self._empty = False
        self._photo.setPixmap(self.img_to_pixmap(img, resize_flag))
        self.fitInView()

    def fitInView(self, scale=True):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_photo():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def wheelEvent(self, event):
        if self.has_photo():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def empty_image(self):
        self._zoom = 0
        self._empty = True
        img = np.zeros((380, 256, 1))
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # bgr -> rgb
        img_final = QImage(img, 380, 256, 3 * 128, QImage.Format_RGB888)
        self._photo.setPixmap(QPixmap.fromImage(img_final))
        self.fitInView()

def cut_img_by_rectangle(mask, img, max_len_x, max_len_y):
    mask_size = mask.shape
    return_img = np.zeros((max_len_x, max_len_y))
    first_flag = False
    first_index_x = 0
    first_index_y = 0
    for index_i in range(mask_size[0]):
        for index_j in range(mask_size[1]):
            if mask[index_i][index_j]>0:
                if first_flag==False:
                    first_index_x = index_i
                    first_index_y = index_j
                    first_flag = True
                if first_flag == True:
                    # print(index_i-first_index_x, index_j-first_index_y, index_i, index_j)
                    return_img[index_i-first_index_x][index_j-first_index_y] = img[index_i][index_j]
    return return_img

def get_rectangle(mask, boundary=0):
    mask_size = mask.shape
    min_x = mask_size[0]
    min_y = mask_size[1]
    max_x = 0
    max_y = 0
    for index_i in range(mask_size[0]):
        for index_j in range(mask_size[1]):
            if mask[index_i][index_j]>0:
                if min_x > index_i:
                    min_x = index_i
                if min_y > index_j:
                    min_y = index_j
                if max_x < index_i:
                    max_x = index_i
                if max_y < index_j:
                    max_y = index_j

    max_len_x = (max_x - min_x)
    max_len_y = (max_y - min_y)

    max_len_x = max_len_x + boundary
    max_len_y = max_len_y + boundary

    center_i_S = ((max_x+min_x)//2)-(max_len_x//2)
    center_i_E = ((max_x+min_x)//2)+(max_len_x//2)
    center_j_S = ((max_y+min_y)//2)-(max_len_y//2)
    center_j_E = ((max_y+min_y)//2)+(max_len_y//2)

    mask_final = np.zeros((mask_size[0], mask_size[1]))
    for i_center_img in range(center_i_S, center_i_E):
        for j_center_img in range(center_j_S, center_j_E):
            mask_final[i_center_img][j_center_img] = 1

    return mask_final, (min_y, min_x), (max_y, max_x), max_len_x, max_len_y

qmut_segment = QMutex()
class Thread_segment(QThread):  # 分割乳腺癌线程
    _signal = pyqtSignal()
    def __init__(self, prediction_bar, img, img_path_seg):
        super().__init__()
        self.prediction_bar = prediction_bar
        self.img = img
        self.img_path_seg = img_path_seg

    def run(self):
        qmut_segment.lock()
        seg_array = predict(self.img, './deep_for_seg/model/checkpoint1.pth')
        # time.sleep(5)
        # seg_array = self.img
        # cv2.rectangle(img, (10, 50), (50, 100), (0, 255, 0), 4)

        self.prediction_bar.seg_img = seg_array
        _, A_Point, B_Point, _, _ = get_rectangle(seg_array)

        img = self.img
        self.prediction_bar.seg_rectangle_img = cv2.rectangle(img, A_Point, B_Point, (238, 99, 100), 8)
        self.prediction_bar.msg_box.hide()

        msg_box_ = QMessageBox(QMessageBox.Question, '提示', '分割完毕，分割结果以图片形式保存到分割图像的文件夹中！', QMessageBox.Yes)
        msg_box_.resize(300, 300)
        reply = msg_box_.exec()
        if reply == QMessageBox.Yes:
            print('YES')
            mask = Image.fromarray(seg_array * 255.)
            img_path_seg_temp = self.img_path_seg.split('/')
            seg_path = self.img_path_seg.replace(img_path_seg_temp[-1], img_path_seg_temp[-1] + '_seg.gif')
            mask.save(seg_path)
        else:
            print('NO')
        self.prediction_bar.show_seg = True
        self.prediction_bar.watch_flag = 1
        qmut_segment.unlock()
        self._signal.emit()


def updata_features(index, features):
    reuren_feature = []
    print(len(features[0]))
    for i in range(len(features)):
        feature = features[i]
        featureup = []
        for j in index:
            featureup.append(feature[int(j)])
        reuren_feature.append(featureup)
    return reuren_feature

def sofmax(logits):
	e_x = np.exp(logits)
	probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
	return probs

qmut_predict = QMutex()
class Thread_predict(QThread):  # 乳腺癌级别预测线程
    _signal_p = pyqtSignal()
    def __init__(self, prediction_bar, img, mask):
        super().__init__()
        self.prediction_bar = prediction_bar
        self.img = img
        self.mask = mask

    def run(self):
        qmut_predict.lock()
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        mask_final, _, _, max_len_x, max_len_y = get_rectangle(self.mask)

        mask_rectangle = cut_img_by_rectangle(mask_final, mask_final, max_len_x, max_len_y)
        mask = cut_img_by_rectangle(mask_final, self.mask, max_len_x, max_len_y)
        cut_img_rectangle = cut_img_by_rectangle(mask_final, img, max_len_x, max_len_y)

        # image = ((cut_img_rectangle - np.mean(cut_img_rectangle))) / (np.std(cut_img_rectangle))
        image = ((cut_img_rectangle - np.min(cut_img_rectangle))) / (np.max(cut_img_rectangle) - np.min(cut_img_rectangle))

        mask_2 = mask_rectangle - mask
        _, radiomics_features_values_rectangle_2 = extra_Radiomics_features(image, mask_2)
        _, radiomics_features_values = extra_Radiomics_features(image, mask)

        features = np.concatenate((radiomics_features_values,
                                   radiomics_features_values_rectangle_2,
                                   np.array(radiomics_features_values_rectangle_2) /
                                   (np.array(radiomics_features_values) + 1e-6)), 0)

        index_f = 0
        for kflod in range(5):
            scaler = pickle.load(open('./predict_model/' + str(kflod) + '_scaler.pkl', 'rb'))
            index = np.load('./predict_model/' + str(kflod) + '_lasso_index.npy')
            LR = joblib.load('./predict_model/' + 'fold_LR_' + str(kflod) + '.pkl')
            features_all_p = []
            features_all_p.append(features)
            features_scaler = scaler.transform(features_all_p)
            features_slected = updata_features(index, features_scaler)
            prob = LR.predict_proba(features_slected)
            prob = sofmax(prob)
            if index_f==0:
                prob_all = prob
            else:
                prob_all = prob_all + prob
            index_f = index_f + 1
        prob_all = prob_all/5
        print(prob_all)

        self.prediction_bar.textEdit_result_gd.append('该病人为BIRADS-2级的概率为:' + str(round(prob_all[0][0], 4)))
        self.prediction_bar.textEdit_result_gd.append('该病人为BIRADS-3级的概率为:' + str(round(prob_all[0][1], 4)))
        self.prediction_bar.textEdit_result_gd.append('该病人大于等于BIRADS-4级的概率为:' + str(round(prob_all[0][2], 4)))
        self.prediction_bar.msg_box.hide()

        msg_box = QMessageBox(QMessageBox.Warning, '提示', '预测完毕!')
        msg_box.resize(300, 300)
        msg_box.exec()

        qmut_predict.unlock()
        self._signal_p.emit()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 650)
        MainWindow.setFixedSize(MainWindow.width(), MainWindow.height())

        font = QtGui.QFont()
        # font.setFamily("Times New Roman")
        font.setPointSize(12)

        font_2 = QtGui.QFont()
        font_2.setPointSize(14)

        self.img = ''
        self.seg_img = ''
        self.seg_rectangle_img = ''
        self.infomation = ''
        self.value = ''
        self.img_path = ''
        self.show_seg = False
        self.watch_flag = 999
        self.thread_SEG = ''
        self.thread_predict = ''
        self.IMIAPD = ''

################################################# self.centralwidget_1 ########################################################################
        self.centralwidget_1 = QtWidgets.QWidget(MainWindow)
        self.centralwidget_1.setObjectName("centralwidget")

        self.graphicsView_x = GraphicsView(self.centralwidget_1)
        self.graphicsView_x.setFixedSize(380, 256)
        self.graphicsView_x.setGeometry(30, 40, 380, 256)
        self.graphicsView_x.setObjectName("graphicsView_x")
        self.graphicsView_x.setStyleSheet("#graphicsView_x{background-color: rgb(0, 0, 0)}")

        self.graphicsView_y = GraphicsView(self.centralwidget_1)
        self.graphicsView_y.setFixedSize(380, 256)
        self.graphicsView_y.setGeometry(460, 40, 380, 256)
        self.graphicsView_y.setObjectName("graphicsView_y")
        self.graphicsView_y.setStyleSheet("#graphicsView_y{background-color: rgb(0, 0, 0)}")

        self.graphicsView_z = GraphicsView(self.centralwidget_1)
        self.graphicsView_z.setFixedSize(380, 256)
        self.graphicsView_z.setGeometry(30, 330, 380, 256)
        self.graphicsView_z.setObjectName("graphicsView_z")
        self.graphicsView_z.setStyleSheet("#graphicsView_z{background-color: rgb(0, 0, 0)}")

        self.graphicsView_3d = GraphicsView(self.centralwidget_1)
        self.graphicsView_3d.setFixedSize(380, 256)
        self.graphicsView_3d.setGeometry(460, 330, 380, 256)
        self.graphicsView_3d.setObjectName("graphicsView_3d")
        self.graphicsView_3d.setStyleSheet("#graphicsView_3d{background-color: rgb(0, 0, 0)}")

        self.vtkWidget = QVtkWidget(self.centralwidget_1)
        self.vtkWidget.setFixedSize(380, 256)
        self.vtkWidget.setGeometry(460, 330, 380, 256)
        self.vtkWidget.setObjectName("openGLWidget")

        self.graphicsView_3d.raise_()

        self.verticalScrollBar_x = QtWidgets.QScrollBar(self.centralwidget_1)
        self.verticalScrollBar_x.setGeometry(QtCore.QRect(420, 70, 16, 200))
        self.verticalScrollBar_x.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar_x.setObjectName("verticalScrollBar_x")
        # self.verticalScrollBar_x.valueChanged.connect(self.sliderval_x)

        self.verticalScrollBar_y = QtWidgets.QScrollBar(self.centralwidget_1)
        self.verticalScrollBar_y.setGeometry(QtCore.QRect(850, 80, 16, 200))
        self.verticalScrollBar_y.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar_y.setObjectName("verticalScrollBar_y")
        # self.verticalScrollBar_y.valueChanged.connect(self.sliderval_y)

        self.verticalScrollBar_z = QtWidgets.QScrollBar(self.centralwidget_1)
        self.verticalScrollBar_z.setGeometry(QtCore.QRect(420, 350, 16, 200))
        self.verticalScrollBar_z.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar_z.setObjectName("verticalScrollBar_z")
        # self.verticalScrollBar_z.valueChanged.connect(self.sliderval_z)

        self.groupBox = QtWidgets.QGroupBox(self.centralwidget_1)
        self.groupBox.setGeometry(QtCore.QRect(900, 30, 151, 200))
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")

        ###################################################################################################
        self.label_imgshape = QtWidgets.QLabel(self.groupBox)
        self.label_imgshape.setGeometry(QtCore.QRect(10, 25, 100, 20))
        self.label_imgshape.setFont(font)
        self.label_imgshape.setObjectName("label_imgshape")

        self.lineEdit_imgshape = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_imgshape.setGeometry(QtCore.QRect(20, 45, 110, 30))
        self.lineEdit_imgshape.setObjectName("lineEdit_imgshape")
        self.lineEdit_imgshape.setFocusPolicy(QtCore.Qt.NoFocus)
        self.lineEdit_imgshape.setReadOnly(True)
        # self.lineEdit_imgshape.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lineEdit_imgshape.setAlignment(QtCore.Qt.AlignLeft)
        ###################################################################################################
        self.label_max = QtWidgets.QLabel(self.groupBox)
        self.label_max.setGeometry(QtCore.QRect(10, 75, 100, 20))
        self.label_max.setFont(font)
        self.label_max.setObjectName("label_max")

        self.lineEdit_max = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_max.setGeometry(QtCore.QRect(20, 95, 110, 30))
        self.lineEdit_max.setObjectName("lineEdit_max")
        self.lineEdit_max.setFocusPolicy(QtCore.Qt.NoFocus)
        self.lineEdit_max.setReadOnly(True)
        self.lineEdit_max.setAlignment(QtCore.Qt.AlignLeft)

        self.label_min = QtWidgets.QLabel(self.groupBox)
        self.label_min.setGeometry(QtCore.QRect(10, 125, 100, 20))
        self.label_min.setFont(font)
        self.label_min.setObjectName("label_min")

        self.lineEdit_min = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_min.setGeometry(QtCore.QRect(20, 145, 110, 30))
        self.lineEdit_min.setObjectName("lineEdit_min")
        self.lineEdit_min.setFocusPolicy(QtCore.Qt.NoFocus)
        self.lineEdit_min.setReadOnly(True)
        self.lineEdit_min.setAlignment(QtCore.Qt.AlignLeft)

################################################# self.centralwidget_1 ########################################################################
################################################## self.胶质瘤预测 #############################################################################
        self.glioma_nii_img = ''
        self.glioma_nii_mask = ''
        self.glioma_nii_img_path = ''
        self.glioma_nii_mask_path = ''
################################################## self.胶质瘤预测 #############################################################################

        self.groupBox_position_gd = QtWidgets.QGroupBox(self.centralwidget_1)
        self.groupBox_position_gd.setGeometry(QtCore.QRect(900, 250, 150, 70))
        self.groupBox_position_gd.setFont(font)
        self.groupBox_position_gd.setObjectName("groupBox_position")

        self.pushButton_load_mask = QtWidgets.QPushButton(self.groupBox_position_gd)
        self.pushButton_load_mask.setGeometry(QtCore.QRect(20, 30, 110, 30))
        self.pushButton_load_mask.setFont(font)
        self.pushButton_load_mask.setObjectName("pushButton_radiomics")
        self.pushButton_load_mask.clicked.connect(self.pushButton_seg_mask)

        self.groupBox_position_gd_2 = QtWidgets.QGroupBox(self.centralwidget_1)
        self.groupBox_position_gd_2.setGeometry(QtCore.QRect(900, 340, 150, 240))
        self.groupBox_position_gd_2.setFont(font)
        self.groupBox_position_gd_2.setObjectName("groupBox_position_gd_2")

        self.pushButton_start_prediction = QtWidgets.QPushButton(self.groupBox_position_gd_2)
        self.pushButton_start_prediction.setGeometry(QtCore.QRect(20, 30, 110, 30))
        self.pushButton_start_prediction.setFont(font)
        self.pushButton_start_prediction.setObjectName("pushButton_start_prediction")
        self.pushButton_start_prediction.clicked.connect(self.prediction_br_tumer_type)

        self.textEdit_result_gd = QtWidgets.QTextEdit(self.groupBox_position_gd_2)
        self.textEdit_result_gd.setGeometry(QtCore.QRect(10, 70, 130, 160))
        self.textEdit_result_gd.setObjectName("textEdit_result_gd")
        self.textEdit_result_gd.setFocusPolicy(QtCore.Qt.NoFocus)

################################################# self.胶质瘤预测 #############################################################################
        MainWindow.setCentralWidget(self.centralwidget_1)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1094, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        ##################################################创建菜单栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1100, 100))
        self.menubar.setFont(font_2)
        self.menubar.setDefaultUp(False)
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        ##################################################创建工具栏
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        ##################################################创建第一列
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setFont(font_2)
        self.menu.setObjectName("menubar")

        self.fileOpenAction = QtWidgets.QAction(MainWindow)
        self.fileOpenAction.setObjectName("fileOpenAction")

        # self.saveimgbutton = QtWidgets.QAction(MainWindow)
        # self.saveimgbutton.setObjectName("saveimgbutton")

        self.fileCloseAction = QtWidgets.QAction(MainWindow)
        self.fileCloseAction.setObjectName("fileCloseAction")

        self.menu.addAction(self.fileOpenAction)
        # self.menu.addAction(self.saveimgbutton)
        self.menu.addAction(self.fileCloseAction)

        self.menubar.addAction(self.menu.menuAction())
        self.menu_help = QtWidgets.QMenu(self.menubar)
        self.menu_help.setFont(font_2)
        self.menu_help.setObjectName("menu_help")
        self.menubar.addAction(self.menu_help.menuAction())

        self.about = QtWidgets.QAction(MainWindow)
        self.about.setObjectName("about")
        self.menu_help.addAction(self.about)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

################################################# self.centralwidget_1 ########################################################################
    def show_glioma_prediction_view(self):
        self.groupBox.show()
        self.groupBox_position_gd.show()
        self.groupBox_position_gd_2.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        ################################################# self.centralwidget_1 ########################################################################
        self.groupBox.setTitle(_translate("MainWindow", "图像信息"))
        self.label_imgshape.setText(_translate("MainWindow", "数据大小:"))
        self.label_max.setText(_translate("MainWindow", "最大像素值:"))
        self.label_min.setText(_translate("MainWindow", "最小像素值:"))
        ################################################# self.centralwidget_1 ########################################################################
        ################################################# self.胶质瘤预测 ########################################################################
        self.groupBox_position_gd.setTitle(_translate("MainWindow", "操作"))
        self.pushButton_load_mask.setText(_translate("MainWindow", "分割肿瘤"))
        self.groupBox_position_gd_2.setTitle(_translate("MainWindow", "乳腺诊断信息"))
        self.pushButton_start_prediction.setText(_translate("MainWindow", "开始预测"))
        ################################################# self.胶质瘤分割展示 ########################################################################
        #################################################### 标签栏 ########################################################################
        self.menu.setTitle(_translate("MainWindow", "文件(&F)"))
        self.fileOpenAction.setText(_translate("MainWindow", "打开2D文件"))
        # self.saveimgbutton.setText(_translate("MainWindow", "保存图像(NifTi)"))
        self.fileCloseAction.setText(_translate("MainWindow", "关闭"))
        #################################################### 标签栏 ########################################################################
        self.menu_help.setTitle(_translate("MainWindow", "帮助(&H)"))
        self.about.setText(_translate("MainWindow", "关于"))


    def pushButton_seg_mask(self):
        if self.img == '':
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '请输入分割数据')
            msg_box.resize(300, 300)
            msg_box.exec()
        else:
            self.IMIAPD.segment_ui_timer.start(1000, self)
            self.watch_flag = 99
            self.msg_box = QMessageBox(QMessageBox.Warning, '提示', '正在分割中请稍后，如需中止分割请按取消键！',
                                  QMessageBox.Cancel)
            self.msg_box.button(QMessageBox.Cancel).setText('取消')
            self.msg_box.resize(300, 300)
            self.thread_SEG = Thread_segment(self, self.img, self.img_path)
            self.thread_SEG.start()
            reply = self.msg_box.exec()
            if reply == QMessageBox.Cancel:
                if self.thread_SEG.isRunning():
                    self.thread_SEG.terminate()
                    self.thread_SEG.quit()
                    self.watch_flag = 99
                    self.seg_rectangle_img =''
                    self.seg_img = ''
                    self.value = ''
                    self.thread_SEG = ''
                    self.show_seg = False
                    self.IMIAPD.segment_ui_timer.stop()
            else:
                print('分割完毕')


    def prediction_br_tumer_type(self):
        if self.img == '':
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '请输入分割数据')
            msg_box.resize(300, 300)
            msg_box.exec()
        elif self.seg_img=='':
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '请先分割数据')
            msg_box.resize(300, 300)
            msg_box.exec()
        else:
            # self.IMIAPD.segment_ui_timer.start(1000, self)
            self.textEdit_result_gd.setText('')
            self.msg_box = QMessageBox(QMessageBox.Warning, '提示', '正在预测中请稍后，如需中止预测请按取消键！',
                                  QMessageBox.Cancel)
            self.msg_box.button(QMessageBox.Cancel).setText('取消')
            self.msg_box.resize(300, 300)
            self.thread_predict = Thread_predict(self, self.img, self.seg_img)
            self.thread_predict.start()
            reply = self.msg_box.exec()
            if reply == QMessageBox.Cancel:
                if self.thread_predict.isRunning():
                    self.thread_predict.terminate()
                    self.thread_predict.quit()
            else:
                print('分割完毕')

#
# def rgb2gray( rgb):
#     A  = rgb.shape
#     if A[2]==3:
#         r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
#         gray = (r + g + b) / 3.0
#     else:
#         gray =rgb[:, :, 0]
#     return gray
# itk_img = sitk.ReadImage('C:/Users/Chichien/Desktop/h_04a_-dicom-00132.dcm')
# img_arrray = sitk.GetArrayFromImage(itk_img)[0]
# img_arrray = rgb2gray(img_arrray)
#
# # import numpy as np
# # itk_img_mask = Image.open('C:/Users/Chichien/Desktop/h_04a_-dicom-00132.dcm_seg.gif')
# # mask = np.array(itk_img_mask)
#
# import numpy as np
# itk_img_mask = sitk.ReadImage('C:/Users/Chichien/Desktop/h_04a_-dicom-00132.nii.gz')
# mask = sitk.GetArrayFromImage(itk_img_mask)[0]
#
# mask_final, _, _, max_len_x, max_len_y = get_rectangle(mask)
# mask_rectangle = cut_img_by_rectangle(mask_final, mask_final, max_len_x, max_len_y)
# mask = cut_img_by_rectangle(mask_final, mask, max_len_x, max_len_y)
# cut_img_rectangle = cut_img_by_rectangle(mask_final, img_arrray, max_len_x, max_len_y)
# image = ((cut_img_rectangle - np.mean(cut_img_rectangle)))/ (np.std(cut_img_rectangle))
# mask_2 = mask_rectangle - mask
# # radiomics_features_values_rectangle_2 = extra_Radiomics_features(image, mask_2)
# radiomics_features_values = extra_Radiomics_features(image, mask)
#
# features = np.concatenate((radiomics_features_values,
#                            radiomics_features_values_rectangle_2,
#                            np.array(radiomics_features_values_rectangle_2) /
#                            (np.array(radiomics_features_values) + 1e-6)), 1)

# def sofmax(logits):
# 	e_x = np.exp(logits)
# 	probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
# 	return probs
#
# A = [[0.1, 2,6]]
# A_s = sofmax(A)
# A_s = sofmax(A)


# def updatafeatures(features, deal_dic, num):
#     i = 0
#     for (key, val) in six.iteritems(deal_dic):
#         i = i + 1
#         featuresup = {str(num) + '_' + str(key): val}
#         features.update(featuresup)
#     return features
#
# def get_feature_one_by_on(image, mask, num):
#     image = sitk.GetImageFromArray(image, False)
#     mask = sitk.GetImageFromArray(mask, False)
#
#     features = {}
#
#     firstorder_ex = firstorder.RadiomicsFirstOrder(image, inputMask=mask)  # 一阶
#     firstorder_ex.enableAllFeatures()
#     firstorder_result = firstorder_ex.execute()
#     updatafeatures(features, firstorder_result, 'firstorder_' + num)
#
#     shape_ex = shape.RadiomicsShape(image, mask)  # 形状
#     shape_ex.enableAllFeatures()
#     shape_result = shape_ex.execute()
#     updatafeatures(features, shape_result, 'shape_' + num)
#
#     glcm_ex = glcm.RadiomicsGLCM(image, mask)  # 灰度共生矩阵
#     glcm_ex.enableAllFeatures()
#     glcm_ex.enableFeatureByName('SumAverage', False)
#     glcm_result = glcm_ex.execute()
#     updatafeatures(features, glcm_result, 'glcm_' + num)
#
#     glszm_ex = glszm.RadiomicsGLSZM(image, mask)  # 灰度大小区域矩阵
#     glszm_ex.enableAllFeatures()
#     glszm_result = glszm_ex.execute()
#     updatafeatures(features, glszm_result, 'glszm_' + num)
#
#     glrlm_ex = glrlm.RadiomicsGLRLM(image, mask)
#     glrlm_ex.enableAllFeatures()
#     glrlm_result = glrlm_ex.execute()
#     updatafeatures(features, glrlm_result, 'glrlm_' + num)
#
#     ngtdm_ex = ngtdm.RadiomicsNGTDM(image, mask)  # 相邻灰度色调差异矩阵
#     ngtdm_ex.enableAllFeatures()
#     ngtdm_result = ngtdm_ex.execute()
#     updatafeatures(features, ngtdm_result, 'ngtdm_' + num)
#
#     gldm_ex = gldm.RadiomicsGLDM(image, mask)  # 灰度依赖矩阵
#     gldm_ex.enableAllFeatures()
#     gldm_result = gldm_ex.execute()
#     updatafeatures(features, gldm_result, 'gldm_' + num)
#     return features
#
# def get_features(img, mask):
#     features_all = {}
#     img = sitk.GetImageFromArray(img, False)
#     mask = sitk.GetImageFromArray(mask, False)
#
#     Orignalimg = imageoperations.getOriginalImage(img, mask)
#     Gradientimg = imageoperations.getGradientImage(img, mask)
#     LBP2Dimg = imageoperations.getLBP2DImage(img, mask)
#     Logimg = imageoperations.getLoGImage(img, mask)
#     Waveletimg = imageoperations.getWaveletImage(img, mask)
#     Exponentialtimg = imageoperations.getExponentialImage(img, mask)
#     SquareIimg = imageoperations.getSquareImage(img, mask)
#     SquareRootimg = imageoperations.getSquareRootImage(img, mask)
#     Logarithmimg = imageoperations.getLogarithmImage(img, mask)
#
#
#     features_Orignalimg = get_feature_one_by_on(Orignalimg, mask, 0)
#     features_Gradientimg = get_feature_one_by_on(Gradientimg, mask, 1)
#     features_LBP2Dimg = get_feature_one_by_on(LBP2Dimg, mask, 2)
#     features_Logimg = get_feature_one_by_on(Logimg, mask, 3)
#     features_Waveletimg = get_feature_one_by_on(Waveletimg, mask, 4)
#     features_Exponentialtimg = get_feature_one_by_on(Exponentialtimg, mask, 5)
#     features_SquareIimg = get_feature_one_by_on(SquareIimg, mask, 6)
#     features_SquareRootimg = get_feature_one_by_on(SquareRootimg, mask, 7)
#     features_Logarithmimg = get_feature_one_by_on(Logarithmimg, mask, 8)
#
#     features_all.update(features_Orignalimg)
#     features_all.update(features_Gradientimg)
#     features_all.update(features_LBP2Dimg)
#     features_all.update(features_Logimg)
#     features_all.update(features_Waveletimg)
#     features_all.update(features_Exponentialtimg)
#     features_all.update(features_SquareIimg)
#     features_all.update(features_SquareRootimg)
#     features_all.update(features_Logarithmimg)
#
#     # for d in Waveletimg:
#     #     features_wavelet = get_feature_one_by_on(d[0], mask, 4)
#     #     features_all.update(features_wavelet)
#     radiomics_features_values = []
#     for (key, val) in six.iteritems(features_all):
#         radiomics_features_values.append(val)
#     return radiomics_features_values


# def rgb2gray( rgb):
#     A  = rgb.shape
#     if A[2]==3:
#         r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
#         gray = (r + g + b) / 3.0
#     else:
#         gray =rgb[:, :, 0]
#     return gray
# itk_img = sitk.ReadImage('C:/Users/Chichien/Desktop/h_04a_-dicom-00132.dcm')
# img_arrray = sitk.GetArrayFromImage(itk_img)[0]
# img = rgb2gray(img_arrray)
# A = np.max(img)
# B = np.min(img)
# # img_mask = img > 0
# # img = 255 - (img-B*1.0/(A-B)) * 255
# # img[img_mask == 0] = 0
# # img = np.array(img, dtype=np.uint8)
# # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # bgr -> rgb
# img = (img * 1.0 / A) * 255
# img = img.astype(np.float16)
# img = np.array(img, dtype=np.uint8)
# img = cv2.equalizeHist(img)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
