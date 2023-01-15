# -*- coding: utf-8 -*-
import time
import numpy as np
import SimpleITK as sitk
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QMovie
from PyQt5.Qt import QBasicTimer, QIcon, QMessageBox, QSettings
from ui_IMIAPD import Ui_MainWindow as IMIAPD_MainWindow
from PyQt5.QtWidgets import QApplication, QFileDialog, QSplashScreen, QMainWindow
from PyQt5.QtCore import Qt, QObject
from PyQt5.Qt import (QThread, pyqtSignal)

class MySplashScreen(QSplashScreen):
    def __init__(self):
        super(MySplashScreen, self).__init__()
        self.movie = QMovie('./ui/load_img.gif')
        self.movie.frameChanged.connect(lambda: self.setPixmap(self.movie.currentPixmap()))
        self.movie.start()

    def mousePressEvent(self, QMouseEvent):
        pass

class LoadDataWorker(QObject):
    finished = pyqtSignal()
    message_signal = pyqtSignal(str)

    def __init__(self):
        super(LoadDataWorker, self).__init__()
        self.message_signal
    def run(self):
        # self.message_signal.emit("加载中...")
        for i in range(2):
            time.sleep(1)
            self.message_signal.emit("加载中...")
        self.finished.emit()

class MainForm(QMainWindow, IMIAPD_MainWindow):
    def __init__(self, splash):
        super(MainForm, self).__init__()
        self.setupUi(self)

        icon = QIcon('./ui/icon.png')
        self.setWindowIcon(icon)
        self.setWindowTitle('医学图像辅助诊断系统 v1.0')
        font = QtGui.QFont()
        font.setPointSize(14)
        self.setFont(font)
        self.IMIAPD = self

        self.segment_ui_timer = QBasicTimer()
        self.Ui_MainWindow = IMIAPD_MainWindow
        self.fileCloseAction.triggered.connect(self.close)
        self.fileOpenAction.triggered.connect(self.fileOpenAction_fun)
        self.about.triggered.connect(self.about_fun)
        self.show_glioma_prediction_view()

###########################################################################################
        self.splash = splash
        self.load_thread = QThread()
        self.load_worker = LoadDataWorker()
        self.load_worker.moveToThread(self.load_thread)
        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.message_signal.connect(self.set_message)
        self.load_worker.finished.connect(self.load_worker_finished)
        self.load_thread.start()

        while self.load_thread.isRunning():
            QtWidgets.qApp.processEvents()  # 不断刷新，保证动画流畅

        self.load_thread.deleteLater()


    def timerEvent(self, event):
        print('监听')
        if self.watch_flag==1:
            if self.show_seg == True:
                self.graphicsView_y.update_image_mask(self.seg_img)
                self.graphicsView_z.update_image_rectangle(self.seg_rectangle_img)
                self.show_seg = False
                self.watch_flag = 99
                self.segment_ui_timer.stop()

    def load_worker_finished(self):
        self.load_thread.quit()
        self.load_thread.wait()


    def set_message(self, message):
        self.splash.showMessage(message, Qt.AlignLeft | Qt.AlignBottom, Qt.white)


    def about_fun(self):
        self.info_ui.show()


    def get_nii_array(self, itk_img):
            nifti_array = sitk.GetArrayFromImage(itk_img)[0]
            spacing = itk_img.GetSpacing()
            size = itk_img.GetSize()
            return nifti_array, spacing, size


    def fileOpenAction_fun(self):
        qSettings = QSettings()
        lastPath = qSettings.value("LastFilePath")

        self.img_path, ok = QFileDialog.getOpenFileName(
            self, '打开', lastPath, 'dicom Files (*.dcm);; nii Files (*.nii);;nii.gz Files (*.nii.gz)')
        if self.img_path != '':
            labelImage = sitk.ReadImage(self.img_path)
            Nii_infomation = self.get_nii_array(labelImage)
            A = Nii_infomation[2][2]
            B = len(Nii_infomation[2])
            if (A == 3 and B == 3) or (A == 1 and B == 3) or B == 2:
                value = []
                self.graphicsView_x.empty_image()
                self.graphicsView_y.empty_image()
                self.graphicsView_z.empty_image()

                value.append(str(Nii_infomation[2]))
                value.append(str(np.max(Nii_infomation[0])))
                value.append(str(np.min(Nii_infomation[0])))
                self.value = value
                img_show = Nii_infomation[0]
                self.seg_img = ''
                self.seg_rectangle_img = ''
                self.show_seg = False
                self.watch_flag = 999
                self.thread_SEG = ''
                self.thread_predict = ''
                self.textEdit_result_gd.setText('')

                self.update_image(img_show)
                self.img = img_show
            else:
                msg_box = QMessageBox(QMessageBox.Warning, '警告', '请输入2D文件')
                msg_box.resize(300, 300)
                msg_box.exec()

    def update_image(self, img_show):
        self.set_value(self.value)
        self.graphicsView_x.update_image(img_show)


    def set_value(self, value):
        self.need_name = ['size', 'max', 'min']
        self.lineEdit_imgshape.setText(value[0])
        self.lineEdit_max.setText(value[1])
        self.lineEdit_min.setText(value[2])
        self.lineEdit_imgshape.setCursorPosition(0)
        self.lineEdit_min.setCursorPosition(0)
        self.lineEdit_max.setCursorPosition(0)


    def closeEvent(self, event):
        showMessage = QMessageBox.question
        reply = showMessage(self, '警告', "系统将退出，是否确认退出?", QMessageBox.Yes |QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setObjectName("app")
    app.setStyleSheet(open('./ui/light.qss', encoding='utf-8').read())
    splash = MySplashScreen()
    splash.show()
    app.processEvents()
    from ui_info import Ui_MainWindow as IMIAPD_info

    class Main_info(QMainWindow, IMIAPD_info):
        def __init__(self):
            super(Main_info, self).__init__()
            self.setupUi(self)
            self.retranslateUi(self)
            icon = QIcon('./ui/icon.png')
            self.setWindowIcon(icon)
            self.setWindowTitle('关于')
            self.setWindowModality(Qt.ApplicationModal)
            self.setWindowFlags(Qt.WindowStaysOnTopHint)

    info_ui = Main_info()

    main_win = MainForm(splash)
    main_win.info_ui = info_ui
    main_win.show()
#####################################################################################
#####################################################################################
    splash.finish(main_win)  # 主界面加载完成后隐藏
    splash.movie.stop()  # 停止动画
    splash.deleteLater()
    sys.exit(app.exec_())
