# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'IMIAPD_convert.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(730, 500)
        MainWindow.setFixedSize(MainWindow.width(), MainWindow.height())

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.textEdit_info = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_info.setGeometry(QtCore.QRect(20, 20, 700, 480))
        self.textEdit_info.setObjectName("textEdit_info")
        self.textEdit_info.setFocusPolicy(QtCore.Qt.NoFocus)
        self.textEdit_info.setReadOnly(True)

        font = QtGui.QFont()
        font.setPointSize(14)
        self.textEdit_info.setFont(font)
        _translate = QtCore.QCoreApplication.translate
        self.textEdit_info.setPlainText(_translate("MWin",
                                                  "      感谢您使用贵州省智能医学影像分析与精准诊断重点实验产品\n"
                                                  "              欢迎访问：http://imiapd.gzu.edu.cn/\n"
                                                  "              这是一个测试的许可协议安装条款说明：\n"
                                                  "------------------------------------------------------------------\n"
                                                  "                 医学图像辅助诊断系统 v1.0\n"
                                                  "\n"
                                                  "             1.IMIAPD致力于一个专业的医学图像处理软件\n"
                                                  "------------------------------------------------------------------\n"
                                                  "  贵州省智能医学影像分析与精准诊断重点实验室于2017年10月获批。实验室主要依托贵州大学计算机科学与技术学院软件工程一级学科博士点、计算机机科学与技术一级学科硕士点、计算机科学与技术重点学科，并与贵州省人民医院放射科、心内科、肿瘤科以及脑神经内科密切合作，是目前我省唯一一个基于人工智能技术从事医学图像处理与精准诊断研究的重点实验室。\n"
                                                  "  实验室围绕国家十三五“精准医疗”规划，结合国家和我省在“大健康”及相关领域的长中期科技发展战略需求，针对我国发病率、致死率最高的三种疾病：心血管疾病、肿瘤以及人脑神经系统疾病，开展智能医学影像分析与精准诊断领域的相关研究。主要瞄准国内外医学成像研究领域的前沿课题，利用大数据、计算机、数学及信息处理方法研究医学图像重建、处理分析、可视化、医学图像深度学习等问题。实验室力争基于影像大数据、基因大数据以及临床问诊大数据，利用深度学习和人工智能技术实现心血管病、人脑神经系统疾病、常见肿瘤的超早期诊断以及个性化精准治疗，进而提高国民健康水平和降低我国卫生总支出。同时为促进学科交叉融合、科技指导临床、重大疾病的有效防治提供技术支撑体系和创新平台。\n"
                                                  "  目前，实验室拥有一支理、工、医相结合的高水平学术队伍，包括大数据、计算理论、人工智能、图像处理、医学成像、计算机视觉、心内科学、病理学、遗传学、神经内科学、肿瘤学等学科的杰出人才。其中固定研究人员26名，其中在国际及全国性学会任职13人，省市学会任职20人，正高职称9人，副高职称11人，博士生导师8人，硕士生导师20人，其中具有博士学位者21人。科研队伍中有10人是从国外引进的人才或曾出国进修1年以上。\n"
                                                  "  实验室长期承担国家自然科学基金项目及国际合作项目，近5年承担国家级、省部级、国际合作课题项目40余项，经费近2000万元，其中，国家自然科学基金项目18项，国际合作项目5项，省部级项目16项。此外，实验室与国内及国际一些著名的科研机构如美国加州伯克利大学、美国Harvard大学医学院、法国INSA-LYON大学、英国University College London、澳大利亚Sydney大学、加拿大Concordia大学，哈尔滨工业大学，北京交通大学、上海交通大学以及中国科学院自动化研究所等科研机构开展了实质性的合作。并积极参加国内及国际相关领域会议，邀请国内外同行来实验室做学术交流和学术报告，并参与多中心的、大型的科研合作项目。这种开放合作方式，将促进实验室快速发展。\n"
                                                  ""))

        # self.pushButton_cancel = QtWidgets.QPushButton(self.centralwidget)
        # self.pushButton_cancel.setGeometry(QtCore.QRect(350, 220, 50, 20))
        # font = QtGui.QFont()
        # font.setFamily("Adobe Devanagari")
        # font.setPointSize(12)
        # self.pushButton_cancel.setFont(font)
        # self.pushButton_cancel.setObjectName("pushButton_cancel")
        # self.pushButton_cancel.clicked.connect(self.pushButton_cancel_info)
########################################################################################################
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 456, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def pushButton_cancel_info(self):
        self.hide()

    def closeEvent(self, event):
        print('关闭')

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        # self.pushButton_cancel.setText(_translate("MainWindow", "关闭"))


