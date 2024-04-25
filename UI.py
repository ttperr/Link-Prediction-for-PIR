# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QGridLayout, QLayout,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QSpacerItem, QVBoxLayout, QWidget, QLabel, QHBoxLayout,QButtonGroup)

class Ui_Widget(object):
    def setupUi(self, Widget,PIR,n_docs):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(960, 540)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Widget.sizePolicy().hasHeightForWidth())
        Widget.setSizePolicy(sizePolicy)
        self.gridLayout_2 = QGridLayout(Widget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setSizeConstraint(QLayout.SetNoConstraint)
        self.gridLayout.setVerticalSpacing(20)
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 0, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 2, 0, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 1, 0, 1, 1)

        self.lineEdit = QLineEdit(Widget)
        self.lineEdit.setObjectName(u"lineEdit")
        font = QFont()
        font.setPointSize(16)
        self.lineEdit.setFont(font)
        self.lineEdit.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.lineEdit, 1, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 1, 3, 1, 1)

        self.lineEdit_2 = QLineEdit(Widget)
        self.lineEdit_2.setObjectName(u"lineEdit_2")
        font1 = QFont()
        font1.setPointSize(15)
        self.lineEdit_2.setFont(font1)
        self.lineEdit_2.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.lineEdit_2, 1, 2, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_3, 3, 0, 1, 1)

        self.scrollArea = QScrollArea(Widget)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setAlignment(Qt.AlignHCenter|Qt.AlignTop)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 490, 366))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName(u"verticalLayout")
        '''
        self.pushButton = QPushButton(self.scrollAreaWidgetContents)
        self.pushButton.setObjectName(u"pushButton")

        self.verticalLayout.addWidget(self.pushButton)

        self.pushButton_2 = QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.verticalLayout.addWidget(self.pushButton_2)
        '''
        font2 = QFont()
        font2.setPointSize(14)
        font2.setKerning(True)
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        for i in range(n_docs):
            button=RichTextPushButton(self.scrollAreaWidgetContents)
            button.setObjectName(u"pushButton_"+str(i))
            button.setFlat(True)
            button.setFont(font2)
            sizePolicy1.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
            button.setSizePolicy(sizePolicy1)
            button.clicked.connect((lambda a: lambda: self.register_click(a))(i)) #terrible solution but no other way to make i binding
            self.verticalLayout.addWidget(button)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.gridLayout.addWidget(self.scrollArea, 2, 1, 1, 2)

        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 2)
        self.gridLayout.setRowStretch(2, 16)
        self.gridLayout.setRowStretch(3, 1)
        self.gridLayout.setColumnStretch(0, 4)
        self.gridLayout.setColumnStretch(1, 7)
        self.gridLayout.setColumnStretch(2, 2)
        self.gridLayout.setColumnStretch(3, 4)

        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)


        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi
    
##FROM HERE ON CUSTOM CODE
        #still inside setupUi
        self.PIR=PIR
        self.lineEdit.returnPressed.connect(self.run_query)
        self.lineEdit_2.returnPressed.connect(self.update_user)
        self.lineEdit_2.textEdited.connect(self.update_user)
        self.last_query_results=[]
        #self.textEdit.selectionChanged.connect(self.selected_doc)

    def run_query(self):
        #print(self.lineEdit.text())
        #print(self.lineEdit_2.text())
        if(self.lineEdit_2.text()==""):
            self.lineEdit_2.setStyleSheet("color:red;")
            print("No user selected. Run again after selecting a user.")
            return
        if(self.PIR.is_new_user(self.lineEdit_2.text())):
            print("New user!")
        self.last_query_results=self.PIR.query(self.lineEdit.text(),self.lineEdit_2.text())
        self.display_query_results()

    def display_query_results(self):
        for i in range(len(self.last_query_results)):
            text=""
            if(self.last_query_results[i][0]>0):
                text+='<span style="color:green;font-size:16px;">▲'+str(self.last_query_results[i][0])+'</span>  '
            elif(self.last_query_results[i][0]<0):
                text+='<span style="color:red;font-size:16px;">▼'+str(abs(self.last_query_results[i][0]))+'</span>  '
            else:
                text+='<span style="color:grey;font-size:16px;">--</span>  '
            text+="<strong>"+self.last_query_results[i][2]+"</strong><br>"
            text+=self.last_query_results[i][3]
            self.verticalLayout.itemAt(i).widget().setText(text)
        for i in range(len(self.last_query_results),self.verticalLayout.count()):
            self.verticalLayout.itemAt(i).widget().setText("")

    def update_user(self):
        if(self.lineEdit_2!=""):
            self.lineEdit_2.setStyleSheet("border:0px;")
        else:
            self.lineEdit_2.setStyleSheet("color:black;")
            return
        #print(self.lineEdit_2.text())
        if(self.PIR.is_new_user(self.lineEdit_2.text())):
            self.lineEdit_2.setStyleSheet("color:green;")
            #print("New user!")
        else:
            self.lineEdit_2.setStyleSheet("color:black;")
    def register_click(self,res_ind):
        if res_ind<len(self.last_query_results):
            self.PIR.register_click([doc[1] for doc in self.last_query_results],self.lineEdit_2.text(),res_ind,self.lineEdit.text())
            #register_click(self,doc_ids,user_id,doc_clicked_index,query_text)

##STOP CUSTOM CODE

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.lineEdit.setPlaceholderText(QCoreApplication.translate("Widget", u"Search", None))
        self.lineEdit_2.setPlaceholderText(QCoreApplication.translate("Widget", u"User", None))
    # retranslateUi


#OTHER CUSTOM CODE
class RichTextPushButton(QPushButton):
    def __init__(self, parent=None, text=None):
        if parent is not None:
            super().__init__(parent)
        else:
            super().__init__()
        self.__lbl = QLabel(self)
        if text is not None:
            self.__lbl.setText(text)
        self.__lyt = QHBoxLayout()
        self.__lyt.setContentsMargins(0, 0, 0, 0)
        self.__lyt.setSpacing(0)
        self.setLayout(self.__lyt)
        self.__lbl.setAttribute(Qt.WA_TranslucentBackground)
        self.__lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.__lbl.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.__lbl.setTextFormat(Qt.RichText)
        self.__lyt.addWidget(self.__lbl)
        return

    def setText(self, text):
        self.__lbl.setText(text)
        self.updateGeometry()
        return

    def sizeHint(self):
        s = QPushButton.sizeHint(self)
        w = self.__lbl.sizeHint()
        s.setWidth(w.width())
        s.setHeight(w.height())
        return s