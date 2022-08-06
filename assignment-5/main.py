import os
import PyQt5.QtGui
from PyQt5 import  QtWidgets
from PyQt5.QtGui import  QPixmap
import cv2
from PyQt5.QtWidgets import QFileDialog
from FaceDetection import FaceDetection
from Performance import accuracy, roc,draw_CM,draw_ROC,read_img,test_imgs, prediction, prob_vector, test_labels, thresh, y

import mainwindow
import sys
import time
from pca import pca

class MainWindow(QtWidgets.QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.images = {0:None,1:None}
        self.inputImages =[self.image_1,self.image_3]
        self.outputImages =[self.image_2,self.image_4,self.image_5,self.image_6]
        self.browse_0.clicked.connect(lambda: self.open(0))       
        self.browse_1.clicked.connect(lambda: self.open(1))
        self.button_1.clicked.connect(lambda: self.Detect(0))
        self.button_2.clicked.connect(lambda: self.Recognize(1))
        self.button_4.clicked.connect(lambda: self.ROC_ACC(2,3))
        self.button_3.clicked.connect(lambda:self.draw_faces(1))
        self.show()
    def open(self,index):
        img_path = QFileDialog.getOpenFileName(None, 'open image', None, "JPG *.jpg;;PNG *.png;;JPEG *.jpeg")[0]
        if img_path:
            self.images[index] = cv2.imread(img_path)
            # print(self.images[index])
            self.inputImages[index].setPixmap(QPixmap(img_path))
            self.outputImages[index].clear()
            self.inputImages[index].setScaledContents(True)
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: please select an image')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_() 
    # def open2(self,index):
    #     img_path = QFileDialog.getOpenFileName(None, 'open image', None, "JPG *.jpg;;PNG *.png;;JPEG *.jpeg")[0]
    #     if img_path:
    #         self.images[index] = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    #         # print(self.images[index])
    #         self.inputImages[index].setPixmap(QPixmap(img_path))
    #         self.outputImages[index].clear()
    #         self.inputImages[index].setScaledContents(True)
    #     else:
    #         msg = PyQt5.QtWidgets.QMessageBox()
    #         msg.setWindowTitle('ERROR')
    #         msg.setText('Error: please select an image')
    #         msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
    #         msg.exec_() 
    def Detect(self,index):    
        t1 = time.time()
        self.faces=FaceDetection(self.images[index]) 
        t2 = time.time()
        if len(self.faces) > 0:
            if os.path.isfile('Images/FacesDetected.png'):
                output_pixmap = QPixmap('Images/FacesDetected.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)   
            self.label_1.setText("Computation Time: "+str(round((t2-t1),3))+"Sec")        
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: No Faces are detected')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_()       
    def Recognize(self,index): 
        print('Reco')   
        # self.faces=FaceDetection(self.images[index])
        # if len(self.faces) > 0:
        t1 = time.time()
        pca(self.images[index])
        # Call your function here
        t2 = time.time()
        if os.path.isfile('Images/FaceRecognized.png'):
            output_pixmap = QPixmap('Images/FaceRecognized.png')
            self.outputImages[index].setPixmap(output_pixmap)
            self.outputImages[index].setScaledContents(True)   
        self.label_2.setText("Computation Time: "+str(round((t2-t1),3))+"Sec")    
        # else:
        #     msg = PyQt5.QtWidgets.QMessageBox()
        #     msg.setWindowTitle('ERROR')
        #     msg.setText('Error: There is no face to recognize')
        #     msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
        #     msg.exec_()  
    def draw_faces(self,index):
        testt=read_img(self.images[index])
        t1=time.time()
        test_imgs(testt)
        t2=time.time()
        if os.path.isfile('E:/cv_tasks/task5/github/CV_Final_Project/Images/Recognized_Multiple.png'):
                output_pixmap = QPixmap('E:/cv_tasks/task5/github/CV_Final_Project/Images/Recognized_Multiple.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)
        self.label_2.setText("Computation Time: "+str(round((t2-t1),3))+"Sec") 


    def ROC_ACC(self,index1,index2):  
        print('Roc')  
        # Call your function here
        class_numm = self.textEdit_1.toPlainText()
        if (class_numm!=''):
            classnum=int(class_numm)
            t1 = time.time()
            R,CM= roc(prob_vector[:,classnum], y[:,classnum], thresh[5])
            draw_CM(CM)
            draw_ROC(R)
            acc = accuracy(prediction, test_labels)
            t2 = time.time()

            if os.path.isfile('E:/cv_tasks/task5/github/CV_Final_Project/Images/CM.png'):
                output_pixmap = QPixmap('E:/cv_tasks/task5/github/CV_Final_Project/Images/CM.png')
                self.outputImages[index2].setPixmap(output_pixmap)
                self.outputImages[index2].setScaledContents(True)

            if os.path.isfile('E:/cv_tasks/task5/github/CV_Final_Project/Images/ROC.png'):
                output_pixmap = QPixmap('E:/cv_tasks/task5/github/CV_Final_Project/Images/ROC.png')
                self.outputImages[index1].setPixmap(output_pixmap)
                self.outputImages[index1].setScaledContents(True)   
            self.label_3.setText("Acuuracy: "+str(round(acc*100))+"%" ) 
            self.label_5.setText("Computation Time: "+str(round((t2-t1),3))+"Sec") 
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: Please enter class number')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_() 

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = MainWindow()
    application.show()
    app.exec_()
        

if __name__ == "__main__":
    main()                