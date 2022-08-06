import os
import PyQt5.QtGui
from PyQt5 import  QtWidgets
from PyQt5.QtGui import  QPixmap
import cv2
from numpy import imag
import main2
from ImageMatching import *
import sys
from SIFT import *
from harris import *


class MainWindow(QtWidgets.QMainWindow, main2.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # self.ui_elements = Ui(self)
        self.images = {0:None,1:None,2:None,3:None}
        self.originals = [self.original_1,self.original_3, self.original_5, self.original_6]
        # self.originals=[self.original_1,self.original_2,self.original_3,self.original_4, self.original_5, self.original_6, self.original_7]
        self.show()
        #connection to open original images 
        self.btn_1.clicked.connect(lambda: self.open_img(0))
        self.btn_3.clicked.connect(lambda: self.open_img(1))
        self.btn_5.clicked.connect(lambda: self.open_img(2))
        self.btn_6.clicked.connect(lambda: self.open_img(3))
        #outputs
        self.btn_2.clicked.connect(lambda: self.harris_output())
        self.btn_4.clicked.connect(lambda: self.sift_output())
        self.btn_7.clicked.connect(lambda: self.ImageMatching())
        
        
    def open_img(self,index):
        img_path = PyQt5.QtWidgets.QFileDialog.getOpenFileName(None, 'open image', None, "JPG *.jpg;;PNG *.png")[0]
        if img_path:
            self.originals[index].setPixmap(QPixmap(img_path))
            self.originals[index].setScaledContents(True)
            print(img_path)
            self.images[index]=cv2.imread(img_path)
            self.images[index] = cv2.cvtColor(self.images[index], cv2.COLOR_BGR2RGB)
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: please select an image')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_()    
    def harris_output(self):
        Threshold = self.textEdit_1.toPlainText()
        Sensetivity = self.textEdit_2.toPlainText()
        if (Threshold!='')and(Sensetivity !=''):
            Threshold=float(Threshold)
            Sensetivity=float(Sensetivity)
            print(Threshold,Sensetivity)
            _, harris_time= harris(self.images[0],self.images[0].shape[0],self.images[0].shape[1],5, Sensetivity, Threshold)
            if os.path.isfile('Harries.png'):
                output_pixmap = QPixmap('Harries.png')
                self.original_2.setPixmap(output_pixmap)
                self.original_2.setScaledContents(True)
                
            self.text_1.setText(str(harris_time)+"Sec")
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: Please enter Threshold value and Sensetivity value')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_()

    def sift_output(self):
        t1 = time.time()
        kp, dc = computeKeypointsAndDescriptors(self.images[1], sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5)
        t2 = time.time()
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(self.images[1], 'gray')
        for pnt in kp:
            ax.scatter(pnt.pt[0], pnt.pt[1], s=(pnt.size)*10, c="red")
        plt.savefig("siftpic.png")

        if os.path.isfile('siftpic.png'):
            output_pixmap = QPixmap('siftpic.png')
            self.original_4.setPixmap(output_pixmap)
            self.original_4.setScaledContents(True)
        self.text_2.setText(str(t2-t1)+"Sec")  
    def ImageMatching(self):
        _,_,computation_time= GetMatchedImage(self.images[2],self.images[3],self.comboBox.currentText())
        if os.path.isfile('MatchedImage.png'):
            output_pixmap = QPixmap('MatchedImage.png')
            self.original_7.setPixmap(output_pixmap)
            self.original_7.setScaledContents(True)
        self.text_3.setText(str(computation_time)+"Sec")       
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = MainWindow()
    application.show()
    app.exec_()
        

if __name__ == "__main__":
    main()