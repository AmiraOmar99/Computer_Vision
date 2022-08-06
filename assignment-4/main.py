import os
import PyQt5.QtGui
from PyQt5 import  QtWidgets
from PyQt5.QtGui import  QPixmap
import cv2
from PyQt5.QtWidgets import QFileDialog
import mainwindow
from Thresholding import *
from luv import *
from segmentation_rgb import *
from segmentation_luv import *
import sys
import time


class MainWindow(QtWidgets.QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.images = {0:None,1:None,2:None,3:None,4:None}
        self.inputImages =[self.image_1,self.image_11,self.image_3,self.image_5,self.image_7,self.image_9]
        self.outputImages =[self.image_2,self.image_12,self.image_4,self.image_6,self.image_8,self.image_10]
        self.browse_0.clicked.connect(lambda: self.open(0))       
        self.browse_1.clicked.connect(lambda: self.open(1))    
        self.browse_2.clicked.connect(lambda: self.open(2))    
        self.browse_3.clicked.connect(lambda: self.open(3))    
        self.browse_4.clicked.connect(lambda: self.open(4))    
        self.browse_5.clicked.connect(lambda: self.open(5))  
        self.button_1.clicked.connect(lambda: self.Thresholding(self.combobox.currentText(),0))  
        self.button_2.clicked.connect(lambda: self.ConvertToLUV(1))
        self.button_3.clicked.connect(lambda: self.kmeans_rgb(2))
        self.button_4.clicked.connect(lambda: self.kmeans_luv(2))
        self.button_5.clicked.connect(lambda: self.region_growing(3))
        self.button_7.clicked.connect(lambda: self.Meanshift_rgb(4))
        self.button_8.clicked.connect(lambda: self.Meanshift_luv(4))   
        self.button_10.clicked.connect(lambda: self.aggloremative_rgb(5))
        self.button_9.clicked.connect(lambda: self.aggloremative_luv(5))   
        
               
 
        self.show()
    def open(self,index):
        img_path = QFileDialog.getOpenFileName(None, 'open image', None, "JPG *.jpg;;PNG *.png")[0]
        if img_path:
            self.images[index] = cv2.imread(img_path)
            print(self.images[index])
            self.inputImages[index].setPixmap(QPixmap(img_path))
            self.inputImages[index].setScaledContents(True)
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: please select an image')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_() 
    def Thresholding(self,Text,index):    
        print(Text) 
        if (Text == "Global Optimal"):
            t1 = time.time()
            OptimalThresholding(self.images[index]) 
            t2 = time.time()
            if os.path.isfile('OptImg.png'):
                output_pixmap = QPixmap('OptImg.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)  
        elif (Text == "Global Otsu"):
            t1 = time.time()
            OtsuThresholding(self.images[index]) 
            t2 = time.time()
            if os.path.isfile('OtsuImg.png'):
                output_pixmap = QPixmap('OtsuImg.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)  
        elif (Text == "Global Spectral"):
            t1 = time.time()
            SpectralThresholding(self.images[index]) 
            t2 = time.time()
            if os.path.isfile('SpectImg.png'):
                output_pixmap = QPixmap('SpectImg.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)                  
        elif (Text == "Local Optimal"):
            regionx=self.textEdit_4.toPlainText() 
            regiony=self.textEdit_5.toPlainText() 
            if ( regionx != "" and regiony !=""):
                t1 = time.time()
                LocalThresholding(self.images[index],int(regionx),int(regiony),OptimalThresholding,'LocOpt.png')  
                t2 = time.time()      
                if os.path.isfile('LocOpt.png'):
                    output_pixmap = QPixmap('LocOpt.png')
                    self.outputImages[index].setPixmap(output_pixmap)
                    self.outputImages[index].setScaledContents(True)
            else:
                msg = PyQt5.QtWidgets.QMessageBox()
                msg.setWindowTitle('ERROR')
                msg.setText('Error: Please enter Region X and Region Y')
                msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
                msg.exec_()   
        elif (Text == "Local Otsu"):
            regionx=self.textEdit_4.toPlainText() 
            regiony=self.textEdit_5.toPlainText() 
            if ( regionx != "" and regiony !=""):
                t1 = time.time()
                LocalThresholding(self.images[index],int(regionx),int(regiony),OtsuThresholding,'LocOtsu.png')        
                t2 = time.time()
                if os.path.isfile('LocOtsu.png'):
                    output_pixmap = QPixmap('LocOtsu.png')
                    self.outputImages[index].setPixmap(output_pixmap)
                    self.outputImages[index].setScaledContents(True)
            else:
                msg = PyQt5.QtWidgets.QMessageBox()
                msg.setWindowTitle('ERROR')
                msg.setText('Error: Please enter Region X and Region Y')
                msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
                msg.exec_()   
        elif (Text == "Local Spectral"):
            regionx=self.textEdit_4.toPlainText() 
            regiony=self.textEdit_5.toPlainText() 
            if ( regionx != "" and regiony !=""):
                t1 = time.time()
                LocalThresholding(self.images[index],int(regionx),int(regiony),SpectralThresholding,'LocSpect.png')        
                t2 = time.time()
                if os.path.isfile('LocSpect.png'):
                    output_pixmap = QPixmap('LocSpect.png')
                    self.outputImages[index].setPixmap(output_pixmap)
                    self.outputImages[index].setScaledContents(True)
            else:
                msg = PyQt5.QtWidgets.QMessageBox()
                msg.setWindowTitle('ERROR')
                msg.setText('Error: Please enter Region X and Region Y')
                msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
                msg.exec_()   
        else:
            self.outputImages[index].clear()
        self.label_1.setText(str(round((t2-t1),3))+"Sec")     

    def ConvertToLUV(self,index):
        img=np.copy(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        t1 = time.time()
        RGB2LUV(img)
        t2 = time.time()
        if os.path.isfile('luv.png'):
            self.outputImages[index].clear()
            output_pixmap = QPixmap('luv.png')
            self.outputImages[index].setPixmap(output_pixmap)
            self.outputImages[index].setScaledContents(True)
        self.label_10.setText(str(round((t2-t1),3))+"Sec") 

    def kmeans_rgb(self,index):
        Cluster_number = self.textEdit_1.toPlainText()
        if (Cluster_number!=''):
            t1 = time.time()
            apply_k_means_rgb(source=self.images[index],k=int(Cluster_number))
            t2 = time.time()
            if os.path.isfile('kmeans_rgb.png'):
                self.outputImages[index].clear()
                output_pixmap = QPixmap('kmeans_rgb.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)
            self.label_2.setText(str(round((t2-t1),3))+"Sec")
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: Please enter cluster numbers')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_() 
    def kmeans_luv(self,index):
        Cluster_number = self.textEdit_1.toPlainText()
        if (Cluster_number!=''):
            t1 = time.time()
            apply_k_means_luv(source=self.images[index],k=int(Cluster_number))
            t2 = time.time()
            if os.path.isfile('kmeans_luv.png'):
                self.outputImages[index].clear()
                output_pixmap = QPixmap('kmeans_luv.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)
            self.label_2.setText(str(round((t2-t1),3))+"Sec")
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: Please enter cluster numbers')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_()
    def region_growing(self,index):
        Threshold = self.textEdit_2.toPlainText()
        # im=np.copy(self.images[index])
        if (Threshold!=''):
            t1 = time.time()
            apply_region_growing(source=self.images[index],threshold=int(Threshold))
            t2 = time.time()
            if os.path.isfile('region_grow.png'):
                self.outputImages[index].clear()
                output_pixmap = QPixmap('region_grow.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)
            self.label_3.setText(str(round((t2-t1),3))+"Sec")
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: Please enter threshold value')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_()
    def Meanshift_rgb(self,index):
        Threshold = self.textEdit_3.toPlainText()
        if (Threshold!=''):
            t1 = time.time()
            apply_mean_shift_rgb(source=self.images[index],threshold=int(Threshold))
            t2 = time.time()
            if os.path.isfile('meanshift_rgb.png'):
                self.outputImages[index].clear()
                output_pixmap = QPixmap('meanshift_rgb.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)
            self.label_4.setText(str(round((t2-t1),3))+"Sec")
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: Please enter threshold value')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_() 
    def Meanshift_luv(self,index):
        Threshold = self.textEdit_3.toPlainText()
        if (Threshold!=''):
            t1 = time.time()
            apply_mean_shift_luv(source=self.images[index],threshold=int(Threshold))
            t2 = time.time()
            if os.path.isfile('meanshift_luv.png'):
                self.outputImages[index].clear()
                output_pixmap = QPixmap('meanshift_luv.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)
            self.label_4.setText(str(round((t2-t1),3))+"Sec")
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: Please enter threshold value')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_()
    def aggloremative_rgb(self,index):
        Cluster_number = self.textEdit_6.toPlainText()
        if (Cluster_number!=''):
            t1 = time.time()
            apply_agglomerative_rgb(source=self.images[index],clusters_numbers=int(Cluster_number),initial_clusters=25)
            t2 = time.time()
            if os.path.isfile('agg_rgb.png'):
                self.outputImages[index].clear()
                output_pixmap = QPixmap('agg_rgb.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)
            self.label_5.setText(str(round((t2-t1),3))+"Sec")
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: Please enter  Cluster number')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_() 
    def aggloremative_luv(self,index):
        ClusterNumber = self.textEdit_6.toPlainText()
        if (ClusterNumber!=''):
            t1 = time.time()
            apply_agglomerative_luv(source=self.images[index],clusters_numbers=int(ClusterNumber),initial_clusters=25)
            t2 = time.time()
            if os.path.isfile('agg_luv.png'):
                self.outputImages[index].clear()
                output_pixmap = QPixmap('agg_luv.png')
                self.outputImages[index].setPixmap(output_pixmap)
                self.outputImages[index].setScaledContents(True)
            self.label_5.setText(str(round((t2-t1),3))+"Sec")
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: Please enter  Cluster number')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_()



def main():
    app = QtWidgets.QApplication(sys.argv)
    application = MainWindow()
    application.show()
    app.exec_()
        

if __name__ == "__main__":
    main()        