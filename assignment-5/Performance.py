
# In[4]:

import glob
import math
import matplotlib.image as mpimg

import cv2
import numpy as np
import os
import seaborn as snNew
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from  tensorflow.keras.utils import to_categorical


# In[5]:


training_path="E:/cv_tasks/task5/github/CV_Final_Project/data/training"
test_path="E:/cv_tasks/task5/github/CV_Final_Project/data/test"


# In[6]:


def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(filename)
            file_paths.append(filepath)
    labels = []
    for image_path in file_paths:
        label=(image_path.split(".")[0]).split("_")[1]
        labels.append(int(label))
    labels = np.array(labels)
    class_num=len(np.unique(labels))

    # print('num of classes is:', class_num)
    return np.array(file_paths), labels  # Self-explanatory.


# In[7]:


def data_prep(direc, paths):
    images = np.ndarray(shape=(len(paths), height*width), dtype=np.float64)
    test_list=[]
    for i in range(len(paths)):
        path= direc+'/'+ paths[i]
        read_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(read_image, (width, height))
        images[i,:] = np.array(resized_image, dtype='float64').flatten()
        test_list.append(resized_image)
    # print(images.shape)

    # print(len(test_list))
    return (images)


# In[8]:


def accuracy(predictions, test_labels):
    l = len(test_labels)
    acc = sum([predictions[i]==test_labels[i] for i in range(l)])/l
    # print('The testing accuracy is: ' + str(acc*100) + '%')
    return acc


# In[58]:


total_images=0
height=80
width=70
images_paths, labels= get_filepaths(training_path)
test_paths, test_labels= get_filepaths(test_path)


# In[59]:


y = to_categorical(test_labels, num_classes = 42)[:,1:]


# In[11]:
def read_img(read_imagee):
    image = np.ndarray(shape=(1, height*width), dtype=np.float64)
    test_list=[]
    read_imagee = cv2.cvtColor(read_imagee, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(read_imagee, (width, height))
    print(resized_image.shape)
    resized_image = np.array(resized_image, dtype='float64').flatten()
    
    image[0,:] =  resized_image
    # print(image.shape)
    return (image)

training_images = data_prep(training_path, images_paths)
# print('-'*60)
test_images = data_prep(test_path, test_paths)
# test_image_path = read_img("E:/cv_tasks/task5/final2/zeft/CV_Final_Project-main/data/test/179_18.jpg")


# In[12]:
#Creating a machine learning to get the probabilities for each class
model = RandomForestClassifier()
model.fit(training_images, labels)

prob_vector = model.predict_proba(test_images)
prediction = model.predict(test_images)
acc = accuracy(prediction, test_labels)
def test_imgs(test_image_path):
    pred = model.predict(test_image_path)####
    idx = np.where(labels == pred[0])[0]
    for i in range(len(idx)):
        img = training_images[idx[i]].reshape(height,width)
        plt.subplot(2,4,1+i)
        plt.imshow(img, cmap='gray')
    plt.savefig('E:/cv_tasks/task5/github/CV_Final_Project/Images/Recognized_Multiple.png')
    # plt.show()

# In[14]:
fpr = {}
tpr = {}
thresh ={}
for i in range(1,7):
    fpr[i], tpr[i], thresh[i] = roc_curve(test_labels, prob_vector[:,i], pos_label=i)


# In[47]:


thresholds = np.arange(0.05, 1.05, 0.05)
subjects = np.arange(1, 42, 1)


# In[69]:


def roc(probabilities, y_test, thresholds):
    roc = np.array([])
    for threshold in thresholds:
        threshold_vector = np.greater_equal(probabilities, threshold).astype(int)
        results = np.where(y_test == 1)[0]
        tp, fp, tn, fn = 0,0,0,0
        for i in range(len(threshold_vector)):
            if i in results:
                #which means that the actual value at these indices is 1
                if threshold_vector[i] == 1:
                    tp +=1
                else:
                    fn +=1
            else:
                if threshold_vector[i] == 0:
                    tn +=1
                elif threshold_vector[i] == 1:
                    fp +=1
                    
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        roc = np.append(roc, [fpr, tpr])
    roc = roc.reshape(-1, 2)
    cm = np.array([[fn,tp],[tn,fp]])
    return(roc,cm)


# def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import itertools

#     accuracy = np.trace(cm) / float(np.sum(cm))
#     misclass = 1 - accuracy

#     if cmap is None:
#         cmap = plt.get_cmap('plasma')

#     plt.figure(figsize=(8, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()

#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names, rotation=45)
#         plt.yticks(tick_marks, target_names)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         cm = np.around(cm, decimals=2)


#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.2f}".format(cm[i, j]),
#                      horizontalalignment="center", fontsize = 20,
#                      color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")


#     plt.tight_layout()
#     plt.ylabel('True label', fontsize = 20)
#     plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize = 20)
#     plt.savefig('D:/zeft/CV_Final_Project-main/Images/CM.png')
#     plt.show()
    
# In[68]:

# class_no= 4
def draw_CM(CM):
    target_names = ['same person', 'different person']
    DetaFrame_cm = pd.DataFrame(CM, range(2), range(2))
    # plt.plot(DetaFrame_cm)
    snNew.heatmap(DetaFrame_cm, annot=True)
    fig, ax = plt.subplots(figsize=(5,5)) 
    snNew.heatmap(DetaFrame_cm, annot=True, linewidths=.7, ax=ax) # Sample figsize in inches
    
    # saved=mpimg.imsave('E:/cv_tasks/task5/final2/zeft/CV_Final_Project-main/Images/CM.png', matrix)
    plt.savefig('E:/cv_tasks/task5/github/CV_Final_Project/Images/CM.png')
    # plt.show()

    
def draw_ROC(ROC):
    plt.clf()
    plt.plot(ROC[:,0],ROC[:,1],color='#0F9D58')
    plt.title('ROC Curve',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    
    plt.savefig('E:/cv_tasks/task5/github/CV_Final_Project/Images/ROC.png')

    # plt.show()

# plot_confusion_matrix(CM, target_names)
# In[65]:

# ROC = roc(prob_vector[:,4], y[:,4], thresh[6],4)
# plt.plot(ROC[:,0],ROC[:,1],color='#0F9D58')
# plt.title('ROC Curve',fontsize=20)
# plt.xlabel('False Positive Rate',fontsize=16)
# plt.ylabel('True Positive Rate',fontsize=16)
# plt.show()


# In[30]:


# fpr = {}
# tpr = {}
# thresh ={}
# colors = ['green', 'blue', 'orange', 'black', 'pink', 'yellow', 'gray', 'red']
# plt.figure(figsize=(10,5))
# for i in range(1,7):
#     fpr[i], tpr[i], thresh[i] = roc_curve(test_labels, prob_vector[:,i], pos_label=i)
#     plt.plot(fpr[i], tpr[i], linestyle='--',color=colors[i], label='Class {} vs Rest'.format(i))
# # plt.figure(figsize=(15, 7))
# plt.title('Multiclass ROC curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive rate')
# plt.legend(loc='best')
# plt.savefig('Multiclass ROC',dpi=300)

