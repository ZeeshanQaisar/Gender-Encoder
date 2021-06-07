from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import os
import numpy as np
import pandas as pd


#Calculating Distances from d1 to d8

def distance(mouth, nose, lefteye, righteye):
 d1 = dist.euclidean(mouth[0], mouth[6])
 d2 = dist.euclidean(mouth[3], mouth[9])
 d3 = dist.euclidean(mouth[0], nose[4])
 d4 = dist.euclidean(mouth[6], nose[8])
 d5 = dist.euclidean(mouth[0], righteye[0])
 d6 = dist.euclidean(mouth[6], lefteye[3])
 d7 = dist.euclidean(mouth[2], mouth[10])
 d8 = dist.euclidean(mouth[4], mouth[8])

 return d1,d2,d3,d4,d5,d6,d7,d8

shape_predictor = '/content/drive/MyDrive/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS['nose']
(eLStart, eLEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(eRStart, eREnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']


directory = '/content/drive/MyDrive/ReFinedDataset'
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        total= directory+'/'+filename
        frame=cv2.imread(total)
        frame=imutils.resize(frame, width=450)
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects=detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            mouth = shape[mStart:mEnd]
            nose = shape[nStart:nEnd]
            lefteye = shape[eLStart:eLEnd]
            rightteye = shape[eRStart:eREnd]
            array=[distance(mouth, nose, lefteye, rightteye)]
          #  print(array)
          #Save Distances into CSV File
            results=pd.DataFrame(array)
            results.to_csv('result6.csv',mode='a',header='True',index='false')




			#RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler 

path =('G:/Masters_Subjects/Context_Security/gender Encoder Project/result.csv') #CSV PATH
dataset =pd.read_csv(path)

X = dataset.iloc[:,1:8]  #X equal to all distances
y = dataset.iloc[:, 0].values    #Y equal to gender

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sc= StandardScaler()   #DataPreProcessing Standarlization 
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators =50,random_state= 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

#Naive Bayes Classifier

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

path ='G:/Masters_Subjects/Context_Security/gender Encoder Project/result.csv'  #CSV PATH
dataset = pd.read_csv(path)
dataset.head()

X = dataset.iloc[:,1:8].values
y = dataset.iloc[:, 0].values

features_train, features_test, target_train, target_test = train_test_split(X,
y,
test_size = 0.3)

sc= StandardScaler() #DataPreProcessing Standarlization
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)

model = GaussianNB()
model.fit(features_train, target_train)

pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
result=classification_report(target_test, pred)
print(result)
print(accuracy)



#SVM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC


url = 'G:/Masters_Subjects/Context_Security/gender Encoder Project/result.csv'
dataset = pd.read_csv(url)  
dataset.head()

#y = df.gender
X = dataset.iloc[:,1:8].values
y = dataset.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc= StandardScaler() #DataPreProcessing Standarlization
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

SVC(C=1.0, cache_size=200, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, 
  shrinking=True, tol=0.001)

predictions = model.predict(X_test)
#print(predictions)

percentage = model.score(X_test, y_test)
from sklearn.metrics import confusion_matrix
res = confusion_matrix(y_test, predictions)
print("Confusion Matrix")
print(res)
print(f"Test Set: {len(X_test)}")
result =classification_report(y_test, predictions)
print(result)
print(f"Accuracy = {percentage*100} %")