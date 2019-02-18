from PIL import Image
import numpy as np
import csv
import cv2,glob
import matplotlib.pyplot as plt
from sklearn import metrics

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
li={"ten","hun","fifty","test"}
for i in li:
    if li!="test":
        images=glob.glob("C:/Users/MY HP/Desktop/"+i+"/*.jpg")
        for image in images:
            img=cv2.imread(image,0)
            re=cv2.resize(img,(150,100))
            cv2.imwrite(image,re)
def createFileList(myDir, format='.jpg'):
    fileList = []
    for root, dirs, files in os.walk(myDir, topdown=False):
        print(root,dirs,files)
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

def fun(mydir,myFileList):
    for file in myFileList:
        # print(file)
        img_file = Image.open(file)
        # img_file.show()

        # get original image parameters...
        width, height = img_file.size
        format = img_file.format
        mode = img_file.mode

        # Make image Greyscale
        img_grey = img_file.convert('L')
        img_grey.save('result.png')
        img_grey.show()

        # Save Greyscale values
        value = np.asarray(img_file.getdata(), dtype=np.int).reshape((img_file.size[1], img_file.size[0]))
        print(value)
        value = value.flatten()
        print(value)
        with open(mydir,'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)
myFileList = createFileList("C:/Users/MY HP/Desktop/ten")
myFileList2 = createFileList("C:/Users/MY HP/Desktop/hun")
myFileList3 = createFileList("C:/Users/MY HP/Desktop/fifty")
myFileList4 = createFileList("C:/Users/MY HP/Desktop/test")
fun("C:/Users/MY HP/Desktop/ten.csv",myFileList)
fun("C:/Users/MY HP/Desktop/hun.csv",myFileList2)
fun("C:/Users/MY HP/Desktop/fifty.csv",myFileList3)
fun("C:/Users/MY HP/Desktop/test.csv",myFileList4)

value[]
i=0
csv_merge = open("C:/Users/MY HP/Desktop/final.csv", 'w')
for file in li:
    if file!="test":
       csv_in = open("C:/Users/MY HP/Desktop/ten/"+li+".csv")
       if file=="ten":
           a=10
       else if file=="hun":
           a=100
       else:
            a=50;
       for line in csv_in:
          csv_merge.write(line)
          csv_merge.write(a)
       csv_in.close()
       csv_merge.close()
csv_header=['label']
myFileList4.write(csv_header)
csv_merge.write(csv_header)
data = pd.read_csv("C:/Users/MY HP/Desktop/final.csv",encoding='latin-1')
Y=data.label
X = data.drop("label", axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2, random_state = 20)
data = pd.read_csv("C:/Users/MY HP/Desktop/test.csv",encoding='latin-1')
Y1=data.label
X1 = data.drop("label", axis=1)
from sklearn.svm import SVC  #92.85
classifier = SVC(kernel='poly')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
value[i] = classifier.predict(X1)
i++
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("svm -poly")
print(accuracy_score(y_test,y_pred)*100)

classifier = SVC(kernel='rbf') #95.05
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
value[i] = classifier.predict(X1)
i++
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("svm -rbf")
print(accuracy_score(y_test,y_pred)*100)

#data = pd.read_csv("C:/Users/MY HP/Desktop/test2.csv",encoding='latin-1')
#Y=data.label
#X = data.drop("label", axis=1)
#y_pred = classifier.predict(X)
#print(y_pred)
#print("svm -poly tested")
#print(accuracy_score(Y,y_pred)*100)

from sklearn.tree import DecisionTreeClassifier #85.2
algo4=DecisionTreeClassifier()
algo4.fit(X_train,y_train)
y_pred=algo4.predict(X_test)
value[i] = algo4.predict(X1)
i++
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("DecisionTree")
print(accuracy_score(y_test,y_pred)*100)
y_pred_proba = algo4.predict_proba(X_test)[::,1]
fpr, tpr,_ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

from sklearn.neighbors import KNeighborsClassifier #78.57
from sklearn.model_selection import KFold,cross_val_score

algo5=KNeighborsClassifier()
cross_val_score(algo5,X_train,y_train,cv=KFold(10))
algo5.fit(X_train,y_train)
y_pred = algo5.predict(X_test)
value[i] = algo5.predict(X1)
i++
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("KNN")
print(accuracy_score(y_test,y_pred)*100)
y_pred_proba = algo5.predict_proba(X_test)[::,1]
fpr, tpr,_ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

classifier = SVC(kernel='linear')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
value[i] = classifier.predict(X1)
i++
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("svm linear")
print(accuracy_score(y_test,y_pred)*100)

from sklearn.naive_bayes import GaussianNB  #85.89
algo6=GaussianNB()
algo6.fit(X_train,y_train)
y_pred = algo6.predict(X_test)
value[i] = algo6.predict(X1)
i++
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Gaussian")
print(accuracy_score(y_test,y_pred)*100)
y_pred_proba = algo6.predict_proba(X_test)[::,1]
fpr, tpr,_ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

from sklearn.ensemble import RandomForestClassifier  #71.42 
algo7=RandomForestClassifier(max_depth=5,random_state=10)
algo7.fit(X_train,y_train)
y_pred = algo7.predict(X_test)
value[i] = algo7.predict(X1)
i++
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("RandomForest")
print(accuracy_score(y_test,y_pred)*100)
y_pred_proba = algo7.predict_proba(X_test)[::,1]
fpr, tpr,_ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

from sklearn.ensemble import AdaBoostClassifier #93.4
model = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
value[i] = model.predict(X1)
i++
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("AdaBoostClassifier")
print(accuracy_score(y_test,y_pred)*100)
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr,_ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


from mlxtend.classifier import StackingClassifier #94.8
clf1 =AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
clf2 =  LogisticRegression()
clf3 = SVC(kernel='poly')
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                          meta_classifier=lr)
sclf.fit(X_train,y_train)
y_pred = sclf.predict(X_test)
value[i] = sclf.predict(X1)
i++
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("stacking")
print(accuracy_score(y_test,y_pred)*100)
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr,_ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
max=0
res=value[0]
for i in value: 
    freq = value.count(i) 
    if freq > max: 
        max = freq 
        res = i
if(res==5):
    print("FIVE")
else if res==10:
    print("TEN")
else if res==50:
    print("FIFTY")
else if res==100:
    print("HUNDRED")
