#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import math
from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap


# In[2]:


# loading data
data = pd.read_json("renttherunway_final_data.json", lines=True)
#print(data)


# In[3]:


#cleaning dataset

#dealing with empty cells 
data.dropna(inplace=True)

#cleaning columns 
data['weight'] = data['weight'].str.replace('lbs', '')
data['weight'] = pd.to_numeric(data['weight'], errors='coerce')
data['height'] = data['height'].str.replace("'", '.')
data['height'] = data['height'].str.replace(" ", '')
data['height'] = data['height'].str.replace('"', '')
data['height'] = pd.to_numeric(data['height'], errors='coerce')
data['bust size'] = data['bust size'].str.replace('+', '')
data['bust size'] = data['bust size'].str.replace('ddd/e', 'e')


def heightConvert(v):
    fraction, whole = math.modf(v)
    f = (fraction*10)/12.0
    w = whole+f
    return w*30.48

data['height'] = data['height'].apply(heightConvert)

#cleaning of category e.g. skirt spelt wrong


# print(data['rented for'].unique())
# print(data['body type'].unique())

#deleting "for"
print(data[data.category == 'for'].shape[0])
data = data[data.category != 'for']

data['category'] = data['category'].str.replace('culottes', 'culotte')
data['category'] = data['category'].str.replace('skort', 'skirt')
data['category'] = data['category'].str.replace('skirts', 'skirt')
data['category'] = data['category'].str.replace('tee', 't-shirt')
data['category'] = data['category'].str.replace('sweatershirt', 'sweatshirt')
data['category'] = data['category'].str.replace('pants', 'pant')
data['category'] = data['category'].str.replace('caftan', 'kaftan')
data['category'] = data['category'].str.replace('legging', 'leggings')
data['category'] = data['category'].str.replace('leggingss', 'leggings')
data['category'] = data['category'].str.replace('trouser', 'trousers')
data['category'] = data['category'].str.replace('trouserss', 'trousers')



#drops some column won't need for my basic model
data.drop(columns=['user_id', 'item_id', 'review_text', 'review_date', 'review_summary'], inplace=True)

print(data['category'].nunique())
print(data['category'].unique())
print(data.shape[0])

target = data['fit'].unique()
print(target)
print(data['body type'].unique())
print(data['age'].unique())
print(data['bust size'].unique())
print(data['rented for'].unique())

print(data)
# print(data.dtypes)

visCopy = data
subsetCopy = data


# In[4]:


# encoding && making new features and normialization/standardisation
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


#encoding
bustsize =pd.get_dummies(data["bust size"],prefix='bust size', drop_first=True)
category = pd.get_dummies(data["category"],prefix='category', drop_first=True)
rent = pd.get_dummies(data["rented for"],prefix='rented for', drop_first=True)
body = pd.get_dummies(data["body type"],prefix='body type', drop_first=True)
fit = data["fit"]



data = pd.concat([data,bustsize],axis=1)
data = pd.concat([data,category],axis=1)
data = pd.concat([data,rent],axis=1)
data = pd.concat([data,body],axis=1)
# data = pd.concat([data,fit],axis=1)
# data.drop(['bust size', 'category','rented for','body type', 'fit'],axis=1, inplace=True)


data.drop(['bust size', 'category','rented for','body type', ],axis=1, inplace=True)


#print(data)

#Standarisation/normailsation 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


#data.loc[:, 'weight':'age'] = StandardScaler().fit_transform(data.loc[:, 'weight':'age'])

norm = MinMaxScaler().fit(data.loc[:, 'weight':'age'])
data.loc[:, 'weight':'age'] = norm.transform(data.loc[:, 'weight':'age'])










#split data 

# print(data[data.fit == 'fit'].shape[0])
# print(data[data.fit == 'large'].shape[0])
# print(data[data.fit == 'small'].shape[0])

from sklearn.model_selection import train_test_split

trainingData, testData= train_test_split(data, test_size=0.25, random_state=0, stratify=fit)
trainingDataVis, testDataVis = train_test_split(visCopy, test_size=0.25, random_state=42)

# print(trainingData[trainingData.fit_large == 1].shape[0])
# print(trainingData[trainingData.fit_small == 1].shape[0])
# print(trainingData.shape)

ohe = OneHotEncoder()

trainingInput = trainingData.loc[:, 'weight':'body type_straight & narrow']
trainingOutput = trainingData.loc[:, 'fit']
testInput = testData.loc[:, 'weight':'body type_straight & narrow']
testOutput = testData.loc[:, 'fit']

trainingOutput = ohe.fit_transform(trainingOutput.values.reshape(-1,1)).toarray()
testOutput = ohe.fit_transform(testOutput.values.reshape(-1,1)).toarray()

print(ohe.categories_)

print(trainingOutput)

print(trainingInput)
# print(trainingOutput)
# print(testInput)
# print(testOutput)


# In[5]:


#visualisation and analyses

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)


#visBody = sns.catplot(x="body type", hue="fit", kind="count", data=trainingDataVis)
#visBody.set_xticklabels(rotation=40, ha="right")

# visCat = sns.catplot(x="category", hue="fit", kind="count", data=trainingDataVis, aspect=3, height=7)
# visCat.set_xticklabels(rotation=40, ha="right")

# visRent = sns.catplot(x="rented for", hue="fit", kind="count", data=trainingDataVis, aspect=1, height=7)
# visRent.set_xticklabels(rotation=40, ha="right")

# visBust = sns.catplot(x="bust size", hue="fit", kind="count", data=trainingDataVis, aspect=2, height=8)
# visBust.set_xticklabels(rotation=90, ha="right")

# visAge = sns.catplot(x="age", hue="fit", kind="count", data=trainingDataVis, aspect=2, height=9)
# visAge.set_xticklabels(rotation=90, ha="right")


#plt.figure(figsize=(10,7))
#visRateSize = sns.scatterplot(x="rating", y="size", hue="fit",data=trainingDataVis)


#visWeightHeight = sns.scatterplot(x="weight", y="height", hue="fit",data=trainingDataVis)



# In[10]:


#selecting and training a regression model 

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight




#np.unique(trainingOutput.argmax(axis=1))





# print()
# print('Balanced current class weight-  (Order)Fit Large Small')
# print(class_weight.compute_class_weight('balanced', [0,1,2], trainingOutput.argmax(axis=1)))



# logr = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced'))
# logr.fit(trainingInput, trainingOutput)

# print('Basic Logistic Regression Accuracy: ', accuracy_score(trainingOutput, logr.predict(trainingInput)))
# print('Basic Logistic Regression Cross validation ', np.mean(cross_val_score(logr, trainingInput, trainingOutput, cv=5)))

# #confusion matrix log
# print('Confusion matrix for balanced classes- Logistic regression')
# print(confusion_matrix(trainingOutput.argmax(axis=1), logr.predict(trainingInput).argmax(axis=1)))


rf = RandomForestClassifier(criterion='gini', class_weight='balanced', max_depth=77)
rf.fit(trainingInput, trainingOutput)

print (roc_auc_score(testOutput, rf.predict(testInput), multi_class='ovr'))

# print('Confusion matrix for balanced classes- Random Forest')
# print(confusion_matrix(trainingOutput.argmax(axis=1), rf.predict(trainingInput).argmax(axis=1)))
# print(rf.class_weight)

# print('Randomforest Accuracy: ', accuracy_score(trainingOutput, rf.predict(trainingInput)))
# print('Randomforest Cross validation ', np.mean(cross_val_score(rf, trainingInput, trainingOutput, cv=5)))

# print('Randomforest Test Accuracy: ', accuracy_score(testOutput, rf.predict(testInput)))
# print('Randomforest Test Cross validation ', np.mean(cross_val_score(rf, testInput, testOutput, cv=5)))



# rfPara = [{'max_depth': [78, 76, 77]}]

# grid_searchLog = GridSearchCV(rf, rfPara, cv=5, return_train_score=True)
# grid_searchLog.fit(trainingInput, trainingOutput)

# print(grid_searchLog.best_params_)

# nb = MultinomialNB() ## reemove one vs res and .argmax trainingdata on accuracy 
# nb.fit(trainingInput, trainingOutput.argmax(axis=1))

# print('Confusion matrix for balanced classes- Multinomial NB')
# print(confusion_matrix(trainingOutput.argmax(axis=1), nb.predict(trainingInput).argmax(axis=1)))

# print('Multinomial NB Accuracy: ', accuracy_score(trainingOutput.argmax(axis=0), nb.predict(trainingInput).argmax(axis=0)))
# print('Multinomial NB Cross validation ', np.mean(cross_val_score(nb, trainingInput, trainingOutput.argmax(axis=1), cv=5)))

# voting_clf = OneVsRestClassifier(VotingClassifier(estimators=[('logr', logr), ('RF', rf), ('NB', nb)], voting='soft'))
# voting_clf.fit(trainingInput,trainingOutput)


# print('Voting Classifier Accuracy: ', accuracy_score(trainingOutput, voting_clf.predict(trainingInput)))
# print('Voting Classifier Cross validation ', np.mean(cross_val_score(voting_clf, trainingInput, trainingOutput, cv=3)))




# In[7]:


from sklearn import preprocessing

def transform(weight1, rating1, height1, size1, age1, bust1, cat1, rented1, bodyT1):
    d1 = pd.get_dummies(pd.DataFrame({'bust size': [bust1]}))
    d2 = pd.get_dummies(pd.DataFrame({'category': [cat1]}))
    d3 = pd.get_dummies(pd.DataFrame({'rented for': [rented1]}))
    d4 = pd.get_dummies(pd.DataFrame({'body type': [bodyT1]}))

    a =[weight1,rating1, height1,size1,age1]
    a1 = np.array(d1.reindex(columns = bustsize.columns, fill_value = 0))
    a2 = np.array(d2.reindex(columns = category.columns, fill_value = 0))
    a3 = np.array(d3.reindex(columns = rent.columns, fill_value = 0))
    a4 = np.array(d4.reindex(columns = body.columns, fill_value = 0))
    
#data.loc[:, 'weight':'age'] = norm.transform(data.loc[:, 'weight':'age'])
    final = np.concatenate((a1,a2,a3,a4),axis=1)
    #print(a)
    
    b = np.array(a)
    #print(b)
    normalised = norm.transform(b.reshape(1,-1))
    #print(normalised)
    result = np.hstack((normalised.flatten(), final.flatten()))
    
    return result

#result = transform(137, 10, 172, 14, 28, "34d", "romper", "vacation", "hourglass")
#print(result)
#print(d1)

#print(ohe.inverse_transform([[1,0,0]]))
#print(ohe.inverse_transform(rf.predict(result.reshape(1,-1))))



# In[8]:


import pickle

# Saving model to disk
pickle.dump(rf, open('model.pkl','wb'))


# In[9]:


from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(start, end):
    return((start[0]+end[0])*0.5,(start[1]+end[1])*0.5 )


# In[ ]:


from flask_ngrok import run_with_ngrok
import operator
import sys

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)
#run_with_ngrok(app)
Bootstrap(app)



result = transform(137, 10, 172, 14, 28, "34d", "romper", "vacation", "hourglass")
resultTest = transform(17, 10, 12, 4, 28, "28d", "romper", "vacation", "hourglass")
output1 = model.predict_proba(result.reshape(1,-1))
output2 = model.predict_proba(resultTest.reshape(1,-1))
# print(model.predict(result.reshape(1,-1)))
# print(result)
# print(resultTest)
# print(output1)
# print(output2)
#print(output1[0][0][1])

#deal with index page of web page
@app.route('/')
def index():
    #return "boss boy"
    return render_template('/diss.html')
pred = "Unfortunately, no available size will fit you."

#deal with output of prediction to web page
@app.route('/predict', methods=['POST'])
def predict():
    resultList = {}
    for x in range(4, 22, 2):
        result = transform(request.form['weight'], 10, request.form['height'], x, request.form['age'], request.form['bustsize'],
                       request.form['product'], request.form['rented'], request.form['bodytype'])
        output1 = model.predict_proba(result.reshape(1,-1))
        #print(output1)
        if(output1[0][0][1] > 0.5):
            resultList.update({x:output1[0][0][1]})
        #output= ohe.inverse_transform(model.predict(result.reshape(1,-1)))
        
        print(resultList)
        if(x == 20):
            
            if(len(resultList) != 0):
                global pred
                pred = max(resultList, key=resultList.get)
    
    
    return render_template('/diss.html', prediction=pred)

#deals with the computer vision computation between server and client
@app.route('/cv', methods=['POST'])
def cv():
    
    image = cv2.imread(request.form['filename'])
    
    grayImageNoBlur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.GaussianBlur(grayImageNoBlur, (7,7), 0) 


# does edge detection, then dilation + erosion to close gaps in between object edges
    edgedImage = cv2.Canny(grayImage, 100, 100) # edge detection

    edgedImage = cv2.dilate(edgedImage, None, iterations=1) #i can add a kernel here, try it
    edgedImage = cv2.erode(edgedImage, None, iterations=1)


 # find contours in the edge map
    cnts = cv2.findContours(edgedImage.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   
    cnts = imutils.grab_contours(cnts)
    
# sort the contours from left-to-right and initialize the 'pixels per metric' variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    
        #** a lot of work here**********
# loop over the contours individually
    originalImage = image.copy()
    for c in cnts:
    # if the contour is not a similar size of reference object, ignore it
        if cv2.contourArea(c) < 18800: #or cv2.contourArea(c) > 150 :
            continue
    # compute the rotated bounding box of the contour
        print(cv2.contourArea(c))
        boundaryBox = cv2.minAreaRect(c)
        boundaryBox = cv2.cv.BoxPoints(boundaryBox) if imutils.is_cv2() else cv2.boxPoints(boundaryBox)
        boundaryBox = np.array(boundaryBox, dtype="int")
    
    
    # order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order, 
    # draws bounding box
    
        boundaryBox = perspective.order_points(boundaryBox)
        cv2.drawContours(originalImage, [boundaryBox.astype("int")], -1, (139, 0, 0), 2)
        plt.imshow(originalImage)
    
    # loop over the original points and draw midpoints
        for (x, y) in boundaryBox:
            cv2.circle(originalImage, (int(x), int(y)), 5, (11, 64, 255), -1)
        (topLeft, topRight, bottomRight, bottomLeft) = boundaryBox
        (topLeftTopRightX, topLeftTopRightY) = midpoint(topLeft, topRight)
        (bottomLeftBottomRightX, bottomLeftBottomRightY) = midpoint(bottomLeft, bottomRight)
    
        (topLeftBottomLeftX, topLeftBottomLeftY) = midpoint(topLeft, bottomLeft)
        (topRightBottomRightX, topRightBottomRightY) = midpoint(topRight, bottomRight)
    
        cv2.circle(originalImage, (int(topLeftTopRightX), int(topLeftTopRightY)), 5, (255, 255, 0), -1)
        cv2.circle(originalImage, (int(bottomLeftBottomRightX), int(bottomLeftBottomRightY)), 5, (255, 255, 0), -1)
        cv2.circle(originalImage, (int(topLeftBottomLeftX), int(topLeftBottomLeftY)), 5, (255, 255, 0), -1)
        cv2.circle(originalImage, (int(topRightBottomRightX), int(topRightBottomRightY)), 5, (255, 255, 0), -1)
    
        cv2.line(originalImage, (int(topLeftTopRightX), int(topLeftTopRightY)), (int(bottomLeftBottomRightX), int(bottomLeftBottomRightY)),
                 (255, 69, 0), 2)
        cv2.line(originalImage, (int(topLeftBottomLeftX), int(topLeftBottomLeftY)), (int(topRightBottomRightX), int(topRightBottomRightY)),
                 (255, 69, 0), 2)
    
    #heightDistance = dist.euclidean((topLeftTopRightX, topLeftTopRightY), (bottomLeftBottomRightX, bottomLeftBottomRightY)) # height distance
        widthDistance = dist.euclidean((topLeftBottomLeftX, topLeftBottomLeftY), (topRightBottomRightX, topRightBottomRightY))
    
        if pixelsPerMetric is None:
            pixelsPerMetric = widthDistance/5.512 #5.512 is the refrence object real width
    
#     realHeight =  heightDistance / pixelsPerMetric
#     realWidth = widthDistance / pixelsPerMetric
    
#     cv2.putText(originalImage, "{:.1f}in".format(realHeight),(int(topLeftTopRightX - 15), int(topLeftTopRightY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65, (255, 255, 255), 2)
#     cv2.putText(originalImage, "{:.1f}in".format(realWidth),(int(topRightBottomRightX + 10), int(topRightBottomRightY)), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65, (255, 255, 255), 2)
        break;
    
    #Viola jones 
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    lowerBodyCascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
    
    face = faceCascade.detectMultiScale(grayImageNoBlur, scaleFactor=1.1, minNeighbors=5)
    lowerbody = lowerBodyCascade.detectMultiScale(grayImageNoBlur, scaleFactor=1.01, minNeighbors=5)

    
# get faces and lowerbody points for measurement
    for (x,y,w,h) in face:
        cv2.rectangle(originalImage, (x,y), (x+w,y+h), (255,253,208), 2)
        startX = x
        startY = y

# for (x,y,w,h) in lowerbody:
#     cv2.rectangle(originalImage, (x,y), (x+w,y+h), (255,253,208), 2)
#     endX1 = x+w
#     endY1 = y+h

    areas = [w*h for x,y,w,h in lowerbody]
    biggestIndex = np.argmax(areas)
    biggest = lowerbody[biggestIndex]
    print(biggest)
    x = biggest[0]
    y= biggest[1]
    w = biggest[2]
    h = biggest[3]
#for (x,y,w,h) in biggest:
    cv2.rectangle(originalImage, (x,y), (x+w,y+h), (255,253,208), 2)
    endX1 = x+w
    endY1 = y+h

# (startX, startY, endX, endY) = face
# (startX1, startY1, endX1, endY1) = lowerbody

    cv2.line(originalImage, (startX, startY), (startX, endY1),(0,100,0) ,3)
    cv2.line(originalImage, (startX, endY1), (endX1, endY1),(0,100,0) ,3)
    cv2.circle(originalImage, (startX, endY1), 7, (231,84,128), -1)

# using the pixelperMetric to calculate the real height
    heightPixelDistance = endY1 - startY
    realHeight = heightPixelDistance / pixelsPerMetric


    
    
        
    
    


    
    
    return render_template('/diss.html', cv=realHeight*2.54)
                                  
if __name__ == "__main__":
    app.run()
    


# In[8]:


# Building models for Subset of product rather than all products
      
dataDress = subsetCopy[(subsetCopy['category'] == 'romper') |(subsetCopy['category'] == 'dress') |(subsetCopy['category'] == 'sheath')|(subsetCopy['category'] == 'shirtdress')|(subsetCopy['category'] == 'gown')|(subsetCopy['category'] == 'jumpsuit')|(subsetCopy['category'] == 'shift')|(subsetCopy['category'] == 'mini')|(subsetCopy['category'] == 'maxi')|(subsetCopy['category'] == 'cape')|(subsetCopy['category'] == 'duster')|(subsetCopy['category'] == 'ballgown')|(subsetCopy['category'] == 'frock')|(subsetCopy['category'] == 'print')|(subsetCopy['category'] == 'midi')|(subsetCopy['category'] == 'peacoat')|(subsetCopy['category'] == 'kaftan')|(subsetCopy['category'] == 'overalls')|(subsetCopy['category'] == 'combo')|(subsetCopy['category'] == 'blouson')|(subsetCopy['category'] == 'kimono')]               


dataJacket = subsetCopy[(subsetCopy['category'] == 'jacket') |(subsetCopy['category'] == 'suit') |(subsetCopy['category'] == 'coat')|(subsetCopy['category'] == 'trench')|(subsetCopy['category'] == 'bomber')|(subsetCopy['category'] == 'blazer')|(subsetCopy['category'] == 'poncho')|(subsetCopy['category'] == 'overcoat')|(subsetCopy['category'] == 'parka')]               


dataTop = subsetCopy[(subsetCopy['category'] == 'sweater') |(subsetCopy['category'] == 'top') |(subsetCopy['category'] == 'shirt')|(subsetCopy['category'] == 'blouse')|(subsetCopy['category'] == 'vest')|(subsetCopy['category'] == 'tank')|(subsetCopy['category'] == 'tunic')|(subsetCopy['category'] == 'cardigan')|(subsetCopy['category'] == 'print')|(subsetCopy['category'] == 'knit')|(subsetCopy['category'] == 'sweatshirt')|(subsetCopy['category'] == 't-shirt')|(subsetCopy['category'] == 'henley')|(subsetCopy['category'] == 'blouson')|(subsetCopy['category'] == 'pullover')|(subsetCopy['category'] == 'turtleneck')|(subsetCopy['category'] == 'hoodie')|(subsetCopy['category'] == 'cami')|(subsetCopy['category'] == 'crewneck')|(subsetCopy['category'] == 'buttondown')]               


dataBottom = subsetCopy[(subsetCopy['category'] == 'leggings') |(subsetCopy['category'] == 'skirt') |(subsetCopy['category'] == 'pant')|(subsetCopy['category'] == 'culotte')|(subsetCopy['category'] == 'midi')|(subsetCopy['category'] == 'trousers')|(subsetCopy['category'] == 'overalls')|(subsetCopy['category'] == 'jogger')|(subsetCopy['category'] == 'tight')|(subsetCopy['category'] == 'jeans')|(subsetCopy['category'] == 'sweatpant')]               



#encoding
bustsize =pd.get_dummies(dataDress["bust size"],prefix='bust size', drop_first=True)
category = pd.get_dummies(dataDress["category"],prefix='category', drop_first=True)
rent = pd.get_dummies(dataDress["rented for"],prefix='rented for', drop_first=True)
body = pd.get_dummies(dataDress["body type"],prefix='body type', drop_first=True)
fitDress = dataDress["fit"]
#fit = pd.get_dummies(dataDress["fit"],prefix='fit', drop_first=True)
dataDress = pd.concat([dataDress,bustsize],axis=1)
dataDress = pd.concat([dataDress,category],axis=1)
dataDress = pd.concat([dataDress,rent],axis=1)
dataDress = pd.concat([dataDress,body],axis=1)
#dataDress = pd.concat([dataDress,fit],axis=1)
dataDress.drop(['bust size', 'category','rented for','body type'],axis=1, inplace=True)

bustsize =pd.get_dummies(dataJacket["bust size"],prefix='bust size', drop_first=True)
category = pd.get_dummies(dataJacket["category"],prefix='category', drop_first=True)
rent = pd.get_dummies(dataJacket["rented for"],prefix='rented for', drop_first=True)
body = pd.get_dummies(dataJacket["body type"],prefix='body type', drop_first=True)
fitJacket = dataJacket["fit"]
#fit = pd.get_dummies(dataJacket["fit"],prefix='fit', drop_first=True)
dataJacket = pd.concat([dataJacket,bustsize],axis=1)
dataJacket = pd.concat([dataJacket,category],axis=1)
dataJacket = pd.concat([dataJacket,rent],axis=1)
dataJacket = pd.concat([dataJacket,body],axis=1)
#dataJacket = pd.concat([dataJacket,fit],axis=1)
dataJacket.drop(['bust size', 'category','rented for','body type'],axis=1, inplace=True)

bustsize =pd.get_dummies(dataTop["bust size"],prefix='bust size', drop_first=True)
category = pd.get_dummies(dataTop["category"],prefix='category', drop_first=True)
rent = pd.get_dummies(dataTop["rented for"],prefix='rented for', drop_first=True)
body = pd.get_dummies(dataTop["body type"],prefix='body type', drop_first=True)
fitTop = dataTop["fit"]
#fit = pd.get_dummies(dataTop["fit"],prefix='fit', drop_first=True)
dataTop = pd.concat([dataTop,bustsize],axis=1)
dataTop = pd.concat([dataTop,category],axis=1)
dataTop = pd.concat([dataTop,rent],axis=1)
dataTop = pd.concat([dataTop,body],axis=1)
#dataTop = pd.concat([dataTop,fit],axis=1)
dataTop.drop(['bust size', 'category','rented for','body type'],axis=1, inplace=True)

bustsize =pd.get_dummies(dataBottom["bust size"],prefix='bust size', drop_first=True)
category = pd.get_dummies(dataBottom["category"],prefix='category', drop_first=True)
rent = pd.get_dummies(dataBottom["rented for"],prefix='rented for', drop_first=True)
body = pd.get_dummies(dataBottom["body type"],prefix='body type', drop_first=True)
fitBottom = dataBottom["fit"]
#fit = pd.get_dummies(dataBottom["fit"],prefix='fit', drop_first=True)
dataBottom = pd.concat([dataBottom,bustsize],axis=1)
dataBottom = pd.concat([dataBottom,category],axis=1)
dataBottom = pd.concat([dataBottom,rent],axis=1)
dataBottom = pd.concat([dataBottom,body],axis=1)
#dataBottom = pd.concat([dataBottom,fit],axis=1)
dataBottom.drop(['bust size', 'category','rented for','body type'],axis=1, inplace=True)





#Standarisation/normailsation 
from sklearn.preprocessing import MinMaxScaler

norm = MinMaxScaler().fit(dataDress.loc[:, 'weight':'age'])
dataDress.loc[:, 'weight':'age'] = norm.transform(dataDress.loc[:, 'weight':'age'])

norm = MinMaxScaler().fit(dataJacket.loc[:, 'weight':'age'])
dataJacket.loc[:, 'weight':'age'] = norm.transform(dataJacket.loc[:, 'weight':'age'])

norm = MinMaxScaler().fit(dataTop.loc[:, 'weight':'age'])
dataTop.loc[:, 'weight':'age'] = norm.transform(dataTop.loc[:, 'weight':'age'])

norm = MinMaxScaler().fit(dataBottom.loc[:, 'weight':'age'])
dataBottom.loc[:, 'weight':'age'] = norm.transform(dataBottom.loc[:, 'weight':'age'])


# print(dataDress)
# print(dataJacket)
# print(dataTop)
# print(dataBottom)


#split data 


from sklearn.model_selection import train_test_split

trainingDataDress, testDataDress = train_test_split(dataDress, test_size=0.25, random_state=42, stratify=fitDress)
trainingDataJacket, testDataJacket = train_test_split(dataJacket, test_size=0.25, random_state=42, stratify=fitJacket)
trainingDataTop, testDataTop = train_test_split(dataTop, test_size=0.25, random_state=42, stratify=fitTop)
trainingDataBottom, testDataBottom = train_test_split(dataBottom, test_size=0.25, random_state=42, stratify=fitBottom)


trainingInputDress = trainingDataDress.loc[:, 'weight':'body type_straight & narrow']
trainingOutputDress = trainingDataDress.loc[:, 'fit']
testInputDress = testDataDress.loc[:, 'weight':'body type_straight & narrow']
testOutputDress = testDataDress.loc[:, 'fit']

trainingOutputDress = ohe.fit_transform(trainingOutputDress.values.reshape(-1,1)).toarray()
testOutputDress = ohe.fit_transform(testOutputDress.values.reshape(-1,1)).toarray()

trainingInputJacket = trainingDataJacket.loc[:, 'weight':'body type_straight & narrow']
trainingOutputJacket = trainingDataJacket.loc[:, 'fit']
testInputJacket = testDataJacket.loc[:, 'weight':'body type_straight & narrow']
testOutputJacket = testDataJacket.loc[:, 'fit']

trainingOutputJacket = ohe.fit_transform(trainingOutputJacket.values.reshape(-1,1)).toarray()
testOutputJacket = ohe.fit_transform(testOutputJacket.values.reshape(-1,1)).toarray()

trainingInputTop = trainingDataTop.loc[:, 'weight':'body type_straight & narrow']
trainingOutputTop = trainingDataTop.loc[:, 'fit']
testInputTop = testDataTop.loc[:, 'weight':'body type_straight & narrow']
testOutputTop = testDataTop.loc[:, 'fit']

trainingOutputTop = ohe.fit_transform(trainingOutputTop.values.reshape(-1,1)).toarray()
testOutputTop = ohe.fit_transform(testOutputTop.values.reshape(-1,1)).toarray()

trainingInputBottom = trainingDataBottom.loc[:, 'weight':'body type_straight & narrow']
trainingOutputBottom = trainingDataBottom.loc[:, 'fit']
testInputBottom = testDataBottom.loc[:, 'weight':'body type_straight & narrow']
testOutputBottom = testDataBottom.loc[:, 'fit']

trainingOutputBottom = ohe.fit_transform(trainingOutputBottom.values.reshape(-1,1)).toarray()
testOutputBottom = ohe.fit_transform(testOutputBottom.values.reshape(-1,1)).toarray()


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

#selecting and training for groups of products

#dress group
# logr = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced'))
# logr.fit(trainingInputDress, trainingOutputDress)

# print('(Dress) Basic Logistic Regression Accuracy: ', accuracy_score(trainingOutputDress, logr.predict(trainingInputDress)))
# print('(Dress) Basic Logistic Regression Cross validation ', np.mean(cross_val_score(logr, trainingInputDress, trainingOutputDress, cv=5)))

# #confusion matrix log
# print('Confusion matrix for balanced classes(dress) - Logistic regression')
# print(confusion_matrix(trainingOutputDress.argmax(axis=1), logr.predict(trainingInputDress).argmax(axis=1)))

rf1 = RandomForestClassifier(criterion='gini',max_depth=76, class_weight='balanced')
rf1.fit(trainingInputDress, trainingOutputDress)


print('(Dress) Randomforest Accuracy: ', accuracy_score(trainingOutputDress, rf1.predict(trainingInputDress)))
print('(Dress) Randomforest Cross validation ', np.mean(cross_val_score(rf1, trainingInputDress, trainingOutputDress, cv=5)))

# print('Confusion matrix for balanced classes(Dress)- Random Forest')
# print(confusion_matrix(trainingOutputDress.argmax(axis=1), rf.predict(trainingInputDress).argmax(axis=1)))

# nb = OneVsRestClassifier(MultinomialNB())
# nb.fit(trainingInputDress, trainingOutputDress)
# print('(Dress) Multinomial NB Accuracy: ', accuracy_score(trainingOutputDress, nb.predict(trainingInputDress)))
# print('(Dress) Multinomial NB Cross validation ', np.mean(cross_val_score(nb, trainingInputDress, trainingOutputDress, cv=5)))
# print('Confusion matrix for balanced classes(Dress)- Multinomial NB')
# print(confusion_matrix(trainingOutputDress.argmax(axis=1), nb.predict(trainingInputDress).argmax(axis=1)))

# voting_clf = OneVsRestClassifier(VotingClassifier(estimators=[('logr', logr), ('RF', rf), ('NB', nb)], voting='soft'))
# voting_clf.fit(trainingInputDress,trainingOutputDress)


# print('(Dress) Voting Classifier Accuracy: ', accuracy_score(trainingOutputDress, voting_clf.predict(trainingInputDress)))
# print('(Dress) Voting Classifier Cross validation ', np.mean(cross_val_score(voting_clf, trainingInputDress, trainingOutputDress, cv=5)))


#jacket group
# logr = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced'))
# logr.fit(trainingInputJacket, trainingOutputJacket)

# print('(Jacket) Basic Logistic Regression Accuracy: ', accuracy_score(trainingOutputJacket, logr.predict(trainingInputJacket)))
# print('(Jacket) Basic Logistic Regression Cross validation ', np.mean(cross_val_score(logr, trainingInputJacket, trainingOutputJacket, cv=5)))

# #confusion matrix log
# print('Confusion matrix for balanced classes(Jacket)- Logistic regression')
# print(confusion_matrix(trainingOutputJacket.argmax(axis=1), logr.predict(trainingInputJacket).argmax(axis=1)))

rf2 = RandomForestClassifier(criterion='gini',max_depth=76, class_weight='balanced')
rf2.fit(trainingInputJacket, trainingOutputJacket)


print('(Jacket) Randomforest Accuracy: ', accuracy_score(trainingOutputJacket, rf2.predict(trainingInputJacket)))
print('(Jacket) Randomforest Cross validation ', np.mean(cross_val_score(rf2, trainingInputJacket, trainingOutputJacket, cv=5)))

# print('Confusion matrix for balanced classes(Jacket)- Random Forest')
# print(confusion_matrix(trainingOutputJacket.argmax(axis=1), rf.predict(trainingInputJacket).argmax(axis=1)))

# nb = OneVsRestClassifier(MultinomialNB())
# nb.fit(trainingInputJacket, trainingOutputJacket)
# print('(Jacket) Multinomial NB Accuracy: ', accuracy_score(trainingOutputJacket, nb.predict(trainingInputJacket)))
# print('(Jacket) Multinomial NB Cross validation ', np.mean(cross_val_score(nb, trainingInputJacket, trainingOutputJacket, cv=5)))
# print('Confusion matrix for balanced classes(Jacket)- Multinomial NB')
# print(confusion_matrix(trainingOutputJacket.argmax(axis=1), nb.predict(trainingInputJacket).argmax(axis=1)))

# voting_clf = OneVsRestClassifier(VotingClassifier(estimators=[('logr', logr), ('RF', rf), ('NB', nb)], voting='soft'))
# voting_clf.fit(trainingInputJacket,trainingOutputJacket)


# print('(Jacket) Voting Classifier Accuracy: ', accuracy_score(trainingOutputJacket, voting_clf.predict(trainingInputJacket)))
# print('(Jacket) Voting Classifier Cross validation ', np.mean(cross_val_score(voting_clf, trainingInputJacket, trainingOutputJacket, cv=5)))


# #top group
# logr = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced'))
# logr.fit(trainingInputTop, trainingOutputTop)

# print('(Top) Basic Logistic Regression Accuracy: ', accuracy_score(trainingOutputTop, logr.predict(trainingInputTop)))
# print('(Top) Basic Logistic Regression Cross validation ', np.mean(cross_val_score(logr, trainingInputTop, trainingOutputTop, cv=5)))

# #confusion matrix log
# print('Confusion matrix for balanced classes(top)- Logistic regression')
# print(confusion_matrix(trainingOutputTop.argmax(axis=1), logr.predict(trainingInputTop).argmax(axis=1)))

rf3 = RandomForestClassifier(criterion='gini',max_depth=77, class_weight='balanced')
rf3.fit(trainingInputTop, trainingOutputTop)


print('(Top) Randomforest Accuracy: ', accuracy_score(trainingOutputTop, rf3.predict(trainingInputTop)))
print('(Top) Randomforest Cross validation ', np.mean(cross_val_score(rf3, trainingInputTop, trainingOutputTop, cv=5)))

# print('Confusion matrix for balanced classes(Top)- Random Forest')
# print(confusion_matrix(trainingOutputTop.argmax(axis=1), rf.predict(trainingInputTop).argmax(axis=1)))

# nb = OneVsRestClassifier(MultinomialNB())
# nb.fit(trainingInputTop, trainingOutputTop)
# print('(Top) Multinomial NB Accuracy: ', accuracy_score(trainingOutputTop, nb.predict(trainingInputTop)))
# print('(Top) Multinomial NB Cross validation ', np.mean(cross_val_score(nb, trainingInputTop, trainingOutputTop, cv=5)))
# print('Confusion matrix for balanced classes(Top)- Multinomial NB')
# print(confusion_matrix(trainingOutputTop.argmax(axis=1), nb.predict(trainingInputTop).argmax(axis=1)))


# voting_clf = OneVsRestClassifier(VotingClassifier(estimators=[('logr', logr), ('RF', rf), ('NB', nb)], voting='soft'))
# voting_clf.fit(trainingInputTop,trainingOutputTop)


# print('(Top) Voting Classifier Accuracy: ', accuracy_score(trainingOutputTop, voting_clf.predict(trainingInputTop)))
# print('(Top) Voting Classifier Cross validation ', np.mean(cross_val_score(voting_clf, trainingInputTop, trainingOutputTop, cv=5)))



#bottom group
# logr = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced'))
# logr.fit(trainingInputBottom, trainingOutputBottom)

# print('(Bottom) Basic Logistic Regression Accuracy: ', accuracy_score(trainingOutputBottom, logr.predict(trainingInputBottom)))
# print('(Bottom) Basic Logistic Regression Cross validation ', np.mean(cross_val_score(logr, trainingInputBottom, trainingOutputBottom, cv=5)))

#  #confusion matrix log
# print('Confusion matrix for balanced classes(bottom)- Logistic regression')
# print(confusion_matrix(trainingOutputBottom.argmax(axis=1), logr.predict(trainingInputBottom).argmax(axis=1)))


rf4 = RandomForestClassifier(criterion='gini',max_depth=78, class_weight='balanced')
rf4.fit(trainingInputBottom, trainingOutputBottom)


print('(Bottom) Randomforest Accuracy: ', accuracy_score(trainingOutputBottom, rf4.predict(trainingInputBottom)))
print('(Bottom) Randomforest Cross validation ', np.mean(cross_val_score(rf4, trainingInputBottom, trainingOutputBottom, cv=5)))

# print('Confusion matrix for balanced classes(Bottom)- Random Forest')
# print(confusion_matrix(trainingOutputBottom.argmax(axis=1), rf.predict(trainingInputBottom).argmax(axis=1)))

# nb = OneVsRestClassifier(MultinomialNB())
# nb.fit(trainingInputBottom, trainingOutputBottom)
# print('(Bottom) Multinomial NB Accuracy: ', accuracy_score(trainingOutputBottom, nb.predict(trainingInputBottom)))
# print('(Bottom) Multinomial NB Cross validation ', np.mean(cross_val_score(nb, trainingInputBottom, trainingOutputBottom, cv=5)))

# print('Confusion matrix for balanced classes(bottom)- Multinomial NB')
# print(confusion_matrix(trainingOutputBottom.argmax(axis=1), nb.predict(trainingInputBottom).argmax(axis=1)))


# voting_clf = OneVsRestClassifier(VotingClassifier(estimators=[('logr', logr), ('RF', rf), ('NB', nb)], voting='soft'))
# voting_clf.fit(trainingInputBottom,trainingOutputBottom)


# print('(Bottom) Voting Classifier Accuracy: ', accuracy_score(trainingOutputBottom, voting_clf.predict(trainingInputBottom)))
# print('(Bottom) Voting Classifier Cross validation ', np.mean(cross_val_score(voting_clf, trainingInputBottom, trainingOutputBottom, cv=5)))


# In[18]:


# #dress
# rfPara = [{'max_depth': [78, 76, 77]}]

# grid_searchLog = GridSearchCV(rf1, rfPara, cv=5, return_train_score=True)
# grid_searchLog.fit(trainingInputDress, trainingOutputDress)

# print(grid_searchLog.best_params_)

# #jacket
# rfPara = [{'max_depth': [78, 76, 77]}]

# grid_searchLog = GridSearchCV(rf2, rfPara, cv=5, return_train_score=True)
# grid_searchLog.fit(trainingInputJacket, trainingOutputJacket)

# print(grid_searchLog.best_params_)

# #top
# rfPara = [{'max_depth': [78, 76, 77]}]

# grid_searchLog = GridSearchCV(rf3, rfPara, cv=5, return_train_score=True)
# grid_searchLog.fit(trainingInputTop, trainingOutputTop)

# print(grid_searchLog.best_params_)

# #bottom
# rfPara = [{'max_depth': [78, 76, 77]}]

# grid_searchLog = GridSearchCV(rf4, rfPara, cv=5, return_train_score=True)
# grid_searchLog.fit(trainingInputBottom, trainingOutputBottom)

# print(grid_searchLog.best_params_)

print('(Dress) Randomforest Test Accuracy: ', accuracy_score(testOutputDress, rf1.predict(testInputDress)))
print('(Dress) Randomforest Test Cross validation ', np.mean(cross_val_score(rf1, testInputDress, testOutputDress, cv=5)))

print('(Jacket) Randomforest Test Accuracy: ', accuracy_score(testOutputJacket, rf2.predict(testInputJacket)))
print('(Jacket) Randomforest Test Cross validation ', np.mean(cross_val_score(rf2, testInputJacket, testOutputJacket, cv=5)))

print('(Top) Randomforest Test Accuracy: ', accuracy_score(testOutputTop, rf3.predict(testInputTop)))
print('(Top) Randomforest Test Cross validation ', np.mean(cross_val_score(rf3, testInputTop, testOutputTop, cv=5)))

print('(Bottom) Randomforest Test Accuracy: ', accuracy_score(testOutputBottom, rf4.predict(testInputBottom)))
print('(Bottom) Randomforest Test Cross validation ', np.mean(cross_val_score(rf4, testInputBottom, testOutputBottom, cv=5)))


# In[ ]:




