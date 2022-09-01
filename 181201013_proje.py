# -*- coding: utf-8 -*-
"""
Created on Fri Jun  24 08:31:55 2022
181201013 Eren Toy
Bil470 Proje
@author: Eren TOY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression":LogisticRegression(),
    "Decision Tree":DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Cat Boost": CatBoostClassifier(),
    "XGBoost ": XGBClassifier()
}

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import metrics

cv = KFold(n_splits = 5, shuffle = True, random_state = 42)

def merges(df):
    relatives=[]
    for i in df["FoodCourt"].values.tolist():
        relatives.append(i)
    relatives1=[]
    for i in df["ShoppingMall"].values.tolist():
        relatives1.append(i) 
    relatives2=[]
    for i in df["Spa"].values.tolist():
        relatives2.append(i)
    relatives3=[]
    for i in df["VRDeck"].values.tolist():
        relatives3.append(i) 
    relatives4=[]
    for i in df["RoomService"].values.tolist():
        relatives4.append(i)     
    re=[]

    for i in range(0, len(relatives)):
        re.append(relatives[i]+relatives1[i]+relatives2[i]+relatives3[i]+relatives4[i])  
    df1=pd.DataFrame(re, columns=['Price Spent'])   
    df= pd.concat([df, df1], axis=1)
    return df

def model(models, results, X_train, y_train, X_test, y_test):
    for i, (model_name,models) in enumerate(models.items()):
        check  = cross_val_score(models, X_train, y_train, cv = cv, scoring = "accuracy",n_jobs= -1)
        print(model_name, check, "\n\n\n")
        models.fit(X_train, y_train)
        y_test_pred = models.predict(X_test)

        results = results.append(pd.DataFrame({'Model': model_name,
                                               'Accuracy': metrics.accuracy_score(y_test, y_test_pred) ,
                                               'F1 score':metrics.f1_score(y_test, y_test_pred),
                                               'ROC value': metrics.roc_auc_score(y_test, y_test_pred),
                                               'Score':models.score(X_test, y_test),
                                               'CV-score':check.mean()
                                               }, index=[0]), ignore_index = True)
    return results

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print(train_data.head())

print('Number of features in dataset:', train_data.shape[1])
print('Number of data in train dataset:', train_data.shape[0])
print('Number of data in test dataset:', test_data.shape[0])

features_description = {
    'PassengerId': "'gggg_pp' Id, 'gggg': group id, 'pp': personal id within group",
    'HomePlanet': "Hometown planet",
    'CryoSleep': "frozen sleep",
    'Cabin': "cabin number where the passenger is staying", 
    'Destination': "planet destination",
    'Age': 'age', 
    'VIP': "paid for special VIP service",
    'RoomService': "billed at luxury amenity",
    'FoodCourt': "billed at luxury amenity",
    'ShoppingMall': "billed at luxury amenity",
    'Spa': "billed at luxury amenity",
    'VRDeck': "billed at luxury amenity",
    'Name': "name",
    'Transported': "Target"
}

print(train_data.isnull().sum())
data = train_data.drop_duplicates()
print(train_data.shape)
print(data.shape)
# no dublicates

# Checking the correlation in heatmap
corr = train_data.corr()
plt.figure(figsize=(24,18))
sns.heatmap(corr, cmap="coolwarm", annot=True)
plt.show()


# veri işleme

# id's of test dat
idCol=test_data.PassengerId.to_numpy()
train_data.set_index('PassengerId', inplace=True)
test_data.set_index('PassengerId', inplace=True)

# drop misiing

data = train_data.dropna()
print(data.shape)

# 2087 datnesi gitti çok
# filling misiing values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns, index=train_data.index)
test_data = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns, index=test_data.index)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
print(train_data.isnull().sum())

from sklearn import preprocessing


print(train_data.head())


# split cabin to 3 new features
train_data[['Deck', 'Number', 'Side']] = train_data['Cabin'].str.split('/', expand=True)
test_data[['Deck', 'Number', 'Side']] = test_data['Cabin'].str.split('/', expand=True)

# one hot vector 
#train_data = pd.get_dummies(train_data, columns=['HomePlanet', 'Destination'])
#test_data = pd.get_dummies(test_data, columns=['HomePlanet', 'Destination'])

# type cast 
train_data.Transported = train_data.Transported.astype('int')
train_data.CryoSleep = train_data.CryoSleep.astype('int')
train_data.VIP = train_data.VIP.astype('int')
train_data.Age = train_data.Age.astype('int')
train_data.RoomService = train_data.RoomService.astype('int')
train_data.FoodCourt = train_data.FoodCourt.astype('int')
train_data.Spa = train_data.Spa.astype('int')
train_data.VRDeck = train_data.VRDeck.astype('int')


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

test_data.CryoSleep = test_data.CryoSleep.astype('int')
test_data.VIP = test_data.VIP.astype('int')
test_data.Age = test_data.Age.astype('int')
test_data.RoomService = test_data.RoomService.astype('int')
test_data.FoodCourt = test_data.FoodCourt.astype('int')
test_data.Spa = test_data.Spa.astype('int')
test_data.VRDeck = test_data.VRDeck.astype('int')

categorical_cols= ['HomePlanet','Destination', 'ShoppingMall','Deck','Side','Number']
for i in categorical_cols:
    print(i)
    arr=np.concatenate((train_data[i], test_data[i])).astype(str)
    le.fit(arr)
    train_data[i]=le.transform(train_data[i].astype(str))
    test_data[i]=le.transform(test_data[i].astype(str))


#drop cabin and name
train_data.drop(columns=['Cabin', 'Name'], inplace=True)
test_data.drop(columns=['Cabin', 'Name'], inplace=True)


# merge spendingss togehter
# train_data=merges(train_data)
# test_data=merges(test_data)

# train_data.drop(['RoomService', 'FoodCourt','ShoppingMall','Spa','VRDeck'], axis=1, inplace=True)
# test_data.drop(['RoomService', 'FoodCourt','ShoppingMall','Spa','VRDeck'], axis=1, inplace=True)

# train_data.drop(['VIP', 'ShoppingMall'], axis=1, inplace=True)
# test_data.drop(['VIP', 'ShoppingMall'], axis=1, inplace=True)

# splitting test and train data
from sklearn.model_selection import train_test_split
y = train_data['Transported']
x = train_data.drop(['Transported'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

results = pd.DataFrame(columns=['Model','Accuracy','F1 score','ROC value','Score','CV-score'])

# fit with k-fold cross validation
results = model(models, results, x_train, y_train, x_test, y_test)

print(results)

score = pd.DataFrame(data={'Model': results['Model'], 'CV-score': results['CV-score']})
plt.figure(figsize=(20, 10))
plot = sns.barplot(x="Model", y="CV-score", data=score, palette="magma")
plot.bar_label(plot.containers[0],fmt = "%.3f")
plt.title('Performance analysis of different classifiers based on Cross Validation Score')
plt.show()


# find efficient model
max_sc = score['CV-score'].max()
print(max_sc)
cnt = -1;

for sc in score['CV-score']:
    cnt+=1
    if max_sc == sc:
        break

# validation data test and csv
submission = pd.DataFrame(columns=["PassengerId","Transported"])
submission["PassengerId"] = idCol
submission.set_index('PassengerId')
submission["Transported"] = list(models.values())[cnt].predict(test_data).astype(bool)
submission.to_csv('submission.csv', index=False)