from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
import xgboost as xgb
import pandas as pd
df = pd.read_csv('trainedited.csv')
m=df.as_matrix()
print (m[0])
# split data into X and y
X = m[0:,1:15]
print (X[0])
Y = m[0:,15]
print Y[0]
# split data into train and test sets
seed = 7
test_size = 0
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
print X_test
print y_test
model = XGBClassifier(eta=0.5,min_child_weight=1,objective="binary:logistic" ,subsample=0.6,max_depth=10,n_estimators = 1000,learning_rate=0.06)
model.fit(X_train, y_train)
#make predictions for test data

y_pred = model.predict_proba(X_train)
print y_pred[1]
predictions = [round(value) for value in y_pred[0:,1]]
print predictions[1]
# for i in y_pred:
#     print i
# evaluate predictions
accuracy = accuracy_score(y_train, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
df2 = pd.read_csv('testedited.csv')
m1=df2.as_matrix()
X1 = m1[0:,2:16]
print (X1[0])
y_pred1 = model.predict_proba(X1)
#predictions1 = [round(value) for value in y_pred]
#print (y_pred1[1][1])
for k in y_pred1:
    print k[1]
import csv


with open('returns2.csv', 'wb') as f:
    writer = csv.writer(f)
    for val in y_pred1[0:,1]:
        writer.writerow([val])

