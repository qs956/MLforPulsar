import numpy as np
import time
from sklearn.metrics import *

part_name = ('Recall','Precision','F1','FN','FP','Train Time(s)','Test Time(s)')
seed = 0

def performance(clf,xtrain,ytrain,xtest,ytest,k):
    np.random.seed(seed)
    time1 = time.time()
    clf.fit(xtrain[:,k],ytrain)
    time2 = time.time()
    y_pred = clf.predict(xtest[:,k])
    time3 = time.time()
    y_pro = clf.predict_proba(xtest[:,k])
    recall = recall_score(ytest,y_pred)
    precision = precision_score(ytest,y_pred)
    f1 = f1_score(ytest,y_pred)
    FN = sum(y_pred[ytest==1] == 0)
    FP = sum(y_pred[ytest==0] == 1)
    return y_pred,y_pro,float(round(recall,2)),float(round(precision,2)),float(round(f1,2)),int(FN),int(FP),float(round(time2-time1,3)),float(round(time3-time2,3))

def performance_rfe(clf,xtrain,ytrain,xtest,ytest):
    np.random.seed(seed)
    time1 = time.time()
    clf.fit(xtrain,ytrain)
    time2 = time.time()
    y_pred = clf.predict(xtest)
    time3 = time.time()
    y_pro = clf.predict_proba(xtest)
    recall = recall_score(ytest,y_pred)
    precision = precision_score(ytest,y_pred)
    f1 = f1_score(ytest,y_pred)
    FN = sum(y_pred[ytest==1] == 0)
    FP = sum(y_pred[ytest==0] == 1)
    return clf.support_,y_pred,y_pro,float(round(recall,2)),float(round(precision,2)),float(round(f1,2)),int(FN),int(FP),float(round(time2-time1,3)),float(round(time3-time2,3))

def performance_rfc(clf,xtrain,ytrain,xtest,ytest):
    np.random.seed(seed)
    time1 = time.time()
    clf.fit(xtrain,ytrain)
    time2 = time.time()
    y_pred = clf.predict(xtest)
    time3 = time.time()
    y_pro = clf.predict_proba(xtest)
    recall = recall_score(ytest,y_pred)
    precision = precision_score(ytest,y_pred)
    f1 = f1_score(ytest,y_pred)
    FN = sum(y_pred[ytest==1] == 0)
    FP = sum(y_pred[ytest==0] == 1)
    return y_pred,y_pro,clf.feature_importances_,float(round(recall,2)),float(round(precision,2)),float(round(f1,2)),int(FN),int(FP),float(round(time2-time1,3)),float(round(time3-time2,3))

def PR_curve(ytest,y_proba):
    precision, recall, _ = precision_recall_curve(ytest,y_proba[:,1])
    area = np.trapz(precision[::-1],recall[::-1],axis = 0)
    return recall,precision,area