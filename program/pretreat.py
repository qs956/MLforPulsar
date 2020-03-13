import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

seed = 0
k_fold=2

def input():
	df = pd.read_csv('out.csv')
	title = list(df.columns.values)
	df = df[~np.isnan(df).any(axis=1)]
	df = df.values
	return df,title

def normalize(xtrain,xtest):
	mu = np.mean(xtrain, axis = 0)
	sigma = np.std(xtrain, axis = 0)
	xtrain_s = xtrain - mu
	xtrain_s /= sigma
	xtest_s = xtest - mu
	xtest_s /= sigma
	return xtrain_s,xtest_s

kf = StratifiedKFold(k_fold,shuffle=True)
smote = SMOTE(ratio=1.0)

def split_data(X,y):
	for tr, te in  kf.split(X,y):
		xtrain, xtest = X[tr], X[te]
		ytrain, ytest = y[tr], y[te]
		sxtrain, sytrain = smote.fit_sample(xtrain,ytrain)
	return xtrain, xtest, ytrain, ytest,sxtrain, sytrain

def kfoldsampling(X,y):
	np.random.seed(seed)
	xtrain, xtest, ytrain, ytest, sxtrain, sytrain=split_data(X,y)
	#no sample
	xtrain_s,xtest_s=normalize(xtrain,xtest)
	np.save('result/train_set/xtrain',xtrain_s)
	np.save('result/train_set/ytrain',ytrain)
	np.save('result/test_set/xtest',xtest_s)
	np.save('result/test_set/ytest',ytest)
	#SMOTE
	np.random.seed(seed)
	sxtrain_s,sxtest_s=normalize(sxtrain,xtest)
	np.save('result/train_set/sxtrain',sxtrain_s)
	np.save('result/train_set/sytrain',sytrain)
	np.save('result/test_set/sxtest',sxtest_s)
	np.save('result/test_set/sytest',ytest)
	#over sample
	np.random.seed(seed)
	ros = RandomOverSampler(random_state=0)
	oxtrain, oytrain = ros.fit_sample(xtrain, ytrain)
	oxtrain_s,oxtest_s=normalize(oxtrain,xtest)
	np.save('result/train_set/oxtrain',oxtrain_s)
	np.save('result/train_set/oytrain',oytrain)
	np.save('result/test_set/oxtest',oxtest_s)
	np.save('result/test_set/oytest',ytest)
	#under sample
	np.random.seed(seed)
	rus = RandomUnderSampler(random_state=0)
	uxtrain, uytrain = rus.fit_sample(xtrain, ytrain)
	uxtrain_s,uxtest_s=normalize(uxtrain,xtest)
	np.save('result/train_set/uxtrain',uxtrain_s)
	np.save('result/train_set/uytrain',uytrain)
	np.save('result/test_set/uxtest',uxtest_s)
	np.save('result/test_set/uytest',ytest)
