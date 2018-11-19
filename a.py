import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

df = pd.read_csv("data_train.csv")
df1= pd.read_csv("data_test.csv")
y = df.target
xr = zip(df.num1,df.num2,df.num3,df.num4,df.num5,df.num6,df.num7,df.num8,df.num9,df.num10,df.num11,df.num12,df.num13,df.num14,df.num15,df.num16,df.num17,df.num18,df.num19,df.num20,df.num21,df.num22,df.num23,df.der1,df.der2,df.der3,df.der4,df.der5,df.der6,df.der7,df.der8,df.der9,df.der10,df.der11,df.der12,df.der13,df.der14,df.der15,df.der16,df.der17,df.der18,df.der19,df.cat1,df.cat2,df.cat3,df.cat4,df.cat5,df.cat6,df.cat7,df.cat8,df.cat9,df.cat10,df.cat11,df.cat12,df.cat13,df.cat14)
xr=list(xr)
xr=np.nan_to_num(xr)
xt = zip(df1.num1,df1.num2,df1.num3,df1.num4,df1.num5,df1.num6,df1.num7,df1.num8,df1.num9,df1.num10,df1.num11,df1.num12,df1.num13,df1.num14,df1.num15,df1.num16,df1.num17,df1.num18,df1.num19,df1.num20,df1.num21,df1.num22,df1.num23,df1.der1,df1.der2,df1.der3,df1.der4,df1.der5,df1.der6,df1.der7,df1.der8,df1.der9,df1.der10,df1.der11,df1.der12,df1.der13,df1.der14,df1.der15,df1.der16,df1.der17,df1.der18,df1.der19,df1.cat1,df1.cat2,df1.cat3,df1.cat4,df1.cat5,df1.cat6,df1.cat7,df1.cat8,df1.cat9,df1.cat10,df1.cat11,df1.cat12,df1.cat13,df1.cat14)
xt=list(xt)
xt=np.nan_to_num(xt)
print('Reading done...')
sm=SMOTE()
xr,y=sm.fit_resample(xr,y)
x,y=shuffle(xr,y,random_state=0)
x_test=xt
x_train=x
y_train=y
clf =BaggingClassifier(svm.SVC(gamma='scale'),n_estimators=20,max_samples=1/20.0)
##clf=svm.SVC(kernel='rbf',C=1.0,gamma='scale') ##use this for better fitting if possible
print('Classifier defined...')
clf.fit(x_train,y_train)
print('Fitting done...')
y_pred=clf.predict(x_test)
print('Prediction done...')
id=df1.id
predict=pd.DataFrame({'id':id,'Prediction':y_pred})
predict.to_csv('predict.csv',index=False)
print('All done.')




