import datetime

import numpy as np
import pandas as pd
from pandas import to_datetime
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR
from sklearn import tree
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from lable import trace_id

# OrdetId,TravelTime,Longtitude,Latitude,Speed,Direction,Trace,ETA
data_df = pd.read_csv("./data_processed/all.csv")
# X - travelTime lon lat speed direction traceid
train_X = np.concatenate((
    np.array(data_df.loc[:,("TravelTime","Longtitude","Latitude","Speed","Direction")]),
    np.array([trace_id[t] for t in data_df["Trace"]]).reshape((-1,1)),
), axis=1)

train_Y = np.array(data_df["ETA"])
print("Read data done.")
clf = make_pipeline(StandardScaler(),tree.DecisionTreeRegressor())

cv_results = cross_validate(clf,train_X,train_Y,cv=4,n_jobs=2)

sorted(cv_results.keys())
print(cv_results)

clf.fit(train_X, train_Y)
print("fit done.")
joblib.dump(clf,"./cache/clf")
