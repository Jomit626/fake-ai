import datetime

import numpy as np
import pandas as pd
from pandas import to_datetime
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from lable import trace_id

clf = joblib.load("./cache/clf")
# loadingOrder,timestamp,longitude,latitude,speed,direction,carrierName,vesselMMSI,onboardDate,TRANSPORT_TRACE
test_data_df = pd.read_csv("./dataset/A_testData0531.csv")
df_timestamp_dt = to_datetime(test_data_df.loc[:, "timestamp"], format=r"%Y-%m-%dT%H:%M:%S.%fZ")
df_onboard_dt = to_datetime(test_data_df.loc[:, "onboardDate"], format=r"%Y/%m/%d  %H:%M:%S")

print("Read data done.")

test_X = np.concatenate((
    np.array([dt.total_seconds() for dt in (df_timestamp_dt - df_onboard_dt)]).reshape((-1,1)),
    np.array(test_data_df.loc[:,("longitude","latitude","speed","direction")]),
    np.array([trace_id[t] for t in test_data_df["TRANSPORT_TRACE"]]).reshape((-1,1)),
), axis=1)

test_Y = clf.predict(test_X)

test_data_df['creatDate'] = datetime.datetime.now().strftime(r'%Y/%m/%d  %H:%M:%S')
test_data_df["ETA"] = pd.Series([ (dt + pd.Timedelta(seconds=eta)).strftime(r'%Y/%m/%d  %H:%M:%S') for eta,dt in zip(test_Y,df_onboard_dt)])

result = test_data_df[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA', 'creatDate']]

result.to_csv(r"./result/result.csv",index=False)
