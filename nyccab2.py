import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")
use=test_df["id"]
train_df['pickup_datetime'] = pd.to_datetime(train_df.pickup_datetime)
test_df['pickup_datetime'] = pd.to_datetime(test_df.pickup_datetime)
train_df.loc[:, 'pickup_date'] = train_df['pickup_datetime'].dt.date
test_df.loc[:, 'pickup_date'] = test_df['pickup_datetime'].dt.date
train_df['dropoff_datetime'] = pd.to_datetime(train_df.dropoff_datetime)

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
train_df.loc[:, 'distance_haversine'] = haversine_array(train_df['pickup_latitude'].values, train_df['pickup_longitude'].values, train_df['dropoff_latitude'].values, train_df['dropoff_longitude'].values)
test_df.loc[:, 'distance_haversine'] = haversine_array(test_df['pickup_latitude'].values, test_df['pickup_longitude'].values, test_df['dropoff_latitude'].values, test_df['dropoff_longitude'].values)    
    
train_df.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(train_df['pickup_latitude'].values, train_df['pickup_longitude'].values, train_df['dropoff_latitude'].values, train_df['dropoff_longitude'].values)
test_df.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(test_df['pickup_latitude'].values, test_df['pickup_longitude'].values, test_df['dropoff_latitude'].values, test_df['dropoff_longitude'].values)

train_df.loc[:, 'direction'] = bearing_array(train_df['pickup_latitude'].values, train_df['pickup_longitude'].values, train_df['dropoff_latitude'].values, train_df['dropoff_longitude'].values)
test_df.loc[:, 'direction'] = bearing_array(test_df['pickup_latitude'].values, test_df['pickup_longitude'].values, test_df['dropoff_latitude'].values, test_df['dropoff_longitude'].values)

train_df['Hour'] = train_df['pickup_datetime'].dt.hour
test_df['Hour'] = test_df['pickup_datetime'].dt.hour
train_df['Minutes'] = train_df['pickup_datetime'].dt.minute
test_df['Minutes'] = test_df['pickup_datetime'].dt.minute
train_df=train_df.drop(["direction","pickup_date","vendor_id","id",'pickup_datetime','dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)
test_df=test_df.drop(["direction","pickup_date","vendor_id","id",'pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)

combine = [train_df, test_df]
for dataset in combine:
    dataset['store_and_fwd_flag'] = dataset['store_and_fwd_flag'].map( {'N': 0, 'Y': 1} ).astype(int)
print("hello11")
x_train = train_df.drop("trip_duration", axis=1)
y_train = train_df["trip_duration"]
x_test = test_df

dtrain = xgb.DMatrix(x_train, label=y_train)
#
dtest = xgb.DMatrix(x_test)
#print("hello1")
#
xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 
            'max_depth': 6,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain)
#print("hello")
#y_pred = model.predict(dtest)    
#rf=RandomForestRegressor(n_estimators=50,max_depth=50,min_samples_split=10)
#rf.fit(x_train,y_train)
#print("hello111")
y_pred=model.predict(x_test)
submission = pd.DataFrame({
        "id": use,
        "trip_duration": y_pred
    })
#submission.to_csv('submission7.csv', index=False)
print(submission)
