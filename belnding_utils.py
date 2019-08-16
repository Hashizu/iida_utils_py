# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import re
import heapq
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import  LabelEncoder
from scipy.special import erfinv
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LinearRegression as LinReg
import gc
from numba.decorators import jit

np.random.seed(0)

home_path ='./drive/My Drive/SIGNATE/'

def mape(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def xgb_mape(preds, dtrain):
  labels = dtrain.get_label()
  return('mape', np.mean(np.abs((labels - preds) / (labels+1))))

# Training

n_model = 2
skf = KFold(n_splits=5, random_state=1128, shuffle=True)
xg_oof_val = np.zeros(len(train))
xg_oof_train = np.zeros(len(train))
lg_oof_val = np.zeros(len(train))
lg_oof_train = np.zeros(len(train))
xg_preds = np.zeros(len(test))
lg_preds = np.zeros(len(test))

X = np.array(train)
X = X.reshape(X.shape[0],X.shape[-1])
y = np.array(data_y)

logs = True
if logs :
  y = np.log(y)

i=0
lgb_params = {
    'task' : 'train',
    'boosting_type' : 'gbdt',
    'num_boost_round' : 8000,
    'objective' : 'gamma',
    'metric' : {'mape'},
    'early_stopping_rounds':500,
    'min_child_weight': 0.5,
    'learning_rate' : 0.01,
    'feature_fraction' : 0.3,
    'verbose' : 0,
    'max_depth': 5,
    'seed' : 625,
    'bagging_fraction' : 0.3,
    'feature_fraction_seed':124,
    'bagging_seed' : 124
}
xgb_params = {
    'n_estimators' : 8000 ,
    'max_depth': 6,
    'min_child_weight': 0.5,
    'learning_rate':0.01,
    'seed' : 625,
    'n_jobs':-1,
    'subsample':0.3,
    'objective':'reg:gamma',
    'colsample_bytree': 0.8
}


es_r = 500
n_verbose = 1000
callback_l = []

lgb_bool = True
xgb_bool = False


for train_pj, test_pj in skf.split(range(data_pj.max()+1)):
  train_index = [x in train_pj for x in data_pj]
  test1_num, test2_num = train_test_split(test_pj, test_size=0.5, random_state=11)  
  test1_index = [x in test1_num for x in data_pj]
  test2_index = [x in test2_num for x in data_pj]
  X_train, X_test1, X_test2 = X[train_index], X[test1_index], X[test2_index]
  y_train, y_test1, y_test2 = y[train_index], y[test1_index], y[test2_index]
  i=i+1
  train_data=lgb.Dataset(X_train, label=y_train)
  print('{} - a'.format(i))    

  if xgb_bool:
    clf = xgb.XGBRegressor(**xgb_params)
    clf.fit(X_train,y_train,
            eval_set=[(X_train, y_train), (X_test1, y_test1)],
            eval_metric = xgb_mape,
            early_stopping_rounds=es_r,
            verbose=n_verbose)
    xg_oof_val[test2_index] += clf.predict(X_test2) / n_model * 2
    xg_oof_train[train_index] += clf.predict(X_train)/(skf.n_splits-1)/n_model
    xg_preds += clf.predict(test)/skf.n_splits/2
  
  if lgb_bool:
    valid_data=lgb.Dataset(X_test1, label=y_test1)
    clf = lgb.train(lgb_params, train_set=train_data, valid_sets=valid_data, verbose_eval=1000, callbacks=callback_l)
    lg_oof_val[test2_index] += clf.predict(X_test2)/n_model*2
    lg_oof_train[train_index] += clf.predict(X_train)/(skf.n_splits-1) /n_model
    lg_preds += clf.predict(test)/(skf.n_splits)/2 
  
  
  print('{} - b'.format(i))

  if xgb_bool:
    clf = xgb.XGBRegressor(**xgb_params)
    clf.fit(X_train,y_train,
            eval_set=[(X_train, y_train), (X_test2, y_test2)],
            eval_metric = xgb_mape,
            early_stopping_rounds=es_r,
            verbose=n_verbose)
    xg_oof_val[test1_index] += clf.predict(X_test1) / n_model * 2
    xg_oof_train[train_index] += clf.predict(X_train)/(skf.n_splits-1)/n_model
    xg_preds += clf.predict(test)/skf.n_splits/2

  if lgb_bool:
    valid_data=lgb.Dataset(X_test2, label=y_test2)
    clf = lgb.train(lgb_params, train_set=train_data, valid_sets=valid_data, verbose_eval=1000, callbacks=callback_l)
    lg_oof_val[test1_index] += clf.predict(X_test1)/n_model*2
    tr_pre = clf.predict(X_train)
    lg_oof_train[train_index] += tr_pre/(skf.n_splits-1) /n_model 
    lg_preds += clf.predict(test)/(skf.n_splits)/2

    
if logs:
  lg_oof_train = pow(np.e,lg_oof_train)
  lg_oof_val = pow(np.e,lg_oof_val)
  lg_preds = pow(np.e,lg_preds)
  xg_oof_train = pow(np.e,xg_oof_train)
  xg_oof_val = pow(np.e,xg_oof_val)
  xg_preds = pow(np.e,xg_preds)
    
print()  
print('LGB train : ' + str(mape(data_y, lg_oof_train)))
print('LGB valid : ' + str(mape(data_y, lg_oof_val)))
print('XGB train : ' + str(mape(data_y, xg_oof_train)))
print('XGB valid : ' + str(mape(data_y, xg_oof_val)))

from scipy.optimize import minimize

# optimize mape ev

def objective_function(beta, X, Y):
      error = mape(Y,np.matmul(X,beta))
      return(error)


af_x = np.zeros((len(df),5))
af_x[:,0] = lg_oof_val
af_x[:,1] = xg_oof_val
af_x[:,2] = np.ones(len(df))*10000
# af_x[:,3] = (lg_oof_val - lg_oof_val.mean())/100
# af_x[:,4] = (xg_oof_val - xg_oof_val.mean())/100

af_pred = np.zeros((len(test),5))
af_pred[:,0] = lg_preds
af_pred[:,1] = xg_preds
af_pred[:,2] = np.ones(len(test))*10000
# af_pred[:,3] = (lg_preds - lg_oof_val.mean())/100
# af_pred[:,4] = (xg_preds - xg_oof_val.mean())/100
  
  
beta_init = np.array([1]*af_x.shape[1])
result = minimize(objective_function, beta_init, args=(af_x,data_y),
                  method='BFGS', options={'maxiter': 100})

beta_hat = result.x
opt_val = np.matmul(af_x,beta_hat)
print(beta_hat)
print(mape(data_y, lg_oof_val),mape(data_y, xg_oof_val) )
print('\t|')
print('\tv')
print(mape(data_y,opt_val))

opt_preds = np.matmul(af_pred,beta_hat)

X__ = X

# pseudo labeling
lgb_params = {
    'task' : 'train',
    'boosting_type' : 'gbdt',
    'num_boost_round' : 8000,
    'objective' : 'gamma',
    'metric' : {'rmse'},
    'early_stopping_rounds':500,
    'learning_rate' : 0.01,
    'verbose' : 0,
    'max_depth': 5,
    'seed' : 625,
}
X_train,X_test, y_train, y_test = train_test_split(X__, np.abs(opt_val - data_y)/data_y)
train_data=lgb.Dataset(X_train, label=y_train)
valid_data=lgb.Dataset(X_test, label=y_test)
clf_z = lgb.train(lgb_params, train_set=train_data, valid_sets=valid_data, verbose_eval=4000, callbacks=callback_l)


# plt.hist(clf_z.predict(test),bins=100);
test_resi = clf_z.predict(test)
resi_test_X = test[test_resi<0.07]
resi_test_y = opt_preds[test_resi<0.07]
resi_test_pj = data_test_pj[test_resi<0.07]

pse_X = np.vstack([X__,resi_test_X])
pse_y = np.hstack([data_y,resi_test_y])
pse_pj = np.hstack([data_pj, resi_test_pj])

# Training

train = pse_X
data_y2 = pse_y
data_pj2 = pse_pj

n_model = 2
skf = KFold(n_splits=5, random_state=1128, shuffle=True)
xg_oof_val = np.zeros(len(train))
xg_oof_train = np.zeros(len(train))
lg_oof_val = np.zeros(len(train))
lg_oof_train = np.zeros(len(train))
xg_preds = np.zeros(len(test))
lg_preds = np.zeros(len(test))

X = np.array(train)
X = X.reshape(X.shape[0],X.shape[-1])
y = np.array(data_y2)

logs = True
if logs :
  y = np.log(y)

i=0
lgb_params = {
    'task' : 'train',
    'boosting_type' : 'gbdt',
    'num_boost_round' : 8000,
    'objective' : 'gamma',
    'metric' : {'mape'},
    'early_stopping_rounds':500,
    'min_child_weight': 0.5,
    'learning_rate' : 0.01,
    'feature_fraction' : 0.3,
    'verbose' : 0,
    'max_depth': 5,
    'seed' : 625,
    'bagging_fraction' : 0.3,
    'feature_fraction_seed':124,
    'bagging_seed' : 124
}
xgb_params = {
    'n_estimators' : 8000 ,
    'max_depth': 6,
    'min_child_weight': 0.5,
    'learning_rate':0.01,
    'seed' : 625,
    'n_jobs':-1,
    'subsample':0.3,
    'objective':'reg:gamma',
    'colsample_bytree': 0.8
}


es_r = 500
n_verbose = 1000
callback_l = []

lgb_bool = True
xgb_bool = True


for train_pj, test_pj in skf.split(range(data_pj2.max()+1)):
  train_index = [x in train_pj for x in data_pj2]
  test1_num, test2_num = train_test_split(test_pj, test_size=0.5, random_state=11)  
  test1_index = [x in test1_num for x in data_pj2]
  test2_index = [x in test2_num for x in data_pj2]
  X_train, X_test1, X_test2 = X[train_index], X[test1_index], X[test2_index]
  y_train, y_test1, y_test2 = y[train_index], y[test1_index], y[test2_index]
  i=i+1
  train_data=lgb.Dataset(X_train, label=y_train)
  print('{} - a'.format(i))    

  if xgb_bool:
    clf = xgb.XGBRegressor(**xgb_params)
    clf.fit(X_train,y_train,
            eval_set=[(X_train, y_train), (X_test1, y_test1)],
            eval_metric = xgb_mape,
            early_stopping_rounds=es_r,
            verbose=n_verbose)
    xg_oof_val[test2_index] += clf.predict(X_test2) / n_model * 2
    xg_oof_train[train_index] += clf.predict(X_train)/(skf.n_splits-1)/n_model
    xg_preds += clf.predict(test)/skf.n_splits/2
  
  if lgb_bool:
    valid_data=lgb.Dataset(X_test1, label=y_test1)
    clf = lgb.train(lgb_params, train_set=train_data, valid_sets=valid_data, verbose_eval=1000, callbacks=callback_l)
    lg_oof_val[test2_index] += clf.predict(X_test2)/n_model*2
    lg_oof_train[train_index] += clf.predict(X_train)/(skf.n_splits-1) /n_model
    lg_preds += clf.predict(test)/(skf.n_splits)/2 
  
  
  print('{} - b'.format(i))

  if xgb_bool:
    clf = xgb.XGBRegressor(**xgb_params)
    clf.fit(X_train,y_train,
            eval_set=[(X_train, y_train), (X_test2, y_test2)],
            eval_metric = xgb_mape,
            early_stopping_rounds=es_r,
            verbose=n_verbose)
    xg_oof_val[test1_index] += clf.predict(X_test1) / n_model * 2
    xg_oof_train[train_index] += clf.predict(X_train)/(skf.n_splits-1)/n_model
    xg_preds += clf.predict(test)/skf.n_splits/2

  if lgb_bool:
    valid_data=lgb.Dataset(X_test2, label=y_test2)
    clf = lgb.train(lgb_params, train_set=train_data, valid_sets=valid_data, verbose_eval=1000, callbacks=callback_l)
    lg_oof_val[test1_index] += clf.predict(X_test1)/n_model*2
    tr_pre = clf.predict(X_train)
    lg_oof_train[train_index] += tr_pre/(skf.n_splits-1) /n_model 
    lg_preds += clf.predict(test)/(skf.n_splits)/2

    
if logs:
  lg_oof_train = pow(np.e,lg_oof_train)
  lg_oof_val = pow(np.e,lg_oof_val)
  lg_preds = pow(np.e,lg_preds)
  xg_oof_train = pow(np.e,xg_oof_train)
  xg_oof_val = pow(np.e,xg_oof_val)
  xg_preds = pow(np.e,xg_preds)
    
print()  
print('LGB train : ' + str(mape(data_y, lg_oof_train[:len(data_y)])))
print('LGB train : ' + str(mape(data_y, lg_oof_val[:len(data_y)])))
print('XGB train : ' + str(mape(data_y, xg_oof_train[:len(data_y)])))
print('XGB valid : ' + str(mape(data_y, xg_oof_val[:len(data_y)])))

# second optimazation

af_x = np.zeros((len(goto),5))
af_x[:,0] = lg_oof_val[:len(goto)]
af_x[:,1] = xg_oof_val[:len(goto)]
af_x[:,2] = np.ones(len(goto))*100000
af_x[:,3] = temp_val

# af_x[:,3] = (lg_oof_val[:len(goto)] - lg_oof_val.mean())/100
# af_x[:,4] = (xg_oof_val[:len(goto)] - xg_oof_val.mean())/100

af_pred = np.zeros((len(ev_goto),5))
af_pred[:,0] = lg_preds
af_pred[:,1] = xg_preds
af_pred[:,2] = np.ones(len(ev_goto))*100000
af_pred[:,3] = temp_d

# af_pred[:,3] = (lg_preds - lg_oof_val.mean())/100
# af_pred[:,4] = (xg_preds - xg_oof_val.mean())/100
  
  
beta_init = np.array([1]*af_x.shape[1])
result = minimize(objective_function, beta_init, args=(af_x,data_y),
                  method='BFGS', options={'maxiter': 100})

beta_hat = result.x
opt_val = np.matmul(af_x,beta_hat)
print(beta_hat)
print(mape(data_y, lg_oof_val[:len(goto)]),mape(data_y, xg_oof_val[:len(goto)]) )
print('\t|')
print('\tv')
print(mape(data_y,opt_val))

opt_preds2 = np.matmul(af_pred,beta_hat)


## after Analysis
"""

from plotly.offline import iplot
import plotly.graph_objs as go

def enable_plotly_in_cell():
  import IPython
  from plotly.offline import init_notebook_mode
  display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
  init_notebook_mode(connected=False)
  
enable_plotly_in_cell()
data_plot = [go.Bar( 
            x=list(data_.columns[3:]),
            y=list(clf.feature_importance()))]
# data_plot = [go.Bar(
#             y=list(genba.isna().sum()),
#             x=list(genba.columns))]
iplot(data_plot, filename='basic-bar')




