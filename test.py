# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 22:18:26 2022

@author: Zheng Gu
"""
#%% Inputs
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

feature_descriptions = pd.read_csv('train_data/feature_descriptions.tsv', sep='\t')
features_train = pd.read_csv('train_data/features_train.tsv', sep='\t')
tfbs_score_BC217_train = pd.read_csv('train_data/tfbs_score_BC217_train.tsv', sep='\t')
tfbs_score_YJF153_train = pd.read_csv('train_data/tfbs_score_YJF153_train.tsv', sep='\t')
features_test = pd.read_csv('test_data/features_test.tsv', sep='\t')
tfbs_score_BC217_test = pd.read_csv('test_data/tfbs_score_BC217_test.tsv', sep='\t')
tfbs_score_YJF153_test = pd.read_csv('test_data/tfbs_score_YJF153_test.tsv', sep='\t')

#%% Feature Engineering
def features_process(features, tfbs_score_BC217, tfbs_score_YJF153):
    data = features.merge(tfbs_score_BC217, how='left', on='gene')
    data = data.merge(tfbs_score_YJF153, how='left', on='gene')
    data.drop(data.columns[[2,3,10,13,16,52,53,54,55,56,57,60,61,62,63,71,72,73,75,76,77]], axis=1, inplace=True)
    data.drop(columns=['LOC_BC217','LOC_YJF153'], inplace=True)
    data = data.fillna(0)
    one_hot_data = pd.get_dummies(data['chromosome'])
    data.drop(columns='chromosome', inplace=True)
    data = pd.concat([data, one_hot_data], axis=1)
    return data
 
train_data = features_process(features_train, tfbs_score_BC217_train, tfbs_score_YJF153_train)

y_all = train_data['DIFF']
X_all = train_data.drop(columns=['SGD','DIFF'])
selector = SelectPercentile(f_regression, percentile=90)
X_new = selector.fit_transform(X_all, y_all)

#%% Modeling
def MSE_output(model):
    model.fit(X_train, y_train)
    y_fitted = model.predict(X_train)
    training_error = mean_squared_error(y_fitted, y_train)
    print("Training MSE: {}\n".format(training_error))
    
    y_cv_fitted = model.predict(X_cv)
    cv_error = mean_squared_error(y_cv_fitted, y_cv)
    print("Cross Validation MSE: {}\n".format(cv_error))

    return [training_error, cv_error]

X_train, X_cv, y_train, y_cv = train_test_split(X_new, y_all, test_size=0.2, random_state=1)

#%% Model Selection
#(1) Linear Regression
from sklearn import linear_model as lm
model1 = lm.LinearRegression(fit_intercept=True)
model1_train_err, model1_cv_err = MSE_output(model1)

#(2) Lasso Regression
model2 = lm.Lasso(normalize=True, tol=1e-2)
model2_train_err, model2_cv_err = MSE_output(model2)

#(3) XGBoost
import xgboost as xgb
model3 = xgb.XGBRegressor(learning_rate=0.1, n_estimators=20, max_depth=10, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_alpha=0, reg_lambda=0)
model3_train_err, model3_cv_err = MSE_output(model3)

#(4) CatBoost
import catboost as cbt
model4 = cbt.CatBoostRegressor(n_estimators=16, depth=2, learning_rate=0.5, logging_level='Silent')
model4_train_err, model4_cv_err = MSE_output(model4)

model = model3

#%% PyTorch
import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1
model5 = LinearRegressionModel(input_dim, output_dim)

epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(model5.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

#%% Make Predictions
test_data = features_process(features_test, tfbs_score_BC217_test, tfbs_score_YJF153_test)
test_SGD = test_data['SGD']
test_data.drop(columns=['SGD','DIFF'], inplace=True)
X_test = selector.transform(test_data)
y_test = model.predict(X_test)
prediction = pd.DataFrame({'DIFF':y_test, 'SGD':test_SGD})
prediction.to_csv('prediction/prediction.csv', index=False)
