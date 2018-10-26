"""
Created on Sat Aug 18 11:32:18 2018

@author: Wesley
"""
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
#invite people for the Kaggle party
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
#warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import math
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, ElasticNet
from xgboost import XGBRegressor
from scipy.stats import norm, skew #for some 

from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import ensemble
from sklearn.svm import SVR

df_train = pd.read_csv(r"C:\House Prices Advanced Regression Techniques\train.csv", header=0)
df_pred = pd.read_csv(r"C:\House Prices Advanced Regression Techniques\test.csv", header=0)
copy_df_train = df_train.copy()
copy_df_pred = df_pred.copy()


#### Root Mean Square Logarithmic Error ###
def logRMSE(y_test, y_pred) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))

# In[]:
# Take a look at the data
print('Training Dataset: \n',df_train.head(5))
print('Test Dataset: \n',df_pred.head(5))
print('Test Dataset Features: \n',df_pred.columns)

target = 'SalePrice'
identifier = 'Id'
balanceamento_dataset = 0
plot = 0
tam_test = 0.3
Imputation = 1 #Caso Imputation=0, usamos o metodo fillna(0) no conjunto de teste
Exclude_Object_Features = 1 #Seleciona Somente as Features Numericas para treinamento
feature_transform = 1
Norm_Features = 1 #Normalizacao das Features de X_train antes do treinamento, e de X_test e X_pred com a normalizacao de X_train para predicoes




# In[]: PERGUNTAS IMPORTANTES 
print('\n O Dataset de Treino e Teste possuem as mesmas features?:',set(df_train.drop(target,1)) == set(df_pred))

# In[]: Outliers
##bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train[target], df_train[var]], axis=1)
if plot:
    data.plot.scatter(x=var, y=target, ylim=(0,800000));
##The two values with bigger 'GrLivArea' seem strange and they are not following the crowd. We can speculate why this is happening. Maybe they refer to agricultural area and that could explain the low price. I'm not sure about this but I'm quite confident that these two points are not representative of the typical case. Therefore, we'll define them as outliers and delete them.  
##The two observations in the top of the plot are those 7.something observations that we said we should be careful about. They look like two special cases, however they seem to be following the trend. For that reason, we will keep them.
##Deletar estes dois pontos aberrantes
df_train = df_train.drop(df_train.index[df_train.sort_values(by=['GrLivArea'],ascending=False)[:2].index])
#
#
##bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train[target], df_train[var]], axis=1)
if plot:
    data.plot.scatter(x=var, y=target, ylim=(0,800000));
##Deletar um Ponto Aberrante
df_train = df_train[~(df_train['TotalBsmtSF'] > 3000)]


# In[]: Dealing with missing data
train_Id = df_train[identifier]
pred_Id = df_pred[identifier]
train_target = df_train[target]
### Merge Train and Prediction Dataset into df_merge. Drop 'Id' and 'SalePrice' columns before merge
df_merge = pd.concat([df_train.drop([target,identifier],1),df_pred.drop(identifier,1)],axis=0)
qualitative_features = [f for f in df_merge.dropna().columns if df_merge.dropna().dtypes[f] == 'object'] #Lista de Features Qualitativas.
quantitative_features = [f for f in df_merge.dropna().columns if df_merge.dropna().dtypes[f] != 'object'] #Lista de Features Qualitativas.

# Numeric Features
print('\nIs there any NaN value of numeric features in the dataset before Imputing?:',df_merge[quantitative_features].isnull().sum().any())
df_merge[quantitative_features].isnull().sum().sort_values(ascending=False)
df_merge['LotFrontage'] = df_merge['LotFrontage'].fillna(0)
df_merge['GarageYrBlt'] = df_merge['GarageYrBlt'].fillna(0)
df_merge['MasVnrArea'] = df_merge['MasVnrArea'].fillna(0)

for i in quantitative_features:
    df_merge[i] = df_merge[i].fillna(df_merge[i].mean())
print('\nIs there any NaN value of numeric features in the dataset after Imputing?:',df_merge[quantitative_features].isnull().sum().any())

# Non-Numeric Features
print('\nIs there any NaN value of numeric features in the dataset before Imputing?:',df_merge[qualitative_features].isnull().sum().any())
df_merge[qualitative_features].isnull().sum().sort_values(ascending=False)
df_merge['PoolQC'] = df_merge['PoolQC'].fillna('None')
df_merge['MiscFeature'] = df_merge['MiscFeature'].fillna('None')
df_merge['Alley'] = df_merge['Alley'].fillna('None')
df_merge['BsmtQual'] = df_merge['BsmtQual'].fillna('None')
df_merge['BsmtCond'] = df_merge['BsmtCond'].fillna('None')
df_merge['BsmtExposure'] = df_merge['BsmtCond'].fillna('None')
df_merge['BsmtFinType1'] = df_merge['BsmtFinType1'].fillna('None')
df_merge['BsmtFinType2'] = df_merge['BsmtFinType2'].fillna('None')
df_merge['FireplaceQu'] = df_merge['FireplaceQu'].fillna('None')
df_merge['GarageType'] = df_merge['GarageType'].fillna('None')
df_merge['GarageFinish'] = df_merge['GarageFinish'].fillna('None')
df_merge['GarageCond'] = df_merge['GarageCond'].fillna('None')
df_merge['GarageQual'] = df_merge['GarageQual'].fillna('None')
df_merge['Fence'] = df_merge['Fence'].fillna('None')
df_merge['MasVnrType'] = df_merge['MasVnrType'].fillna('None')
df_merge["Functional"] = df_merge["Functional"].fillna("Typ")
df_merge.drop('Utilities',axis=1,inplace=True)
last_qlfeat = ['MSZoning','KitchenQual','Exterior2nd','Exterior1st','Electrical','SaleType']
qualitative_features = [f for f in df_merge.dropna().columns if df_merge.dropna().dtypes[f] == 'object'] #Lista de Features Qualitativas.
quantitative_features = [f for f in df_merge.dropna().columns if df_merge.dropna().dtypes[f] != 'object'] #Lista de Features Qualitativas.

for i in last_qlfeat:# Mode for the rest features
    df_merge[i] =  df_merge[i].fillna(df_merge[i].mode()[0])

print('\nIs there any NaN value  in the dataset after Imputing and get_dummies?:',df_merge.isnull().sum().any())



# Imputation and Feature Labeling of Categorical Variables With Dummies
df_merge = pd.get_dummies(data=df_merge, columns=qualitative_features)

# Feature Transform
train_target = np.log1p(train_target)


# In[]: Normalization
if Norm_Features:
    max_pesos = df_merge.max(axis=0)
    df_merge.loc[:,max_pesos != 0] = df_merge.loc[:,max_pesos != 0]/max_pesos[max_pesos != 0]

print('\n Is there any null value in the df_merge?:',df_merge.isnull().sum().any())


# In[]: Restore df_train and df_pred
df_train = df_merge[:len(df_train)]
df_train[target] = train_target
df_pred = df_merge[len(df_train):]



# In[]: Division X_train/X_test
df_train = shuffle(df_train) #shuffle data before division
train_target = df_train[target] # Just for code readibility
predictors = df_train.drop([target], axis=1)
X_train, X_test, y_train, y_test = train_test_split(predictors, 
                                                    train_target,
                                                    train_size=1-tam_test, 
                                                    test_size=tam_test, 
                                                    random_state=0)
X_pred = df_pred

# In[]: Lasso Model
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1, 3, 6, 10, 30, 60, 100], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)


y_train_las = lasso.predict(X_train)
y_test_las = lasso.predict(X_test)
las_prediction = lasso.predict(X_pred)

# In[]: Ridge Model
ridge = RidgeCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1, 3, 6, 10, 30, 60, 100])
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

y_train_rdg = ridge.predict(X_train)
y_test_rdg = ridge.predict(X_test)
rdg_prediction = ridge.predict(X_pred)


# In[]: XGBoost
xgb_model = XGBRegressor()

##### GRID SEARCH XGB #####
params = {'n_estimators':[1000],'learning_rate':[0.01,0.05],
'max_depth':[3,5]}
grid = GridSearchCV(xgb_model, params)
grid.fit(X_train, y_train)
y_train_gridXGB = grid.best_estimator_.predict(X_train)
y_test_gridXGB = grid.best_estimator_.predict(X_test)
gridXGB_prediction = grid.best_estimator_.predict(X_pred)
##### GRID SEARCH XGB #####


xgb_model = XGBRegressor(n_estimators=10000, learning_rate=0.05)

xgb_model.fit(X_train, y_train, early_stopping_rounds=5, 
             eval_set=[(X_test, y_test)], verbose=False)


        
y_train_xgb = xgb_model.predict(X_train)
y_test_xgb = xgb_model.predict(X_test)
xgb_prediction = xgb_model.predict(X_pred)




# In[]: ElasticNet Model
#elnr = ElasticNet(alpha=0.001,l1_ratio=0.3,max_iter=3000,n_jobs=-1)

elnr = ElasticNetCV(l1_ratio=[0.2,0.65],alphas=[0.001,0.005],n_jobs=-1,max_iter=3000)


elnr.fit(X_train,y_train)

y_train_eln = elnr.predict(X_train)
y_test_eln = elnr.predict(X_test)
eln_prediction = elnr.predict(X_pred)

# In[]: Plots and Results
if feature_transform:
    y_train = np.expm1(y_train)
    y_test = np.expm1(y_test)

   
    y_train_las = np.expm1(y_train_las)
    y_test_las = np.expm1(y_test_las)
    las_prediction = np.expm1(las_prediction)
    
    y_train_rdg = np.expm1(y_train_rdg)
    y_test_rdg = np.expm1(y_test_rdg)
    rdg_prediction = np.expm1(rdg_prediction)
    
    y_train_xgb = np.expm1(y_train_xgb)
    y_test_xgb = np.expm1(y_test_xgb)
    xgb_prediction = np.expm1(xgb_prediction)
    
    y_train_gridXGB = np.expm1(y_train_gridXGB)
    y_test_gridXGB = np.expm1(y_test_gridXGB)
    gridXGB_prediction = np.expm1(gridXGB_prediction)

    y_train_eln = np.expm1(y_train_eln)
    y_test_eln = np.expm1(y_test_eln)
    eln_prediction = np.expm1(eln_prediction)
else:
    pass
feature_transform=0

df_resultado_treinamento = pd.DataFrame(
        {'x_train':list(X_train.values),
         'y_train':list(y_train),
         'y_train_lasso':y_train_las,
         'y_train_ridge':y_train_rdg,
         'y_train_xgb':y_train_xgb,
         'y_train_gridXGB':y_train_gridXGB,
         'y_train_eln':y_train_eln,
                })
    
df_resultado_val = pd.DataFrame(
        {'x_val':list(X_test.values),
         'y_val':list(y_test),  
         'y_test_lasso':y_test_las,
         'y_test_ridge':y_test_rdg,
         'y_test_xgb':y_test_xgb,
         'y_test_gridXGB':y_test_gridXGB,
         'y_test_eln':y_test_eln
                })
    
plt.figure(1)
#plt.subplot(3, 1, 2)
df_resultado_treinamento = df_resultado_treinamento.sort_values('y_train',ascending=1)
plt.plot(list(df_resultado_treinamento['y_train']),'r-.',label='y_train')
plt.plot(list(df_resultado_treinamento['y_train_lasso']),'g-.',label='LassoTrain')
plt.plot(list(df_resultado_treinamento['y_train_ridge']),'y-.',label='RidgeTrain')
plt.plot(list(df_resultado_treinamento['y_train_xgb']),'k-.',label='XGB')
plt.legend()
#plt.xlabel('time (s)')
#plt.ylabel('Undamped')


#Plot Validacao
#plt.subplot(3, 1, 3)
plt.figure(2)
df_resultado_val = df_resultado_val.sort_values('y_val',ascending=1)
plt.plot(list(df_resultado_val['y_val']),'r-.',label='y_test')
plt.plot(list(df_resultado_val['y_test_lasso']),'g-.',label='LassoTest')
plt.plot(list(df_resultado_val['y_test_ridge']),'y-.',label='RidgeTest')
plt.plot(list(df_resultado_val['y_test_xgb']),'k-.',label='XGB')
plt.legend()
plt.show()


## Avaliacao do modelo
print("\nLasso logRMSE on Training set :", logRMSE(y_train,y_train_las))
print("Ridge logRMSE on Training set :", logRMSE(y_train,y_train_rdg))
print("XGB logRMSE on Training set :", logRMSE(y_train,y_train_xgb))
print("gridXGB logRMSE on Training set :", logRMSE(y_train,y_train_gridXGB))


print("\nLasso logRMSE on Test set :", logRMSE(y_test,y_test_las))
print("Ridge logRMSE on Test set :", logRMSE(y_test,y_test_rdg))
print("XGB logRMSE on Test set :", logRMSE(y_test,y_test_xgb))
print("gridXGB logRMSE on Training set :", logRMSE(y_test,y_test_gridXGB))

print("eln logRMSE on Test set :", logRMSE(y_test,y_test_eln))

print("StackedModel logRMSE on Test set :", logRMSE(y_test,(y_test_xgb+y_test_rdg+y_test_las + y_test_gridXGB
                                                            +y_test_eln)/(len(df_resultado_val.columns)-2)))

# In[]: Write to csv
df_saida = pd.DataFrame(
        {identifier:pred_Id,
        target: (xgb_prediction + las_prediction + rdg_prediction + gridXGB_prediction
                   +eln_prediction  )/(len(df_resultado_val.columns)-2)
                })
  
df_saida.to_csv(r'C:\House Prices Advanced Regression Techniques/answer.csv',index=False)