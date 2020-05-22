#!/usr/bin/env python
# coding: utf-8

# In[214]:


import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor( 
           criterion='mae', 
           max_depth=8,
           max_leaf_nodes=None,
           min_impurity_split=None,
           min_samples_leaf=1,
           min_samples_split=2,
           min_weight_fraction_leaf=0.0,
           n_estimators= 500,
           n_jobs=-1,
           random_state=0,
           verbose=0,
           warm_start=False
)


# In[240]:


#bring in the six packs
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_resposta = pd.DataFrame()







# Seleciona somente linhas com valores nesses 4 quesitos a baixo na base de train
df_train = df_train.loc[
      (df_train['NU_NOTA_CN'].notnull())  & (df_train['NU_NOTA_CN'] != 0) & (df_train['NU_NOTA_CH'].notnull())      & (df_train['NU_NOTA_CH'] != 0) 
    & (df_train['NU_NOTA_LC'].notnull())  & (df_train['NU_NOTA_LC'] != 0) & (df_train['NU_NOTA_REDACAO'].notnull()) & (df_train['NU_NOTA_REDACAO'] != 0)    
]

df_test = df_test.loc[
      (df_test['NU_NOTA_CN'].notnull())  & (df_test['NU_NOTA_CN'] != 0) & (df_test['NU_NOTA_CH'].notnull())      & (df_test['NU_NOTA_CH'] != 0) 
    & (df_test['NU_NOTA_LC'].notnull())  & (df_test['NU_NOTA_LC'] != 0) & (df_test['NU_NOTA_REDACAO'].notnull()) & (df_test['NU_NOTA_REDACAO'] != 0)    
]





# In[216]:


features = [
    'NU_NOTA_CN',
    'NU_NOTA_CH',
    'NU_NOTA_LC',
    'NU_NOTA_REDACAO',
    'NU_NOTA_COMP1',
    'NU_NOTA_COMP2',
    'NU_NOTA_COMP3',
    'NU_NOTA_COMP4',
    'NU_NOTA_COMP5']
features_corr = [
    'NU_NOTA_MT',
    'NU_NOTA_CN',
    'NU_NOTA_CH',
    'NU_NOTA_LC',
    'NU_NOTA_REDACAO',
    'NU_NOTA_COMP1',
    'NU_NOTA_COMP2',
    'NU_NOTA_COMP3',
    'NU_NOTA_COMP4',
    'NU_NOTA_COMP5']


# In[217]:


print(df_train[features].isnull().sum())


# In[218]:


corr = df_train[features_corr].corr()
ax = plt.subplots(figsize=(11, 8))
sns.heatmap(corr,  annot=True, annot_kws={"size": 10})


# In[219]:


x0 = df_train['NU_NOTA_CN'].fillna(0)
x1 = df_test['NU_NOTA_CN'].fillna(0)

y0 = df_train['NU_NOTA_CH'].fillna(0)
y1 = df_test['NU_NOTA_CH'].fillna(0)

z0 = df_train['NU_NOTA_LC'].fillna(0)
z1 = df_test['NU_NOTA_LC'].fillna(0)

#sns.distplot(x0)
#sns.distplot(x1)

sns.distplot(y0)
sns.distplot(y1)

#sns.distplot(z0)
#sns.distplot(z1)

plt.legend(labels=['TRAIN', 'TEST'], ncol=2, loc='upper left');


# In[220]:


print(df_test[features].isnull().sum())


# In[221]:


df_test['NU_NOTA_COMP1'].fillna(0,inplace=True)
df_test['NU_NOTA_COMP2'].fillna(0,inplace=True)
df_test['NU_NOTA_COMP3'].fillna(0,inplace=True)
df_test['NU_NOTA_COMP4'].fillna(0,inplace=True)
df_test['NU_NOTA_COMP5'].fillna(0,inplace=True)


df_train['NU_NOTA_COMP1'].fillna(0,inplace=True)
df_train['NU_NOTA_COMP2'].fillna(0,inplace=True)
df_train['NU_NOTA_COMP3'].fillna(0,inplace=True)
df_train['NU_NOTA_COMP4'].fillna(0,inplace=True)
df_train['NU_NOTA_COMP5'].fillna(0,inplace=True)


# In[222]:


y_train = df_train['NU_NOTA_MT']
x_train = df_train[features]
x_test = df_test[features]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test)


# In[223]:


print(regressor.fit(x_train, y_train))


# In[241]:


y_pred_test = regressor.predict(x_test)
print (y_pred_test)
df_resposta['NU_INSCRICAO'] = df_test['NU_INSCRICAO']
df_resposta['NU_NOTA_MT'] = y_pred_test
df_resposta = df_resposta.loc[
      (df_resposta['NU_INSCRICAO'].notnull())]

#print(df_resposta['NU_NOTA_MT'])
y_pred_train = regressor.predict(x_train)
print (y_pred_test)
df_resposta.to_csv('answer1.csv', index=False, header=True, decimal=".")


# In[238]:


print('MAE:', metrics.mean_absolute_error(y_train, y_pred_train).round(8)  )
print('MSE:', metrics.mean_squared_error(y_train, y_pred_train).round(8) )  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)).round(8))


# In[ ]:





# In[ ]:




