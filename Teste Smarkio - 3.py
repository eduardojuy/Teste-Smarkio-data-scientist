#!/usr/bin/env python
# coding: utf-8

# 3-Crie um classificador que tenha como output se os dados com status igual a
# revision estão corretos ou não (Sugestão : Técnica de cross-validation K-fold);

# In[129]:


from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import datasets
from sklearn.svm import SVC


# In[5]:


df = pd.read_excel('teste_smarkio_Lbs.xls','Análise_ML')


# In[6]:


df


# In[7]:


df.isnull().sum()


# In[9]:


values = {'True_class': df['Pred_class']}


# In[10]:


df.fillna(value=values,inplace = True)


# In[62]:


df.isnull().sum()


# In[154]:


drop_status = df.drop(columns=['status'])
status = df['status']


# In[52]:


from sklearn.model_selection import train_test_split


# In[107]:


x_train, x_test, y_train, y_test = train_test_split(drop_status,status,test_size=0.3)


# In[136]:


svm = SVC(gamma='auto')
svm.fit(x_train,y_train)


# In[142]:


y_svm_p = svm.predict(x_test)


# In[139]:


scores_svm = []
scores_svm = cross_val_score(svm, x_test, y_test, cv=2)


# In[150]:


print('Acurácia do Classificador: {:0.4f}'.format(scores_svm.mean()))


# In[144]:


from sklearn.metrics import classification_report


# In[151]:


print("Métricas de Avaliação do Classifcador:\n")
print(classification_report(y_test,y_svm_p))


# In[ ]:




