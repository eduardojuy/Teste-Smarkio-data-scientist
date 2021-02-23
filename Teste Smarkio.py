#!/usr/bin/env python
# coding: utf-8

# 1- Análise exploratória dos dados utilizando estatística descritiva e inferencial,
# considerando uma, duas e/ou mais variáveis;

# Importação das libs

# In[35]:


import pandas as pd 
import sweetviz as sv
import numpy as np
import sklearn
import matplotlib.pyplot as plt


# Chamando o arquivo do excel

# In[3]:


df = pd.read_excel('teste_smarkio_Lbs.xls','Análise_ML')


# Verificando quantas linhas tem dado nulo

# In[5]:


df.isnull().sum()


# Tratando os dados da coluna True_class(Todos que o true_class for NaN, considerar o valor da coluna Pred_class)

# In[16]:


values = {'True_class': df['Pred_class']}


# In[6]:


df.fillna(value=values,inplace = True)


# Verificando quantas linhas tem dado nulo(depois de rodar o comando para tratar os dados)

# In[83]:


df.isnull().sum()


# Trazendo todas as informações do DataFrame

# In[8]:


df


# Agrupando a coluna status pela contagem dos valores da coluna probabilidade

# In[119]:


status_group_probabi = df.groupby('status')['probabilidade'].value_counts()


# In[120]:


status_group_probabi


# In[96]:


status = df['status']


# Agrupando a coluna status com as informações da coluna True_class

# In[11]:


status_group_true =df.groupby('status')['True_class'].value_counts()


# In[98]:


status_group_true


# Fazendo a contagem do total da coluna status, agrupado por cada status.

# In[13]:


status_group_status = df.groupby('status')['status'].count()


# In[14]:


status_group_status


# Utilizando o comando describe()

# In[15]:


df.describe()


# 2 -Calcule o desempenho do modelo de classificação utilizando pelo menos três
# métricas;

# In[40]:


accuracy = sklearn.metrics.accuracy_score(df['True_class'], df['Pred_class'])


# In[115]:


accuracy


# In[37]:


from sklearn.metrics import classification_report


# In[39]:


print(classification_report,(df['True_class'], (df['Pred_class'])))


# In[60]:


from sklearn.metrics import confusion_matrix


# In[61]:


y_true = df['True_class']
y_pred = df['Pred_class']


# confusion_matrix

# In[62]:


confusion_matrix(y_true, y_pred)


# In[128]:


plt.plot(status,y_true)
plt.show()


# In[129]:


plt.plot(status,y_pred)
plt.show()

